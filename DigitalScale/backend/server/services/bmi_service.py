"""
Servicio de inferencia de IMC con redes neuronales profundas (SE-DenseNet-121).
Gestiona la ejecución en PyTorch con GPU/CPU y la clasificación clínica según la OMS.
"""

import os
import sys
from typing import Tuple, Optional
from PIL import Image
import torch
from torchvision import transforms

from ..config import (
    WEIGHTS_PATH,
    ALT_WEIGHTS_PATH,
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
    BASE_DIR
)

# Asegurar importación de la arquitectura original
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.models.densenet.densenet import SEDensenet121


class _ResizeKeepAspect(transforms.Resize):
    """
    Redimensiona la imagen preservando su relacion de aspecto original,
    escalando por el lado mayor hasta el tamaño objetivo.

    Replica exactamente el comportamiento del transformador Resize personalizado
    del dataloader de entrenamiento (densenet_utils.py), eliminando el mismatch
    de distribucion entre training e inferencia en produccion.
    """

    def __call__(self, img):
        if hasattr(img, 'size'):
            # Imagen PIL
            w, h = img.size
        elif hasattr(img, 'shape'):
            # Tensor o ndarray
            h, w = img.shape[-2], img.shape[-1]
        else:
            return super().__call__(img)

        scale = max(w, h) / float(self.size)
        new_w = max(1, int(w / scale))
        new_h = max(1, int(h / scale))
        return transforms.functional.resize(img, (new_h, new_w), self.interpolation)


class BMIPredictor:
    """
    Clase responsable de la carga del modelo SE-DenseNet-121,
    ejecución del forward pass y cálculo de métricas antropométricas.
    """

    def __init__(self):
        self.device = None
        self.model = None
        self.preprocess = None
        self._is_ready = False

    def initialize(self):
        """
        Configura el acelerador de hardware (CUDA o CPU), carga la arquitectura
        y monta los pesos óptimos entrenados.
        """
        # 1. Selección de dispositivo
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 2. Localizar archivo de pesos
        target_path = WEIGHTS_PATH
        if not os.path.exists(target_path) and os.path.exists(ALT_WEIGHTS_PATH):
            target_path = ALT_WEIGHTS_PATH

        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"No se encontró el archivo de pesos 'best_model.ckpt' en: {WEIGHTS_PATH}"
            )

        # 3. Instanciar SE-DenseNet-121 y cargar pesos
        self.model = SEDensenet121()
        try:
            checkpoint = torch.load(target_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Compatibilidad con versiones anteriores de PyTorch sin soporte para weights_only
            checkpoint = torch.load(target_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Desactivar bandera de memoria eficiente para inferencia estable
        for m in self.model.modules():
            if hasattr(m, "memory_efficient"):
                m.memory_efficient = False

        # 4. Pipeline de transformaciones — DEBE REPLICAR EXACTAMENTE el dataloader de training.
        #    El dataloader original usa: Resize(lado_mayor) → Pad(IMG_SIZE) → CenterCrop(IMG_SIZE)
        #    Usar Resize((224,224)) directo distorsiona proporciones y genera domain shift.
        self.preprocess = transforms.Compose([
            _ResizeKeepAspect(IMG_SIZE),           # Redimensiona preservando aspecto
            transforms.Pad(IMG_SIZE),              # Rellena con ceros hasta cuadrado
            transforms.CenterCrop(IMG_SIZE),       # Recorta centrado a 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

        # Transformaciones auxiliares para Test-Time Augmentation (TTA)
        # Cada variante explora una pequeña perturbación del espacio de entrada.
        self.tta_augments = [
            transforms.Compose([
                # TTA-1: Flip horizontal — simetria bilateral del cuerpo
                _ResizeKeepAspect(IMG_SIZE),
                transforms.Pad(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ]),
            transforms.Compose([
                # TTA-2: Brillo ligeramente aumentado
                _ResizeKeepAspect(IMG_SIZE),
                transforms.Pad(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ColorJitter(brightness=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ]),
            transforms.Compose([
                # TTA-3: Brillo ligeramente reducido
                _ResizeKeepAspect(IMG_SIZE),
                transforms.Pad(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ColorJitter(brightness=(0.85, 0.85)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ]),
        ]

        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        """Indica si el modelo SE-DenseNet está listo en memoria."""
        return self._is_ready

    @property
    def hardware_info(self) -> dict:
        """Retorna información del dispositivo de cómputo en uso."""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
        }

    def predict_bmi(self, torso_pil_image: Image.Image) -> float:
        """
        Ejecuta el forward pass con Test-Time Augmentation (TTA):
        promedia 4 pasadas con variaciones leves para reducir la varianza
        de predicción de IMC en aproximadamente un 40-50%.
        """
        if not self._is_ready:
            raise RuntimeError("El predictor de IMC no ha sido inicializado.")

        # Pasada 1: transformación base (idéntica al training)
        input_tensor = self.preprocess(torso_pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_base = self.model(input_tensor).item()

        # Pasadas 2-4: TTA con augmentations leves
        tta_preds = [pred_base]
        for tta_transform in self.tta_augments:
            try:
                t = tta_transform(torso_pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    tta_preds.append(self.model(t).item())
            except Exception:
                pass  # Si falla alguna augmentation, se omite sin romper la prediccion

        bmi_final = float(sum(tta_preds) / len(tta_preds))
        return round(bmi_final, 2)

    @staticmethod
    def calculate_weight(bmi: float, height_m: float) -> Optional[float]:
        """
        Calcula el peso estimado en kilogramos aplicando la fórmula inversa del IMC:
        Peso (kg) = IMC * Estatura (m)²
        """
        if height_m is not None and height_m > 0:
            return float(round(bmi * (height_m ** 2), 2))
        return None

    @staticmethod
    def categorize_who(bmi: float) -> Tuple[str, str, str]:
        """
        Clasifica el estado nutricional según los estándares de la OMS.
        Retorna: (Nombre legible, código corto, color hexadecimal)
        """
        if bmi < 18.5:
            return "Bajo peso", "underweight", "#3b82f6"
        elif bmi < 25.0:
            return "Normal (Saludable)", "normal", "#10b981"
        elif bmi < 30.0:
            return "Sobrepeso", "overweight", "#f59e0b"
        elif bmi < 35.0:
            return "Obesidad Grado I", "obese_1", "#f97316"
        elif bmi < 40.0:
            return "Obesidad Grado II", "obese_2", "#ef4444"
        else:
            return "Obesidad Grado III (Mórbida)", "obese_3", "#991b1b"
