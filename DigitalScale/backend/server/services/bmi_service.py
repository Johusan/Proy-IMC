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

        # 4. Pipeline de transformaciones para PyTorch
        self.preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

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
        Ejecuta el forward pass de la red neuronal sobre el recorte del torso
        y retorna el Índice de Masa Corporal estimado en kg/m².
        """
        if not self._is_ready:
            raise RuntimeError("El predictor de IMC no ha sido inicializado.")

        input_tensor = self.preprocess(torso_pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output_bmi = self.model(input_tensor).item()

        return float(round(output_bmi, 2))

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
