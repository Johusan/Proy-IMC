import os
import sys
import time
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Forzar salida en consola sin buffer para ver el progreso en tiempo real
sys.stdout.reconfigure(line_buffering=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
for p in [os.path.join(PROJECT_ROOT, "DigitalScale"), os.path.join(PROJECT_ROOT, "DigitalScale", "backend")]:
    if p not in sys.path:
        sys.path.append(p)

# Importar arquitectura SE-DenseNet-121
try:
    from DigitalScale.backend.src.models.densenet.densenet import SEDensenet121, load_pretrained_densenet
except ImportError:
    try:
        from DigitalScale.src.models.densenet.densenet import SEDensenet121, load_pretrained_densenet
    except ImportError:
        from src.models.densenet.densenet import SEDensenet121, load_pretrained_densenet

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


class _ResizeKeepAspect(transforms.Resize):
    """
    Redimensiona la imagen preservando su relación de aspecto original,
    escalando por el lado mayor hasta IMG_SIZE.
    Evita la deformación anatómica de cuerpos humanos.
    """
    def __call__(self, img):
        if hasattr(img, 'size'):
            w, h = img.size
        elif hasattr(img, 'shape'):
            h, w = img.shape[-2], img.shape[-1]
        else:
            return super().__call__(img)

        scale = max(w, h) / float(self.size)
        new_w = max(1, int(w / scale))
        new_h = max(1, int(h / scale))
        return transforms.functional.resize(img, (new_h, new_w), self.interpolation)


class TorsoBMIDataset(Dataset):
    """
    Dataset especializado en recortes anatómicos de torso
    con las mismas transformaciones de entrada que el servidor web.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row.get("cropped_photo_path", "")

        # Respaldo si no existe la ruta guardada
        if not os.path.exists(image_path):
            image_path = row.get("photo_path", "")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Respaldo negro si estuviera corrupta
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        bmi = float(row["bmi"])
        return image, torch.tensor(bmi, dtype=torch.float32)


def train_model(epochs=35, batch_size=16, lr=1e-4):
    print("=" * 65, flush=True)
    print("DIGITALSCALE AI — ENTRENAMIENTO DE PRECISIÓN (RTX 3050)", flush=True)
    print("=" * 65, flush=True)

    # 1. Verificación de Hardware (GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] Aceleración activa: {gpu_name} ({vram_gb:.1f} GB VRAM)", flush=True)
    else:
        device = torch.device("cpu")
        print("[ADVERTENCIA] CUDA no detectado. El entrenamiento correrá en CPU.", flush=True)

    # 2. Cargar Dataset con Recortes Anatómicos
    torso_csv = os.path.join(CURRENT_DIR, "torso_unified_dataset.csv")
    unified_csv = os.path.join(CURRENT_DIR, "unified_dataset.csv")

    if not os.path.exists(torso_csv):
        print(f"\n[INFO] No se encontró '{os.path.basename(torso_csv)}'.", flush=True)
        print("Ejecutando preprocesador anatómico de torso con MediaPipe primero...", flush=True)
        from preprocess_dataset import run_torso_preprocessing
        run_torso_preprocessing()

    if not os.path.exists(torso_csv):
        if os.path.exists(unified_csv):
            print(f"[AVISO] Usando dataset unificado sin recorte: {unified_csv}", flush=True)
            df = pd.read_csv(unified_csv)
        else:
            print(f"Error fatal: No se encontró ningún dataset en {CURRENT_DIR}", flush=True)
            return
    else:
        df = pd.read_csv(torso_csv)

    train_df = df[df["partition"] == "train"]
    val_df = df[df["partition"] == "val"]
    test_df = df[df["partition"] == "test"]

    print(f"\nParticiones del Dataset:", flush=True)
    print(f"   * Entrenamiento: {len(train_df):4d} imágenes", flush=True)
    print(f"   * Validación:    {len(val_df):4d} imágenes", flush=True)
    print(f"   * Prueba ciega:  {len(test_df):4d} imágenes", flush=True)

    # 3. Pipeline de Transformaciones — ALINEACIÓN 100% CON PRODUCCIÓN
    #    Usa ResizeKeepAspect -> Pad -> CenterCrop para mantener proporciones reales del cuerpo
    train_transforms = transforms.Compose([
        _ResizeKeepAspect(IMG_SIZE),
        transforms.Pad(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=6),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    eval_transforms = transforms.Compose([
        _ResizeKeepAspect(IMG_SIZE),
        transforms.Pad(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    train_dataset = TorsoBMIDataset(train_df, transform=train_transforms)
    val_dataset = TorsoBMIDataset(val_df, transform=eval_transforms)
    test_dataset = TorsoBMIDataset(test_df, transform=eval_transforms)

    # num_workers=0 para máxima estabilidad en Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Inicializar SE-DenseNet-121
    print("\nInicializando arquitectura SE-DenseNet-121...", flush=True)
    model = SEDensenet121()
    load_pretrained_densenet(model)
    model.to(device)

    for m in model.modules():
        if hasattr(m, "memory_efficient"):
            m.memory_efficient = False

    # Smooth L1 Loss (Huber Loss) es mucho más robusto contra outliers que MSE puro
    criterion = nn.SmoothL1Loss(beta=1.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_mae = float("inf")
    best_val_mape = float("inf")
    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model.ckpt")

    print(f"\nIniciando ciclo de entrenamiento ({epochs} épocas)...", flush=True)
    print("-" * 65, flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # FASE DE ENTRENAMIENTO
        model.train()
        train_loss, train_mae = 0.0, 0.0
        train_count = 0

        for images, bmis in train_loader:
            images = images.to(device, non_blocking=True)
            bmis = bmis.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, bmis)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_curr = images.size(0)
            train_loss += loss.item() * batch_size_curr
            train_mae += torch.sum(torch.abs(outputs - bmis)).item()
            train_count += batch_size_curr

        train_loss /= train_count
        train_mae /= train_count

        # FASE DE VALIDACIÓN
        model.eval()
        val_loss, val_mae, val_mape = 0.0, 0.0, 0.0
        val_count = 0

        with torch.no_grad():
            for images, bmis in val_loader:
                images = images.to(device, non_blocking=True)
                bmis = bmis.to(device, non_blocking=True).unsqueeze(1)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, bmis)

                batch_size_curr = images.size(0)
                val_loss += loss.item() * batch_size_curr
                val_mae += torch.sum(torch.abs(outputs - bmis)).item()
                val_mape += torch.sum(torch.abs((outputs - bmis) / bmis)).item()
                val_count += batch_size_curr

        val_loss /= val_count
        val_mae /= val_count
        val_mape = (val_mape / val_count) * 100.0

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Guardar si es el mejor modelo en validación
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_val_mape = val_mape
            checkpoint_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "MAE": val_mae,
                "MAPE": val_mape,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint_state, best_ckpt_path)
            best_flag = " ⭐ [MEJOR MODELO GUARDADO]"
        else:
            best_flag = ""

        print(
            f"Época [{epoch:02d}/{epochs:02d}] (LR: {curr_lr:.1e}) | "
            f"Train MAE: {train_mae:.2f} | "
            f"Val MAE: {val_mae:.2f} kg/m² | Val MAPE: {val_mape:.2f}%{best_flag}",
            flush=True,
        )

    total_min = (time.time() - start_time) / 60.0
    print("=" * 65, flush=True)
    print(f"Entrenamiento completado en {total_min:.2f} minutos.", flush=True)
    print(f"Mejor MAE en Validación: {best_val_mae:.2f} kg/m² | MAPE: {best_val_mape:.2f}%", flush=True)
    print(f"Archivo de pesos óptimo: {best_ckpt_path}", flush=True)

    # 5. Evaluación en Prueba Ciega (Held-Out Test Set)
    print("\nEvaluando en el conjunto de prueba independiente...", flush=True)
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        test_mae, test_mape = 0.0, 0.0
        test_count = 0
        with torch.no_grad():
            for images, bmis in test_loader:
                images = images.to(device)
                bmis = bmis.to(device).unsqueeze(1)
                outputs = model(images)

                batch_size_curr = images.size(0)
                test_mae += torch.sum(torch.abs(outputs - bmis)).item()
                test_mape += torch.sum(torch.abs((outputs - bmis) / bmis)).item()
                test_count += batch_size_curr

        test_mae /= test_count
        test_mape = (test_mape / test_count) * 100.0
        print(f"Métricas finales en conjunto de prueba (N={test_count} imágenes):", flush=True)
        print(f"   * MAE:  {test_mae:.2f} kg/m²", flush=True)
        print(f"   * MAPE: {test_mape:.2f}%", flush=True)

    print("\n[SIGUIENTE PASO] Ejecuta `python export_to_digitalscale.py` para sincronizar con la web.")


if __name__ == "__main__":
    train_model(epochs=35, batch_size=16, lr=1e-4)
