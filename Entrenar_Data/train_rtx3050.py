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

# Force unbuffered stdout for live logging
sys.stdout.reconfigure(line_buffering=True)

# Add root projects directory to PYTHONPATH to import DigitalScale modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "DigitalScale") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "DigitalScale"))
if os.path.join(PROJECT_ROOT, "DigitalScale", "backend") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "DigitalScale", "backend"))

# Import SE-DenseNet from DigitalScale (supporting both backend/src and src)
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

class CustomBMIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Fallback to local Image Dataset if path from CSV belongs to another machine
        if not os.path.exists(image_path):
            filename = str(row['image_id']).strip()
            folder = "Image_train" if row['partition'] in ['train', 'val'] else "Image_test"
            alt_path = os.path.join(CURRENT_DIR, "Image Dataset", folder, filename)
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                # Check extension swap (.jpg <-> .png)
                base_name, _ = os.path.splitext(filename)
                for ext in [".jpg", ".png"]:
                    candidate = os.path.join(CURRENT_DIR, "Image Dataset", folder, base_name + ext)
                    if os.path.exists(candidate):
                        image_path = candidate
                        break
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            # Fallback black image if corrupted or missing
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        return image, torch.tensor(bmi, dtype=torch.float32)

def train_model(epochs=20, batch_size=16, lr=1e-4):
    print("=" * 60, flush=True)
    print("DIGITAL SCALE - RTX 3050 HIGH PERFORMANCE TRAINING", flush=True)
    print("=" * 60, flush=True)
    
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Acceleration Active: {gpu_name}", flush=True)
        print(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("CUDA not detected. Training will run on CPU.", flush=True)

    # 2. Load Dataset CSV
    dataset_csv = os.path.join(CURRENT_DIR, "unified_dataset.csv")
    if not os.path.exists(dataset_csv):
        print(f"Error: {dataset_csv} not found. Run dataset_adapter.py first.", flush=True)
        return

    df = pd.read_csv(dataset_csv)
    train_df = df[df['partition'] == 'train']
    val_df = df[df['partition'] == 'val']
    test_df = df[df['partition'] == 'test']

    print(f"\nDataset Split:", flush=True)
    print(f"   Train samples: {len(train_df)}", flush=True)
    print(f"   Val samples:   {len(val_df)}", flush=True)
    print(f"   Test samples:  {len(test_df)}", flush=True)

    # 3. Data Transformations
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    train_dataset = CustomBMIDataset(train_df, transform=train_transforms)
    val_dataset = CustomBMIDataset(val_df, transform=val_transforms)
    test_dataset = CustomBMIDataset(test_df, transform=val_transforms)

    # Use num_workers=0 on Windows for stability with PIL image decoding
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Model Setup
    print("\nInitializing SE-DenseNet-121 Architecture...", flush=True)
    model = SEDensenet121()
    load_pretrained_densenet(model)
    model.to(device)

    # Disable memory efficient features for export compatibility
    for m in model.modules():
        if hasattr(m, 'memory_efficient'):
            m.memory_efficient = False

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_mae = float('inf')
    best_val_mape = float('inf')
    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model.ckpt")

    print("\nStarting Training Loop...", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # TRAIN EPOCH
        model.train()
        train_loss, train_mae = 0.0, 0.0
        train_count = 0

        for images, bmis in train_loader:
            images = images.to(device)
            bmis = bmis.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
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

        # VALIDATION EPOCH
        model.eval()
        val_loss, val_mae, val_mape = 0.0, 0.0, 0.0
        val_count = 0

        with torch.no_grad():
            for images, bmis in val_loader:
                images = images.to(device)
                bmis = bmis.to(device).unsqueeze(1)
                
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
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

        scheduler.step(val_loss)

        # Check if best model
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
            best_str = " (BEST MODEL SAVED)"
        else:
            best_str = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train MAE: {train_mae:.2f} | Val Loss: {val_loss:.2f} | Val MAE: {val_mae:.2f} | Val MAPE: {val_mape:.2f}%{best_str}", flush=True)

    elapsed = time.time() - start_time
    print("-" * 60, flush=True)
    print(f"Training Completed in {elapsed/60:.2f} minutes.", flush=True)
    print(f"Best Validation MAE: {best_val_mae:.2f} kg/m2 | MAPE: {best_val_mape:.2f}%", flush=True)
    print(f"Best Model Checkpoint: {best_ckpt_path}", flush=True)

    # TEST EVALUATION
    print("\nEvaluating Best Model on Held-Out Test Set...", flush=True)
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
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
        print(f"Test Set Metrics (N={test_count}):", flush=True)
        print(f"   MAE:  {test_mae:.2f} kg/m2", flush=True)
        print(f"   MAPE: {test_mape:.2f}%", flush=True)

if __name__ == "__main__":
    train_model(epochs=20, batch_size=16, lr=1e-4)
