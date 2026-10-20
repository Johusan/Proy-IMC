import os
import sys
import shutil
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
for p in [os.path.join(PROJECT_ROOT, "DigitalScale", "backend"), os.path.join(PROJECT_ROOT, "DigitalScale")]:
    if p not in sys.path:
        sys.path.append(p)

try:
    from DigitalScale.backend.src.models.densenet.densenet import SEDensenet121
except ImportError:
    try:
        from DigitalScale.src.models.densenet.densenet import SEDensenet121
    except ImportError:
        from src.models.densenet.densenet import SEDensenet121

def export_checkpoint():
    print("=" * 60)
    print("EXPORTING VERIDICAL WEIGHTS TO DIGITALSCALE & ANDROID")
    print("=" * 60)
    
    src_ckpt = os.path.join(CURRENT_DIR, "output", "best_model.ckpt")
    if not os.path.exists(src_ckpt):
        print(f"Error: Checkpoint file not found at {src_ckpt}")
        print("   Please run train_rtx3050.py first!")
        return

    # Define target destinations
    target_weights_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "DigitalScale", "backend", "weights"))
    target_model5_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "DigitalScale", "backend", "trained_models", "model_5"))

    os.makedirs(target_weights_dir, exist_ok=True)
    os.makedirs(target_model5_dir, exist_ok=True)

    dest_ckpt_1 = os.path.join(target_weights_dir, "best_model.ckpt")
    dest_ckpt_2 = os.path.join(target_model5_dir, "best_model.ckpt")

    # Copy files
    shutil.copy2(src_ckpt, dest_ckpt_1)
    shutil.copy2(src_ckpt, dest_ckpt_2)

    print(f"Checkpoint copied to Python weights dir:")
    print(f"   {dest_ckpt_1}")
    print(f"Checkpoint copied to Model_5 conversion dir:")
    print(f"   {dest_ckpt_2}")

    # Test loading state_dict into SE-DenseNet-121
    print("\nTesting Checkpoint Loading & Forward Pass...")
    device = torch.device("cpu")
    model = SEDensenet121()
    
    checkpoint = torch.load(dest_ckpt_1, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    sample_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        sample_output = model(sample_input).item()

    print(f"[OK] Model loaded successfully!")
    print(f"[OK] Sample Forward Pass Output (BMI): {sample_output:.2f} kg/m2")
    print(f"[OK] Original Checkpoint Metrics -> MAE: {checkpoint.get('MAE', 'N/A')}, MAPE: {checkpoint.get('MAPE', 'N/A')}%")

    print("\nWeights export completed successfully!")
    print("Next step: Run `python webcam_demo.py` in your laptop or run `convert_model_executorch_fixed.py` for Android!")

if __name__ == "__main__":
    export_checkpoint()
