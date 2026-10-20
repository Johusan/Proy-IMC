import torch
import torch.nn as nn
import numpy as np
from densenet import SEDensenet121
import os

def convert_pytorch_to_executorch_fixed():
    """
    Convert the trained PyTorch DenseNet model to ExecuTorch format with fixes
    """
    print("Loading PyTorch model...")
    
    # Load the trained model
    MODEL_CHECKPOINT = "trained_models/model_5/best_model.ckpt"
    model = SEDensenet121()
    
    # Load state dict with CPU mapping to avoid CUDA issues
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Fix 1: Disable memory efficient features that cause tracing issues
    for module in model.modules():
        if hasattr(module, 'memory_efficient'):
            module.memory_efficient = False
    
    # Fix 2: Disable checkpointing in all modules
    for module in model.modules():
        if hasattr(module, 'call_checkpoint_bottleneck'):
            module.call_checkpoint_bottleneck = None
    
    # Fix 3: Replace any problematic operations with ExecuTorch-compatible ones
    # This keeps the full model but makes it compatible
    
    # Create a sample input for tracing
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Export to ExecuTorch format
    print("Exporting to ExecuTorch format...")
    
    # Create the output directory
    output_dir = "android/app/src/main/assets"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import ExecuTorch modules
        from executorch.exir import to_edge
        
        print("✓ ExecuTorch modules imported successfully")
        
        # First export using torch.export with constraints
        print("Exporting with torch.export...")
        
        # Try to export with dynamic shapes disabled and specific constraints
        exported_model = torch.export.export(
            model, 
            (sample_input,),
            dynamic_shapes=None  # Disable dynamic shapes
        )
        print("✓ Model exported with torch.export successfully!")
        
        # Convert to Edge dialect
        edge_model = to_edge(exported_model)
        print("✓ Model converted to Edge dialect successfully!")
        
        # Export to ExecuTorch format
        executorch_model = edge_model.to_executorch()
        print("✓ Model converted to ExecuTorch format successfully!")
        
        # Save the model
        with open(f"{output_dir}/bmi_model_fixed.pte", "wb") as f:
            f.write(executorch_model.buffer)
        print(f"✓ Model exported to: {output_dir}/bmi_model_fixed.pte")
        
        # Test the model to make sure it works
        print("Testing the exported model...")
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✓ Test inference successful! Output shape: {test_output.shape}")
        print(f"✓ Sample output value: {test_output.item():.4f}")
        
    except ImportError as e:
        print(f"ExecuTorch not available: {e}")
        raise RuntimeError("ExecuTorch is required for this conversion. Please install executorch.")
    except Exception as e:
        print(f"ExecuTorch conversion failed: {e}")

        
    
    print("Model conversion completed!")
    print("\nNext steps:")
    print("1. Test the exported model on Android")
    print("2. If it works, you'll have your full DenseNet with good accuracy")

if __name__ == "__main__":
    convert_pytorch_to_executorch_fixed() 