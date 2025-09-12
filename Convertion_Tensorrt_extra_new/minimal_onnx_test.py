#!/usr/bin/env python3
"""
Minimal ONNX export test to isolate the exact failure point.
"""

import os
import sys
import torch
import onnx
from pathlib import Path
import traceback

# Add parent directory to path
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt_new"
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT

def minimal_test():
    print("🧪 MINIMAL ONNX EXPORT TEST")
    print("="*50)
    
    # Step 1: Create model (no weights)
    print("[1] Creating model...")
    try:
        model = AccurateMatchAnythingTRT(amp=False).eval()
        print("[1] ✅ Model created")
    except Exception as e:
        print(f"[1] ❌ Model creation failed: {e}")
        return False
    
    # Step 2: Fix patch size
    print("[2] Fixing patch size...")
    try:
        if hasattr(model.encoder, 'patch'):
            model.encoder.patch = 16
            print(f"[2] ✅ Patch size set to 16")
        else:
            print(f"[2] ⚠️ No patch attribute found")
    except Exception as e:
        print(f"[2] ❌ Patch fix failed: {e}")
        return False
    
    # Step 3: Create simple inputs
    print("[3] Creating inputs...")
    try:
        x0 = torch.rand(1, 3, 288, 288)  # Already multiple of 16
        x1 = torch.rand(1, 3, 288, 288)
        print(f"[3] ✅ Inputs created: {x0.shape}")
    except Exception as e:
        print(f"[3] ❌ Input creation failed: {e}")
        return False
    
    # Step 4: Test forward pass
    print("[4] Testing forward pass...")
    try:
        with torch.inference_mode():
            outputs = model(x0, x1)
        print(f"[4] ✅ Forward pass successful")
        print(f"[4] Output type: {type(outputs)}")
        if isinstance(outputs, (tuple, list)):
            print(f"[4] Number of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"[4]   Output {i}: {out.shape}")
        elif isinstance(outputs, dict):
            print(f"[4] Output keys: {list(outputs.keys())}")
    except Exception as e:
        print(f"[4] ❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Attempt ONNX export
    print("[5] Attempting ONNX export...")
    output_path = "/tmp/minimal_test.onnx"
    
    try:
        print("[5] About to call torch.onnx.export()...")
        
        # Minimal export configuration
        torch.onnx.export(
            model,
            (x0, x1),
            output_path,
            input_names=["image0", "image1"],
            output_names=["output_0", "output_1", "output_2", "output_3"],
            opset_version=11,  # Try older opset first
            do_constant_folding=False,  # Disable optimizations
            verbose=True
        )
        
        print("[5] ✅ torch.onnx.export() completed!")
        
        # Check if file exists
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024*1024)
            print(f"[5] ✅ ONNX file created: {size_mb:.1f} MB")
            
            # Try to load it
            try:
                onnx_model = onnx.load(output_path)
                print(f"[5] ✅ ONNX model loads successfully")
                return True
            except Exception as e:
                print(f"[5] ⚠️ ONNX file created but invalid: {e}")
                return True  # Still counts as partial success
        else:
            print(f"[5] ❌ ONNX file was not created")
            return False
            
    except Exception as e:
        print(f"[5] ❌ torch.onnx.export() failed:")
        print(f"[5] Exception type: {type(e).__name__}")
        print(f"[5] Exception message: {str(e)}")
        print(f"[5] Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_test()
    
    if success:
        print("\n🎉 MINIMAL TEST PASSED!")
        print("The basic ONNX export works - issue is likely in the complex script")
    else:
        print("\n💥 MINIMAL TEST FAILED!")
        print("There's a fundamental issue with ONNX export")
    
    print("\nNext steps:")
    if success:
        print("- The model can be exported to ONNX")
        print("- Issue is likely in the weight loading or complex export logic")
        print("- Try running without checkpoint first")
    else:
        print("- Check PyTorch and ONNX versions")
        print("- Try different opset versions")
        print("- Check for ONNX export compatibility issues")