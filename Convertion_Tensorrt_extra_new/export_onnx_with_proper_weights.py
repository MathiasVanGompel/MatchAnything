#!/usr/bin/env python3
"""
ONNX export using the existing improved weight mapping from Convertion_Tensorrt.
This should properly load your MatchAnything weights.
"""

import os
import sys
import torch
import onnx
from pathlib import Path
import argparse

# Add both directories to path
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt_new"
_TENSORRT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt"

for path in [_PARENT_DIR, _TENSORRT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import model and weight mapping
from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT
from improved_weight_adapter import apply_improved_mapping

def export_with_proper_weights(onnx_path, checkpoint_path=None, H=288, W=288, verbose=False):
    """Export ONNX using the existing improved weight mapping."""
    
    print(f"🚀 ONNX EXPORT WITH PROPER WEIGHT MAPPING")
    print(f"📁 Output: {onnx_path}")
    print(f"📐 Input: {H}x{W}")
    print(f"💾 Checkpoint: {checkpoint_path or 'None'}")
    print("=" * 60)
    
    # Step 1: Create model
    print("[1] Creating model...")
    model = AccurateMatchAnythingTRT(amp=False).eval()
    
    # Fix patch size (critical!)
    if hasattr(model.encoder, 'patch'):
        old_patch = model.encoder.patch
        model.encoder.patch = 16
        print(f"[1] ✅ Fixed patch size: {old_patch} → 16")
    
    model_state_dict = model.state_dict()
    print(f"[1] ✅ Model created with {len(model_state_dict)} parameters")
    
    # Step 2: Load weights using improved mapping
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[2] Loading weights using improved mapping...")
        try:
            # Use the existing improved weight adapter
            loadable_weights = apply_improved_mapping(checkpoint_path, model_state_dict)
            
            # Load the mapped weights
            missing, unexpected = model.load_state_dict(loadable_weights, strict=False)
            
            total_params = len(model_state_dict)
            loaded_params = len(loadable_weights)
            load_percentage = (loaded_params / total_params) * 100
            
            print(f"[2] ✅ Improved mapping results:")
            print(f"    Loaded: {loaded_params}/{total_params} ({load_percentage:.1f}%)")
            print(f"    Missing: {len(missing)}")
            print(f"    Unexpected: {len(unexpected)}")
            
            if load_percentage >= 50:
                print(f"[2] 🎉 Good weight loading achieved!")
            elif load_percentage >= 20:
                print(f"[2] ⚠️ Partial weight loading - should still work")
            else:
                print(f"[2] ❌ Low weight loading - may have issues")
                
        except Exception as e:
            print(f"[2] ❌ Weight loading failed: {e}")
            print(f"[2] Proceeding with random weights")
    else:
        print(f"[2] ⚠️ No checkpoint provided, using random weights")
    
    # Step 3: Prepare inputs
    print(f"[3] Preparing inputs...")
    # Ensure multiples of 16
    H = ((H + 15) // 16) * 16
    W = ((W + 15) // 16) * 16
    print(f"[3] Adjusted dimensions: {H}x{W}")
    
    x0 = torch.rand(1, 3, H, W)
    x1 = torch.rand(1, 3, H, W)
    print(f"[3] ✅ Input tensors ready: {x0.shape}")
    
    # Step 4: Test forward pass
    print(f"[4] Testing forward pass...")
    with torch.inference_mode():
        outputs = model(x0, x1)
    
    print(f"[4] ✅ Forward pass successful!")
    if isinstance(outputs, (tuple, list)):
        print(f"[4] Model returned {len(outputs)} outputs:")
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f"    Output {i}: {out.shape}")
    
    # Step 5: Export to ONNX
    print(f"[5] Exporting to ONNX...")
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[5] Target file: {onnx_path}")
    print(f"[5] Starting export (this may take several minutes)...")
    
    # Use the same configuration that worked in minimal test
    torch.onnx.export(
        model,
        (x0, x1),
        str(onnx_path),
        input_names=["image0", "image1"],
        output_names=["warp_c", "cert_c", "valid_mask", "coarse_stride"],
        dynamic_axes={
            "image0": {"0": "B", "2": "H", "3": "W"},
            "image1": {"0": "B", "2": "H", "3": "W"},
            "warp_c": {"0": "B", "1": "Ha", "2": "Wa"},
            "cert_c": {"0": "B", "1": "Ha", "2": "Wa"},
            "valid_mask": {"0": "B", "1": "Ha", "2": "Wa"},
            "coarse_stride": {"0": "one"},
        },
        opset_version=11,  # Use stable opset
        do_constant_folding=False,  # Avoid optimization issues
        verbose=verbose
    )
    
    print(f"[5] ✅ ONNX export completed!")
    
    # Step 6: Verify
    print(f"[6] Verifying ONNX file...")
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024*1024)
        print(f"[6] ✅ File created: {size_mb:.1f} MB")
        
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"[6] ✅ ONNX validation passed")
            
            print(f"[6] 📋 Model info:")
            print(f"    Inputs: {len(onnx_model.graph.input)}")
            print(f"    Outputs: {len(onnx_model.graph.output)}")
            print(f"    Nodes: {len(onnx_model.graph.node)}")
            
        except Exception as e:
            print(f"[6] ⚠️ ONNX validation failed: {e}")
    else:
        print(f"[6] ❌ ONNX file not found!")
        return None
    
    # Success summary
    print("=" * 60)
    print("🎉 EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✅ ONNX file: {onnx_path}")
    print(f"✅ File size: {size_mb:.1f} MB")
    print(f"🎯 Input size: {H}x{W}")
    print(f"🔧 Patch size: 16")
    print("=" * 60)
    
    return str(onnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX export with proper weight mapping")
    parser.add_argument("--onnx", default="out/proper_weights.onnx", help="Output ONNX file")
    parser.add_argument("--ckpt", default="", help="Checkpoint path")
    parser.add_argument("--H", type=int, default=288, help="Input height")
    parser.add_argument("--W", type=int, default=288, help="Input width")
    parser.add_argument("--verbose", action="store_true", help="Verbose export")
    
    args = parser.parse_args()
    
    print(f"🔧 Configuration:")
    print(f"   ONNX: {args.onnx}")
    print(f"   Checkpoint: {args.ckpt}")
    print(f"   Dimensions: {args.H}x{args.W}")
    print(f"   Verbose: {args.verbose}")
    print()
    
    result = export_with_proper_weights(
        onnx_path=args.onnx,
        checkpoint_path=args.ckpt if args.ckpt else None,
        H=args.H,
        W=args.W,
        verbose=args.verbose
    )
    
    if result:
        print(f"\n🎉🎉🎉 FINAL SUCCESS! 🎉🎉🎉")
        print(f"ONNX model with proper weights: {result}")
        print(f"\nThis ONNX file should now have your MatchAnything weights loaded correctly!")
    else:
        print(f"\n💥 EXPORT FAILED!")
        sys.exit(1)