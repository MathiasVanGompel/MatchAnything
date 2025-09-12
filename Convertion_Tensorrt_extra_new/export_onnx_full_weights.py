#!/usr/bin/env python3
"""
ONNX export that ensures full model weights are included.
This version should create a properly sized ONNX file with all weights.
"""

import os
import sys
import torch
import onnx
import gc
from pathlib import Path
import argparse

# Add parent directory to path
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt_new"

for path in [_THIS_DIR, _PARENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT
from improved_weight_adapter import apply_improved_mapping

def export_onnx_full_weights(onnx_path, checkpoint_path=None, H=288, W=288, verbose=False):
    """Export ONNX ensuring full model weights are included."""
    
    print(f"🚀 FULL WEIGHT ONNX EXPORT")
    print(f"📁 Output: {onnx_path}")
    print(f"📐 Input: {H}x{W}")
    print(f"💾 Checkpoint: {checkpoint_path or 'None'}")
    print("-" * 60)
    
    # Create model
    print("Creating model...")
    model = AccurateMatchAnythingTRT(amp=False).eval()
    
    # Fix patch size
    if hasattr(model.encoder, 'patch'):
        model.encoder.patch = 16
        print("✅ Fixed patch size to 16")
    
    # Load weights with improved mapping
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Loading weights with improved mapping...")
        try:
            loadable_weights = apply_improved_mapping(checkpoint_path, model.state_dict())
            missing, unexpected = model.load_state_dict(loadable_weights, strict=False)
            
            total_params = len(model.state_dict())
            loaded_params = len(loadable_weights)
            load_percentage = (loaded_params / total_params) * 100
            
            print(f"📊 Weight loading results:")
            print(f"   Loaded: {loaded_params}/{total_params} ({load_percentage:.1f}%)")
            print(f"   Missing: {len(missing)}")
            
            # Show some missing parameters for debugging
            if missing:
                print(f"   Some missing params: {missing[:5]}")
            
            if load_percentage >= 80:
                print("🎉 Excellent weight loading!")
            elif load_percentage >= 50:
                print("✅ Good weight loading!")
            else:
                print("⚠️ Partial weight loading")
                
        except Exception as e:
            print(f"❌ Weight loading failed: {e}")
            print("Proceeding with random weights")
    
    # Prepare inputs
    H = ((H + 15) // 16) * 16
    W = ((W + 15) // 16) * 16
    print(f"📐 Adjusted dimensions: {H}x{W}")
    
    x0 = torch.rand(1, 3, H, W)
    x1 = torch.rand(1, 3, H, W)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.inference_mode():
        outputs = model(x0, x1)
    print("✅ Forward pass successful")
    
    # Show output info
    if isinstance(outputs, (tuple, list)):
        print(f"📤 Model outputs: {len(outputs)} tensors")
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f"   Output {i}: {out.shape}")
    
    # Create output directory
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ONNX export with full weights
    print(f"🔄 Exporting ONNX with full weights...")
    print(f"   Target: {onnx_path}")
    print(f"   This will create a large file (~500MB-1GB+)...")
    
    try:
        # Configuration that ensures full model export
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
            opset_version=17,  # Use modern opset for better support
            do_constant_folding=True,  # Enable optimizations
            export_params=True,  # CRITICAL: Include all parameters
            keep_initializers_as_inputs=False,  # Don't duplicate weights as inputs
            verbose=verbose
        )
        
        print("✅ ONNX export completed!")
        
        # Verify file exists and check size
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024*1024)
            print(f"✅ ONNX file created: {size_mb:.1f} MB")
            
            # Expected size analysis
            if size_mb < 100:
                print(f"⚠️ File size seems small - weights may not be fully included")
                print(f"   Expected size: 500MB-1GB+ for full MatchAnything model")
            elif size_mb < 500:
                print(f"✅ Reasonable file size - partial weights included")
            else:
                print(f"🎉 Large file size - full weights likely included!")
            
            # Try validation
            try:
                print(f"Validating ONNX model...")
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print(f"✅ ONNX validation passed")
                
                print(f"📋 Model info:")
                print(f"   Graph inputs: {len(onnx_model.graph.input)}")
                print(f"   Graph outputs: {len(onnx_model.graph.output)}")
                print(f"   Graph nodes: {len(onnx_model.graph.node)}")
                print(f"   Initializers: {len(onnx_model.graph.initializer)}")
                
            except Exception as e:
                print(f"⚠️ ONNX validation failed: {e}")
                print(f"   File created but may have issues")
            
            return str(onnx_path)
        else:
            print("❌ ONNX file was not created")
            return None
            
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full weight ONNX export")
    parser.add_argument("--onnx", default="Convertion_Tensorrt_extra_new/out/full_weights.onnx", help="Output ONNX file")
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
    
    result = export_onnx_full_weights(
        onnx_path=args.onnx,
        checkpoint_path=args.ckpt if args.ckpt else None,
        H=args.H,
        W=args.W,
        verbose=args.verbose
    )
    
    if result:
        print(f"\n🎉 SUCCESS!")
        print(f"ONNX model exported to: {result}")
        
        # File size analysis
        size_mb = Path(result).stat().st_size / (1024*1024)
        if size_mb >= 500:
            print(f"🎉 Excellent! Large file size ({size_mb:.1f} MB) indicates full weights")
        elif size_mb >= 100:
            print(f"✅ Good file size ({size_mb:.1f} MB) - most weights included")
        else:
            print(f"⚠️ Small file size ({size_mb:.1f} MB) - some weights may be missing")
            
        print(f"\n💡 Next steps:")
        print(f"   - Test the ONNX model with inference")
        print(f"   - Compare results with PyTorch model")
        print(f"   - Use for TensorRT conversion if needed")
        
    else:
        print(f"\n💥 EXPORT FAILED!")
        sys.exit(1)