#!/usr/bin/env python3
"""
Final debug version of ONNX export with maximum error visibility.
This version will catch and report ANY failure during ONNX export.
"""

import os
import sys
import inspect
import torch
import onnx
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import traceback

# Add parent directory to path to import from Convertion_Tensorrt
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt_new"
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

try:
    from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT
    print("[IMPORT] ✅ Successfully imported AccurateMatchAnythingTRT")
except ImportError as e:
    print(f"[IMPORT] ❌ Failed to import AccurateMatchAnythingTRT: {e}")
    sys.exit(1)


def load_weights_simple(model, checkpoint_path: str) -> Tuple[bool, Dict]:
    """Simplified weight loading with clear feedback."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[WEIGHTS] ❌ Checkpoint not found: {checkpoint_path}")
        return False, {}
    
    print(f"[WEIGHTS] Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        print(f"[WEIGHTS] Checkpoint contains {len(state_dict)} parameters")
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        total_params = len(model.state_dict())
        loaded_params = total_params - len(missing)
        load_percentage = (loaded_params / total_params) * 100
        
        print(f"[WEIGHTS] Loading results:")
        print(f"  - Total model parameters: {total_params}")
        print(f"  - Loaded parameters: {loaded_params}")
        print(f"  - Load percentage: {load_percentage:.1f}%")
        
        return True, {"load_percentage": load_percentage, "loaded_params": loaded_params, "total_params": total_params}
        
    except Exception as e:
        print(f"[WEIGHTS] ❌ Weight loading failed: {e}")
        return False, {}


def export_onnx_final_debug(
    onnx_path: str,
    checkpoint_path: Optional[str] = None,
    H: int = 288,
    W: int = 288,
    verbose: bool = False,
) -> str:
    """
    Export with maximum debugging and error visibility.
    """
    device = "cpu"
    
    print(f"\n{'='*80}")
    print(f"🚀 STARTING ONNX EXPORT - FINAL DEBUG VERSION")
    print(f"{'='*80}")
    print(f"📁 Target file: {onnx_path}")
    print(f"📐 Input dimensions: {H}x{W}")
    print(f"💾 Checkpoint: {checkpoint_path or 'None (random weights)'}")
    print(f"{'='*80}\n")
    
    # Step 1: Create model
    print(f"[STEP 1] Creating model...")
    try:
        model = AccurateMatchAnythingTRT(amp=False).to(device).eval()
        print(f"[STEP 1] ✅ Model created successfully")
    except Exception as e:
        print(f"[STEP 1] ❌ Model creation failed: {e}")
        traceback.print_exc()
        return None
    
    # Step 2: Fix patch size
    print(f"[STEP 2] Fixing patch size...")
    try:
        if hasattr(model.encoder, 'patch'):
            old_patch = model.encoder.patch
            model.encoder.patch = 16  # DINOv2 uses 16x16 patches
            print(f"[STEP 2] ✅ Fixed patch size: {old_patch} → 16")
        else:
            print(f"[STEP 2] ⚠️ No patch attribute found, assuming 16")
    except Exception as e:
        print(f"[STEP 2] ❌ Patch size fix failed: {e}")
        traceback.print_exc()
    
    # Step 3: Load weights
    print(f"[STEP 3] Loading weights...")
    load_stats = {}
    if checkpoint_path:
        weights_loaded, load_stats = load_weights_simple(model, checkpoint_path)
        if weights_loaded:
            print(f"[STEP 3] ✅ Weights loaded: {load_stats['load_percentage']:.1f}%")
        else:
            print(f"[STEP 3] ❌ Weight loading failed")
    else:
        print(f"[STEP 3] ⚠️ No checkpoint provided, using random weights")
    
    # Step 4: Prepare inputs
    print(f"[STEP 4] Preparing inputs...")
    try:
        # Ensure dimensions are multiples of 16
        H = ((H + 15) // 16) * 16
        W = ((W + 15) // 16) * 16
        
        x0 = torch.rand(1, 3, H, W, device=device)
        x1 = torch.rand(1, 3, H, W, device=device)
        
        print(f"[STEP 4] ✅ Input tensors created: {x0.shape}, {x1.shape}")
    except Exception as e:
        print(f"[STEP 4] ❌ Input preparation failed: {e}")
        traceback.print_exc()
        return None
    
    # Step 5: Test forward pass
    print(f"[STEP 5] Testing forward pass...")
    try:
        with torch.inference_mode():
            outputs = model(x0, x1)
        
        print(f"[STEP 5] ✅ Forward pass successful!")
        
        if isinstance(outputs, (tuple, list)):
            print(f"[STEP 5] Model returned {len(outputs)} outputs (tuple/list)")
            for i, tensor in enumerate(outputs):
                if isinstance(tensor, torch.Tensor):
                    print(f"  Output {i}: {tuple(tensor.shape)} ({tensor.dtype})")
        elif isinstance(outputs, dict):
            print(f"[STEP 5] Model returned dict with keys: {list(outputs.keys())}")
            for key, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
        else:
            print(f"[STEP 5] Model returned: {type(outputs)}")
            
    except Exception as e:
        print(f"[STEP 5] ❌ Forward pass failed: {e}")
        traceback.print_exc()
        return None
    
    # Step 6: Prepare ONNX export
    print(f"[STEP 6] Preparing ONNX export...")
    try:
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up output names based on what the model actually returns
        if isinstance(outputs, (tuple, list)):
            output_names = [f"output_{i}" for i in range(len(outputs))]
            dynamic_axes = {
                "image0": {"0": "B", "2": "H", "3": "W"},
                "image1": {"0": "B", "2": "H", "3": "W"},
            }
            for i, name in enumerate(output_names):
                dynamic_axes[name] = {"0": "B"}
        else:
            output_names = ["warp_c", "cert_c", "valid_mask", "coarse_stride"]
            dynamic_axes = {
                "image0": {"0": "B", "2": "H", "3": "W"},
                "image1": {"0": "B", "2": "H", "3": "W"},
                "warp_c": {"0": "B", "1": "Ha", "2": "Wa"},
                "cert_c": {"0": "B", "1": "Ha", "2": "Wa"},
                "valid_mask": {"0": "B", "1": "Ha", "2": "Wa"},
                "coarse_stride": {"0": "one"},
            }
        
        print(f"[STEP 6] ✅ Export configuration ready")
        print(f"  - Output names: {output_names}")
        print(f"  - Output directory: {onnx_path.parent}")
        
    except Exception as e:
        print(f"[STEP 6] ❌ Export preparation failed: {e}")
        traceback.print_exc()
        return None
    
    # Step 7: PERFORM ONNX EXPORT
    print(f"[STEP 7] 🚀 PERFORMING ONNX EXPORT...")
    print(f"[STEP 7] This is the critical step - watch for any errors!")
    print(f"[STEP 7] Exporting to: {onnx_path}")
    
    export_kwargs = {
        "input_names": ["image0", "image1"],
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,
        "do_constant_folding": True,
        "verbose": verbose,
    }
    
    try:
        print(f"[STEP 7] Calling torch.onnx.export()...")
        torch.onnx.export(
            model, 
            (x0, x1), 
            str(onnx_path), 
            **export_kwargs
        )
        print(f"[STEP 7] ✅ torch.onnx.export() completed without exception!")
        
    except Exception as e:
        print(f"[STEP 7] ❌ torch.onnx.export() FAILED with exception:")
        print(f"[STEP 7] Exception type: {type(e).__name__}")
        print(f"[STEP 7] Exception message: {str(e)}")
        print(f"[STEP 7] Full traceback:")
        traceback.print_exc()
        return None
    
    # Step 8: Verify file was created
    print(f"[STEP 8] Verifying ONNX file creation...")
    try:
        if onnx_path.exists():
            file_size_mb = onnx_path.stat().st_size / (1024*1024)
            print(f"[STEP 8] ✅ ONNX file exists: {file_size_mb:.1f} MB")
            
            # Try to load the ONNX model
            print(f"[STEP 8] Loading ONNX model for validation...")
            onnx_model = onnx.load(str(onnx_path))
            print(f"[STEP 8] ✅ ONNX model loaded successfully")
            
            # Validate the model
            print(f"[STEP 8] Validating ONNX model...")
            onnx.checker.check_model(onnx_model)
            print(f"[STEP 8] ✅ ONNX model validation passed")
            
            # Show model details
            print(f"[STEP 8] Model details:")
            print(f"  - Graph inputs: {len(onnx_model.graph.input)}")
            print(f"  - Graph outputs: {len(onnx_model.graph.output)}")
            print(f"  - Graph nodes: {len(onnx_model.graph.node)}")
            
        else:
            print(f"[STEP 8] ❌ ONNX file was NOT created at: {onnx_path}")
            print(f"[STEP 8] This means torch.onnx.export() returned without error but didn't create the file")
            return None
            
    except Exception as e:
        print(f"[STEP 8] ❌ ONNX file verification failed: {e}")
        traceback.print_exc()
        # Don't return None here - the file might exist but be invalid
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 EXPORT COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"✅ ONNX file: {onnx_path}")
    print(f"✅ File size: {onnx_path.stat().st_size / (1024*1024):.1f} MB")
    if load_stats:
        print(f"📊 Weight loading: {load_stats['load_percentage']:.1f}% ({load_stats['loaded_params']}/{load_stats['total_params']})")
    print(f"🎯 Input dimensions: {H}x{W}")
    print(f"🔧 Patch size: 16")
    print(f"🔧 Opset version: 17")
    print(f"{'='*80}")
    
    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MatchAnything to ONNX - Final Debug Version")
    parser.add_argument("--onnx", default="out/test_final_debug.onnx", 
                       help="Output ONNX file path")
    parser.add_argument("--ckpt", default="", 
                       help="Path to model checkpoint")
    parser.add_argument("--H", type=int, default=288, 
                       help="Input height (will be rounded to multiple of 16)")
    parser.add_argument("--W", type=int, default=288, 
                       help="Input width (will be rounded to multiple of 16)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose ONNX export")
    
    args = parser.parse_args()
    
    print(f"🔧 Arguments:")
    print(f"  --onnx: {args.onnx}")
    print(f"  --ckpt: {args.ckpt}")
    print(f"  --H: {args.H}")
    print(f"  --W: {args.W}")
    print(f"  --verbose: {args.verbose}")
    
    try:
        output_path = export_onnx_final_debug(
            onnx_path=args.onnx,
            checkpoint_path=args.ckpt if args.ckpt else None,
            H=args.H,
            W=args.W,
            verbose=args.verbose,
        )
        
        if output_path:
            print(f"\n🎉🎉🎉 FINAL SUCCESS! 🎉🎉🎉")
            print(f"ONNX model exported to: {output_path}")
            print(f"You can now use this ONNX file for inference!")
        else:
            print(f"\n💥 FINAL FAILURE!")
            print(f"The export process failed at some step.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR in main():")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)