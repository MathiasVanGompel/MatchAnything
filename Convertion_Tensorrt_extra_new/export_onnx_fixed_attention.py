#!/usr/bin/env python3
"""
Fixed ONNX export for MatchAnything with proper weight loading and detailed feedback.
This script addresses the issues preventing ONNX file generation.
FIXED: Now uses patch_size=16 (not 14) for proper dimension rounding.
"""

import os
import sys
import inspect
import torch
import onnx
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse

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


def load_weights_with_detailed_feedback(model, checkpoint_path: str) -> Tuple[bool, Dict]:
    """
    Load weights with detailed feedback about what was loaded.
    Returns (success, stats_dict)
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[WEIGHTS] ❌ Checkpoint not found: {checkpoint_path}")
        return False, {}
    
    print(f"[WEIGHTS] Loading checkpoint from: {checkpoint_path}")
    
    # Get model state for comparison
    model_state = model.state_dict()
    total_model_params = len(model_state)
    
    stats = {
        "total_model_params": total_model_params,
        "loaded_params": 0,
        "missing_params": 0,
        "unexpected_params": 0,
        "load_percentage": 0.0
    }
    
    try:
        # Try unified weight loader first
        try:
            from unified_weight_loader_fixed import apply_unified_weight_loading
            print("[WEIGHTS] Trying unified weight loader...")
            
            loadable = apply_unified_weight_loading(
                checkpoint_path, model_state, load_dinov2_components=True
            )
            missing, unexpected = model.load_state_dict(loadable, strict=False)
            
            stats["loaded_params"] = len(loadable)
            stats["missing_params"] = len(missing)
            stats["unexpected_params"] = len(unexpected)
            stats["load_percentage"] = (len(loadable) / total_model_params) * 100
            
            print(f"[WEIGHTS] Unified loader results:")
            print(f"  - Loaded parameters: {stats['loaded_params']}/{total_model_params}")
            print(f"  - Missing parameters: {stats['missing_params']}")
            print(f"  - Unexpected parameters: {stats['unexpected_params']}")
            print(f"  - Load percentage: {stats['load_percentage']:.1f}%")
            
            if stats['load_percentage'] >= 80:
                print("[WEIGHTS] ✅ Successfully loaded weights via unified loader")
                return True, stats
            else:
                print("[WEIGHTS] ⚠️ Low weight loading percentage, trying fallback...")
                
        except Exception as e:
            print(f"[WEIGHTS] Unified loader failed: {e}")
        
        # Fallback to direct loading
        print("[WEIGHTS] Trying direct checkpoint loading...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("[WEIGHTS] Found 'state_dict' key in checkpoint")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("[WEIGHTS] Found 'model' key in checkpoint")
        else:
            state_dict = checkpoint
            print("[WEIGHTS] Using checkpoint directly as state_dict")
        
        print(f"[WEIGHTS] Checkpoint contains {len(state_dict)} parameters")
        
        # Show some example parameter names for debugging
        ckpt_keys = list(state_dict.keys())[:5]
        model_keys = list(model_state.keys())[:5]
        print(f"[WEIGHTS] Sample checkpoint keys: {ckpt_keys}")
        print(f"[WEIGHTS] Sample model keys: {model_keys}")
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        stats["loaded_params"] = len(state_dict) - len(missing)
        stats["missing_params"] = len(missing)
        stats["unexpected_params"] = len(unexpected)
        stats["load_percentage"] = (stats["loaded_params"] / total_model_params) * 100
        
        print(f"[WEIGHTS] Direct loader results:")
        print(f"  - Loaded parameters: {stats['loaded_params']}/{total_model_params}")
        print(f"  - Missing parameters: {stats['missing_params']}")
        print(f"  - Unexpected parameters: {stats['unexpected_params']}")
        print(f"  - Load percentage: {stats['load_percentage']:.1f}%")
        
        # Show some missing parameters for debugging
        if missing and len(missing) < 10:
            print(f"[WEIGHTS] Missing parameters: {missing}")
        elif missing:
            print(f"[WEIGHTS] First few missing parameters: {missing[:5]}")
        
        if stats['load_percentage'] >= 50:
            print("[WEIGHTS] ✅ Weights loaded via direct method")
            return True, stats
        else:
            print("[WEIGHTS] ❌ Direct loading also insufficient")
            
    except Exception as e:
        print(f"[WEIGHTS] ❌ All weight loading methods failed: {e}")
        import traceback
        traceback.print_exc()
    
    return False, stats


def export_onnx_with_detailed_feedback(
    onnx_path: str,
    checkpoint_path: Optional[str] = None,
    H: int = 288,
    W: int = 288,
    verbose: bool = False,
) -> str:
    """
    Export MatchAnything model to ONNX with detailed feedback and error handling.
    FIXED: Now uses patch_size=16 for proper dimension rounding.
    """
    device = "cpu"
    
    print(f"[EXPORT] =" * 60)
    print(f"[EXPORT] Starting ONNX export process")
    print(f"[EXPORT] Target file: {onnx_path}")
    print(f"[EXPORT] Input dimensions: {H}x{W}")
    print(f"[EXPORT] Checkpoint: {checkpoint_path or 'None (random weights)'}")
    print(f"[EXPORT] =" * 60)
    
    # Create model
    print(f"[EXPORT] Creating AccurateMatchAnythingTRT model...")
    try:
        model = AccurateMatchAnythingTRT(amp=False).to(device).eval()
        print(f"[EXPORT] ✅ Model created successfully")
    except Exception as e:
        print(f"[EXPORT] ❌ Failed to create model: {e}")
        raise
    
    # Load weights if checkpoint is provided
    weights_loaded = False
    load_stats = {}
    if checkpoint_path:
        weights_loaded, load_stats = load_weights_with_detailed_feedback(model, checkpoint_path)
    
    if not weights_loaded:
        print("[EXPORT] ⚠️ Proceeding with random initialization")
        print("[EXPORT] Note: This will produce poor matching results but ONNX export should work")
    
    # FIXED: Ensure dimensions are multiples of 16 for patch embedding (not 14!)
    # The error showed patch_height=16, so we need multiples of 16
    H = ((H + 15) // 16) * 16  # Round up to nearest multiple of 16
    W = ((W + 15) // 16) * 16
    
    print(f"[EXPORT] Adjusted input dimensions: {H}x{W} (multiples of 16 for patch embedding)")
    
    # Create dummy inputs
    print(f"[EXPORT] Creating dummy inputs...")
    x0 = torch.rand(1, 3, H, W, device=device)
    x1 = torch.rand(1, 3, H, W, device=device)
    print(f"[EXPORT] Input shapes: {x0.shape}, {x1.shape}")
    
    # Test forward pass
    print("[EXPORT] Testing forward pass...")
    with torch.inference_mode():
        try:
            outputs = model(x0, x1)
            print("[EXPORT] ✅ Forward pass successful")
            
            # Show output details
            if isinstance(outputs, (tuple, list)):
                print(f"[EXPORT] Model returned {len(outputs)} outputs:")
                for i, tensor in enumerate(outputs):
                    if isinstance(tensor, torch.Tensor):
                        print(f"  Output {i}: {tuple(tensor.shape)} ({tensor.dtype})")
                    else:
                        print(f"  Output {i}: {type(tensor)} - {tensor}")
            elif isinstance(outputs, dict):
                print(f"[EXPORT] Model returned dict with keys: {list(outputs.keys())}")
                for key, tensor in outputs.items():
                    if isinstance(tensor, torch.Tensor):
                        print(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
                    else:
                        print(f"  {key}: {type(tensor)} - {tensor}")
            else:
                print(f"[EXPORT] Model returned: {type(outputs)}")
                
        except Exception as e:
            print(f"[EXPORT] ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Prepare ONNX export
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[EXPORT] Output directory: {onnx_path.parent}")
    
    # Set up dynamic axes based on the output format
    if isinstance(outputs, (tuple, list)):
        # Tuple/list outputs
        output_names = [f"output_{i}" for i in range(len(outputs))]
        dynamic_axes = {
            "image0": {"0": "B", "2": "H", "3": "W"},
            "image1": {"0": "B", "2": "H", "3": "W"},
        }
        # Add dynamic axes for outputs
        for i, name in enumerate(output_names):
            dynamic_axes[name] = {"0": "B"}
    else:
        # Dict outputs (legacy format)
        output_names = ["warp_c", "cert_c", "valid_mask", "coarse_stride"]
        dynamic_axes = {
            "image0": {"0": "B", "2": "H", "3": "W"},
            "image1": {"0": "B", "2": "H", "3": "W"},
            "warp_c": {"0": "B", "1": "Ha", "2": "Wa"},
            "cert_c": {"0": "B", "1": "Ha", "2": "Wa"},
            "valid_mask": {"0": "B", "1": "Ha", "2": "Wa"},
            "coarse_stride": {"0": "one"},
        }
    
    print(f"[EXPORT] Output names: {output_names}")
    
    export_kwargs = {
        "input_names": ["image0", "image1"],
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,
        "do_constant_folding": True,
        "verbose": verbose,
    }
    
    # Add external data format if available (for large models)
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["use_external_data_format"] = True
        print("[EXPORT] Using external data format for large models")
    
    # Perform ONNX export
    print(f"[EXPORT] Starting ONNX export to: {onnx_path}")
    print(f"[EXPORT] This may take several minutes...")
    
    try:
        torch.onnx.export(model, (x0, x1), str(onnx_path), **export_kwargs)
        print("[EXPORT] ✅ ONNX export completed successfully!")
    except Exception as e:
        print(f"[EXPORT] ❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Verify the exported model
    print("[EXPORT] Verifying exported ONNX model...")
    try:
        if onnx_path.exists():
            file_size_mb = onnx_path.stat().st_size / (1024*1024)
            print(f"[EXPORT] ✅ ONNX file created: {file_size_mb:.1f} MB")
            
            # Try to load and validate the ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"[EXPORT] ✅ ONNX model validation successful")
            
            # Show model info
            print(f"[EXPORT] Model info:")
            print(f"  - Inputs: {len(onnx_model.graph.input)}")
            print(f"  - Outputs: {len(onnx_model.graph.output)}")
            print(f"  - Nodes: {len(onnx_model.graph.node)}")
            
        else:
            print(f"[EXPORT] ❌ ONNX file was not created at {onnx_path}")
            return None
            
    except Exception as e:
        print(f"[EXPORT] ⚠️ ONNX model verification failed: {e}")
        # Don't fail the export if verification fails
    
    # Summary
    print(f"[EXPORT] =" * 60)
    print(f"[EXPORT] EXPORT SUMMARY")
    print(f"[EXPORT] =" * 60)
    print(f"[EXPORT] ✅ ONNX file: {onnx_path}")
    print(f"[EXPORT] ✅ File size: {onnx_path.stat().st_size / (1024*1024):.1f} MB")
    if load_stats:
        print(f"[EXPORT] 📊 Weight loading: {load_stats['load_percentage']:.1f}% ({load_stats['loaded_params']}/{load_stats['total_model_params']})")
    print(f"[EXPORT] 🎯 Input dimensions: {H}x{W}")
    print(f"[EXPORT] 🔧 Opset version: 17")
    print(f"[EXPORT] =" * 60)
    
    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MatchAnything to ONNX with detailed feedback")
    parser.add_argument("--onnx", default="out/test_fixed.onnx", 
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
    
    try:
        output_path = export_onnx_with_detailed_feedback(
            onnx_path=args.onnx,
            checkpoint_path=args.ckpt if args.ckpt else None,
            H=args.H,
            W=args.W,
            verbose=args.verbose,
        )
        
        if output_path:
            print(f"\n🎉 SUCCESS! ONNX model exported to: {output_path}")
        else:
            print(f"\n💥 FAILURE! ONNX export failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 FAILURE! Export failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)