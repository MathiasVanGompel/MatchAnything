#!/usr/bin/env python3
"""
Fixed ONNX export for MatchAnything with proper weight loading.
Combines the best of both conversion approaches.
"""

import os
import sys
import inspect
import torch
import onnx
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path to import from Convertion_Tensorrt
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt"
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT


def load_weights_safely(model, checkpoint_path: str) -> bool:
    """
    Load weights using the best available method.
    Returns True if weights were loaded successfully.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[WEIGHTS] Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Try the improved unified weight loader first
        from unified_weight_loader_fixed import apply_unified_weight_loading
        
        model_state = model.state_dict()
        loadable = apply_unified_weight_loading(
            checkpoint_path, model_state, load_dinov2_components=True
        )
        missing, unexpected = model.load_state_dict(loadable, strict=False)
        
        loaded_pct = (len(loadable) / len(model_state)) * 100
        print(f"[WEIGHTS] Unified loader: {len(loadable)}/{len(model_state)} ({loaded_pct:.1f}%) loaded")
        print(f"[WEIGHTS] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        if loaded_pct >= 80:  # Good loading threshold
            print("[WEIGHTS] ✅ Successfully loaded weights via unified loader")
            return True
        else:
            print("[WEIGHTS] ⚠️ Low weight loading percentage, trying fallback...")
            
    except Exception as e:
        print(f"[WEIGHTS] Unified loader failed: {e}")
    
    try:
        # Fallback to direct loading
        print("[WEIGHTS] Trying direct checkpoint loading...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded_count = len(state_dict) - len(missing)
        loaded_pct = (loaded_count / len(model.state_dict())) * 100
        
        print(f"[WEIGHTS] Direct load: {loaded_count}/{len(model.state_dict())} ({loaded_pct:.1f}%) loaded")
        
        if loaded_pct >= 50:  # Lower threshold for direct loading
            print("[WEIGHTS] ✅ Weights loaded via direct method")
            return True
        else:
            print("[WEIGHTS] ❌ Direct loading also insufficient")
            
    except Exception as e:
        print(f"[WEIGHTS] Direct loading failed: {e}")
    
    return False


def export_onnx(
    onnx_path: str,
    checkpoint_path: Optional[str] = None,
    H: int = 840,
    W: int = 840,
    verbose: bool = False,
) -> str:
    """
    Export MatchAnything model to ONNX with proper weight loading.
    
    Args:
        onnx_path: Output ONNX file path
        checkpoint_path: Path to model checkpoint
        H, W: Input image dimensions (should be multiples of 14)
        verbose: Enable verbose ONNX export
    
    Returns:
        Path to exported ONNX file
    """
    device = "cpu"
    
    print(f"[EXPORT] Creating model...")
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()
    
    # Load weights if checkpoint is provided
    weights_loaded = False
    if checkpoint_path:
        weights_loaded = load_weights_safely(model, checkpoint_path)
    
    if not weights_loaded:
        print("[EXPORT] ⚠️ Proceeding with random initialization")
        print("[EXPORT] Note: This will produce poor matching results")
    
    # Create dummy inputs (ensure dimensions are multiples of 14)
    H = ((H + 13) // 14) * 14  # Round up to nearest multiple of 14
    W = ((W + 13) // 14) * 14
    
    print(f"[EXPORT] Using input dimensions: {H}x{W}")
    x0 = torch.rand(1, 3, H, W, device=device)
    x1 = torch.rand(1, 3, H, W, device=device)
    
    # Test forward pass
    print("[EXPORT] Testing forward pass...")
    with torch.inference_mode():
        try:
            outputs = model(x0, x1)
            print("[EXPORT] ✅ Forward pass successful")
            print(f"[EXPORT] Output shapes:")
            for key, tensor in outputs.items():
                print(f"  {key}: {tuple(tensor.shape)}")
        except Exception as e:
            print(f"[EXPORT] ❌ Forward pass failed: {e}")
            raise
    
    # Prepare ONNX export
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    dynamic_axes = {
        "image0": {"0": "B", "2": "H", "3": "W"},
        "image1": {"0": "B", "2": "H", "3": "W"},
        "warp_c": {"0": "B", "1": "Ha", "2": "Wa"},
        "cert_c": {"0": "B", "1": "Ha", "2": "Wa"},
        "valid_mask": {"0": "B", "1": "Ha", "2": "Wa"},
        "coarse_stride": {"0": "one"},
    }
    
    export_kwargs = {
        "input_names": ["image0", "image1"],
        "output_names": ["warp_c", "cert_c", "valid_mask", "coarse_stride"],
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,
        "do_constant_folding": True,
        "verbose": verbose,
    }
    
    # Add external data format if available
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["use_external_data_format"] = True
    
    print(f"[EXPORT] Exporting to ONNX: {onnx_path}")
    try:
        torch.onnx.export(model, (x0, x1), str(onnx_path), **export_kwargs)
        print("[EXPORT] ✅ ONNX export successful")
    except Exception as e:
        print(f"[EXPORT] ❌ ONNX export failed: {e}")
        raise
    
    # Verify the exported model
    try:
        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        print(f"[EXPORT] ✅ ONNX model verification successful")
        print(f"[EXPORT] Model size: {onnx_path.stat().st_size / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"[EXPORT] ⚠️ ONNX model verification failed: {e}")
    
    return str(onnx_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export MatchAnything to ONNX (fixed version)")
    parser.add_argument("--onnx", default="out/matchanything_fixed.onnx", 
                       help="Output ONNX file path")
    parser.add_argument("--checkpoint", default="", 
                       help="Path to model checkpoint")
    parser.add_argument("--H", type=int, default=840, 
                       help="Input height (will be rounded to multiple of 14)")
    parser.add_argument("--W", type=int, default=840, 
                       help="Input width (will be rounded to multiple of 14)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose ONNX export")
    
    args = parser.parse_args()
    
    try:
        output_path = export_onnx(
            onnx_path=args.onnx,
            checkpoint_path=args.checkpoint if args.checkpoint else None,
            H=args.H,
            W=args.W,
            verbose=args.verbose,
        )
        print(f"\n[SUCCESS] ONNX model exported to: {output_path}")
    except Exception as e:
        print(f"\n[FAILURE] Export failed: {e}")
        sys.exit(1)