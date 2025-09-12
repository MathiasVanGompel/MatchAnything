#!/usr/bin/env python3
"""
Minimal memory ONNX export for MatchAnything.
Uses aggressive memory optimization techniques to prevent OOM.
"""

import os
import sys
import torch
import gc
from pathlib import Path
from typing import Optional

# Add parent directory to path
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt"
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT


def minimal_weight_loading(model, checkpoint_path: str) -> bool:
    """Minimal weight loading to save memory."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return False
    
    print(f"[WEIGHTS] Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load only state_dict to save memory
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        
        # Quick prefix mapping
        mapped = {}
        for k, v in checkpoint.items():
            # Remove common prefixes
            key = k
            for prefix in ["module.", "_orig_mod.", "matcher.model.", "matcher.", "model."]:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            
            # Map to encoder.dino if needed
            if not key.startswith("encoder.") and not key.startswith("matcher."):
                if any(x in key for x in ["backbone", "vit", "dino", "blocks", "patch_embed", "norm"]):
                    key = "encoder.dino." + key
            
            mapped[key] = v
        
        # Load with strict=False
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        loaded = len(mapped) - len(missing)
        total = len(model.state_dict())
        
        print(f"[WEIGHTS] Loaded {loaded}/{total} ({100*loaded/total:.1f}%)")
        
        # Clean up immediately
        del checkpoint, mapped
        gc.collect()
        
        return loaded / total > 0.5  # 50% threshold
        
    except Exception as e:
        print(f"[WEIGHTS] Loading failed: {e}")
        return False


def export_minimal_memory(
    onnx_path: str,
    checkpoint_path: Optional[str] = None,
    H: int = 560,  # Smaller default size
    W: int = 560,
) -> str:
    """Export with minimal memory usage."""
    
    print("[EXPORT] Creating model with minimal memory...")
    
    # Force CPU and disable autograd
    device = "cpu"
    torch.set_grad_enabled(False)
    
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()
    
    # Disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Load weights if provided
    if checkpoint_path:
        minimal_weight_loading(model, checkpoint_path)
    
    # Fix patch size
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'patch'):
        model.encoder.patch = 14
    
    # Use smaller input size to reduce memory
    patch_size = 14
    H = ((H + patch_size - 1) // patch_size) * patch_size
    W = ((W + patch_size - 1) // patch_size) * patch_size
    
    print(f"[EXPORT] Using minimal input size: {H}x{W}")
    
    # Create inputs with explicit memory management
    x0 = torch.rand(1, 3, H, W, device=device, dtype=torch.float32, requires_grad=False)
    x1 = torch.rand(1, 3, H, W, device=device, dtype=torch.float32, requires_grad=False)
    
    # Test forward pass
    print("[EXPORT] Testing forward pass...")
    with torch.no_grad(), torch.inference_mode():
        try:
            outputs = model(x0, x1)
            print("[EXPORT] ✅ Forward pass successful")
            # Clean up outputs immediately
            del outputs
            gc.collect()
        except Exception as e:
            print(f"[EXPORT] ❌ Forward pass failed: {e}")
            raise
    
    # Prepare export directory
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Minimal export settings
    export_kwargs = {
        "input_names": ["image0", "image1"],
        "output_names": ["warp_c", "cert_c", "valid_mask", "coarse_stride"],
        "opset_version": 11,  # Lower opset for compatibility
        "do_constant_folding": False,  # Disable to save memory
        "verbose": False,
        "training": torch.onnx.TrainingMode.EVAL,
        "export_params": True,
    }
    
    # Force garbage collection before export
    gc.collect()
    
    print(f"[EXPORT] Exporting to ONNX (minimal memory mode): {onnx_path}")
    
    try:
        with torch.no_grad():
            torch.onnx.export(model, (x0, x1), str(onnx_path), **export_kwargs)
        print("[EXPORT] ✅ ONNX export successful")
        
        # Check file size
        size_mb = onnx_path.stat().st_size / (1024*1024)
        print(f"[EXPORT] Model size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"[EXPORT] ❌ ONNX export failed: {e}")
        raise
    finally:
        # Clean up
        del x0, x1, model
        gc.collect()
    
    return str(onnx_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal memory ONNX export")
    parser.add_argument("--onnx", default="output/matchanything_minimal.onnx")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--H", type=int, default=560)
    parser.add_argument("--W", type=int, default=560)
    
    args = parser.parse_args()
    
    try:
        output_path = export_minimal_memory(
            onnx_path=args.onnx,
            checkpoint_path=args.checkpoint if args.checkpoint else None,
            H=args.H,
            W=args.W,
        )
        print(f"\n[SUCCESS] Minimal memory export to: {output_path}")
    except Exception as e:
        print(f"\n[FAILURE] Export failed: {e}")
        sys.exit(1)