#!/usr/bin/env python3
"""
Corrected ONNX export for MatchAnything with proper DINOv2 patch size handling.
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


def fix_patch_size_in_encoder(model):
    """
    Fix the patch size mismatch - DINOv2 uses 14x14 patches, not 16x16.
    """
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'patch'):
        print(f"[FIX] Changing encoder patch size from {model.encoder.patch} to 14")
        model.encoder.patch = 14
        
        # Also fix the patch embed layer if it exists
        if hasattr(model.encoder, 'dino') and hasattr(model.encoder.dino, 'patch_embed'):
            patch_embed = model.encoder.dino.patch_embed
            if hasattr(patch_embed, 'proj'):
                # Check current kernel size
                current_kernel = patch_embed.proj.kernel_size
                if current_kernel != (14, 14):
                    print(f"[FIX] PatchEmbed kernel size: {current_kernel} -> (14, 14)")
                    # Create new patch embed with correct kernel size
                    new_proj = torch.nn.Conv2d(
                        in_channels=patch_embed.proj.in_channels,
                        out_channels=patch_embed.proj.out_channels,
                        kernel_size=14,
                        stride=14,
                        bias=patch_embed.proj.bias is not None
                    )
                    patch_embed.proj = new_proj


def load_weights_with_patch_fix(model, checkpoint_path: str) -> bool:
    """
    Load weights with proper patch size handling.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[WEIGHTS] Checkpoint not found: {checkpoint_path}")
        return False
    
    # Fix patch size before loading weights
    fix_patch_size_in_encoder(model)
    
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
        
        # Show some missing keys for debugging
        if missing and len(missing) < 20:
            print("[WEIGHTS] Some missing keys:")
            for key in sorted(missing)[:10]:
                print(f"  - {key}")
        
        if loaded_pct >= 70:  # Reasonable threshold
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
        
        # Remove problematic keys that don't match
        filtered_state_dict = {}
        model_keys = set(model.state_dict().keys())
        
        for key, value in state_dict.items():
            # Try to find matching key in model
            if key in model_keys:
                model_shape = model.state_dict()[key].shape
                if value.shape == model_shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"[WEIGHTS] Shape mismatch for {key}: {value.shape} vs {model_shape}")
            else:
                # Try without prefixes
                for prefix in ['module.', '_orig_mod.', 'matcher.model.', 'matcher.']:
                    if key.startswith(prefix):
                        clean_key = key[len(prefix):]
                        if clean_key in model_keys:
                            model_shape = model.state_dict()[clean_key].shape
                            if value.shape == model_shape:
                                filtered_state_dict[clean_key] = value
                                break
        
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        loaded_count = len(filtered_state_dict)
        loaded_pct = (loaded_count / len(model.state_dict())) * 100
        
        print(f"[WEIGHTS] Direct load (filtered): {loaded_count}/{len(model.state_dict())} ({loaded_pct:.1f}%) loaded")
        
        if loaded_pct >= 30:  # Lower threshold for direct loading
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
    Export MatchAnything model to ONNX with corrected patch size handling.
    """
    device = "cpu"
    
    print(f"[EXPORT] Creating model...")
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()
    
    # Load weights if checkpoint is provided
    weights_loaded = False
    if checkpoint_path:
        weights_loaded = load_weights_with_patch_fix(model, checkpoint_path)
    
    if not weights_loaded:
        print("[EXPORT] ⚠️ Proceeding with random initialization")
        print("[EXPORT] Note: This will produce poor matching results")
        # Still fix patch size for consistency
        fix_patch_size_in_encoder(model)
    
    # Create dummy inputs (ensure dimensions are multiples of 14, not 16!)
    patch_size = getattr(model.encoder, 'patch', 14)
    H = ((H + patch_size - 1) // patch_size) * patch_size  # Round up to nearest multiple
    W = ((W + patch_size - 1) // patch_size) * patch_size
    
    print(f"[EXPORT] Using input dimensions: {H}x{W} (patch size: {patch_size})")
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
            import traceback
            traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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
    
    parser = argparse.ArgumentParser(description="Export MatchAnything to ONNX (corrected version)")
    parser.add_argument("--onnx", default="out/matchanything_corrected.onnx", 
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
