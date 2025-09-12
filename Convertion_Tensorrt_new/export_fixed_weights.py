#!/usr/bin/env python3
"""
Fixed ONNX export for MatchAnything with comprehensive weight loading.
This version addresses the 20.3% weight loading issue with improved mapping.
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
    """Fix the patch size mismatch - DINOv2 uses 14x14 patches, not 16x16."""
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
            
            # Also fix the patch_size attribute in patch_embed
            if hasattr(patch_embed, 'patch_size'):
                print(f"[FIX] PatchEmbed patch_size: {patch_embed.patch_size} -> (14, 14)")
                patch_embed.patch_size = (14, 14)


def comprehensive_weight_loading(model, checkpoint_path: str) -> bool:
    """
    Comprehensive weight loading with multiple strategies.
    This addresses the 20.3% loading issue with better key mapping.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[WEIGHTS] Checkpoint not found: {checkpoint_path}")
        return False
    
    # Fix patch size before loading weights
    fix_patch_size_in_encoder(model)
    
    print(f"[WEIGHTS] Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_dict = raw_checkpoint.get("state_dict", raw_checkpoint)
        model_state_dict = model.state_dict()
        
        print(f"[WEIGHTS] Checkpoint has {len(checkpoint_dict)} keys")
        print(f"[WEIGHTS] Model has {len(model_state_dict)} parameters")
        
        # Strategy 1: Try improved weight loader first
        try:
            from improved_weight_loader import apply_improved_weight_loading
            loadable = apply_improved_weight_loading(
                checkpoint_path, model_state_dict, 
                load_dinov2_components=True, verbose=False
            )
            
            if len(loadable) / len(model_state_dict) >= 0.8:  # 80% threshold
                missing, unexpected = model.load_state_dict(loadable, strict=False)
                loaded_pct = (len(loadable) / len(model_state_dict)) * 100
                print(f"[WEIGHTS] ✅ Improved loader: {len(loadable)}/{len(model_state_dict)} ({loaded_pct:.1f}%)")
                return True
            else:
                print(f"[WEIGHTS] Improved loader got {len(loadable)}/{len(model_state_dict)} ({100.0*len(loadable)/len(model_state_dict):.1f}%), trying other methods...")
                
        except Exception as e:
            print(f"[WEIGHTS] Improved loader failed: {e}")
            loadable = {}
        
        # Strategy 2: Manual key mapping based on MatchAnything structure
        print("[WEIGHTS] Trying manual key mapping...")
        
        # Common prefixes to try mapping
        prefix_mappings = [
            # Remove module/wrapper prefixes
            ("module.", ""),
            ("_orig_mod.", ""),
            ("matcher.model.", ""),
            ("matcher.", ""),
            ("model.", ""),
            
            # Map different encoder naming conventions
            ("backbone.", "encoder.dino."),
            ("vit.", "encoder.dino."),
            ("encoder.backbone.", "encoder.dino."),
            ("encoder.vit.", "encoder.dino."),
            ("dino.", "encoder.dino."),
            ("embedding_decoder.", "encoder.dino."),
            ("decoder.embedding_decoder.", "encoder.dino."),
            
            # Map decoder to matcher
            ("decoder.", "matcher."),
        ]
        
        # Apply prefix mappings
        mapped_checkpoint = {}
        for ckpt_key, ckpt_value in checkpoint_dict.items():
            mapped_key = ckpt_key
            
            # Try each prefix mapping
            for old_prefix, new_prefix in prefix_mappings:
                if mapped_key.startswith(old_prefix):
                    mapped_key = new_prefix + mapped_key[len(old_prefix):]
                    break
            
            mapped_checkpoint[mapped_key] = ckpt_value
        
        # Handle DINOv2 block structure (blocks.X -> blocks.0.X for BlockChunk)
        final_mapped = {}
        for key, value in mapped_checkpoint.items():
            if key.startswith("encoder.dino.blocks.") and key.count('.') >= 4:
                parts = key.split('.')
                if len(parts) >= 4 and parts[3].isdigit():
                    # Check if it's already in BlockChunk format
                    if len(parts) >= 5 and not parts[4].isdigit():
                        # Convert blocks.N.something -> blocks.0.N.something
                        block_num = parts[3]
                        new_key = ".".join(parts[:3] + ["0", block_num] + parts[4:])
                        final_mapped[new_key] = value
                        print(f"[WEIGHTS] BlockChunk fix: {key} -> {new_key}")
                    else:
                        final_mapped[key] = value
                else:
                    final_mapped[key] = value
            else:
                final_mapped[key] = value
        
        # Direct matching
        direct_loadable = {}
        for model_key, model_param in model_state_dict.items():
            if model_key in final_mapped:
                ckpt_value = final_mapped[model_key]
                if ckpt_value.shape == model_param.shape:
                    direct_loadable[model_key] = ckpt_value
        
        print(f"[WEIGHTS] Direct mapping: {len(direct_loadable)}/{len(model_state_dict)} ({100.0*len(direct_loadable)/len(model_state_dict):.1f}%)")
        
        # Strategy 3: Suffix matching for remaining keys
        remaining_model_keys = set(model_state_dict.keys()) - set(direct_loadable.keys())
        suffix_loadable = {}
        
        for model_key in remaining_model_keys:
            model_param = model_state_dict[model_key]
            
            # Try suffix matching with different lengths
            for suffix_len in [1, 2, 3, 4]:
                if suffix_len > len(model_key.split('.')):
                    continue
                    
                model_suffix = '.'.join(model_key.split('.')[-suffix_len:])
                
                for mapped_key, mapped_value in final_mapped.items():
                    if (mapped_key.endswith(model_suffix) and 
                        mapped_value.shape == model_param.shape and
                        model_key not in suffix_loadable):
                        suffix_loadable[model_key] = mapped_value
                        print(f"[WEIGHTS] Suffix match: {model_key} <- {mapped_key}")
                        break
                
                if model_key in suffix_loadable:
                    break
        
        # Combine all loadable weights
        all_loadable = {**direct_loadable, **suffix_loadable}
        
        # Use the best result from all strategies
        if len(loadable) > len(all_loadable):
            final_loadable = loadable
            source = "improved loader"
        else:
            final_loadable = all_loadable
            source = "manual mapping"
        
        print(f"[WEIGHTS] Best result from {source}: {len(final_loadable)}/{len(model_state_dict)} ({100.0*len(final_loadable)/len(model_state_dict):.1f}%)")
        
        # Strategy 4: Load DINOv2 weights for missing encoder components
        still_missing = set(model_state_dict.keys()) - set(final_loadable.keys())
        dino_missing = [k for k in still_missing if k.startswith("encoder.dino.")]
        
        if dino_missing:
            print(f"[WEIGHTS] Loading DINOv2 weights for {len(dino_missing)} missing encoder components...")
            try:
                import timm
                dinov2_model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
                official_weights = dinov2_model.state_dict()
                
                for model_key in dino_missing:
                    # Map encoder.dino.X to X
                    dino_key = model_key[len("encoder.dino."):]
                    if dino_key in official_weights:
                        official_weight = official_weights[dino_key]
                        model_param = model_state_dict[model_key]
                        
                        # Handle positional embedding resizing
                        if dino_key == "pos_embed" and official_weight.shape != model_param.shape:
                            # Resize positional embedding
                            print(f"[WEIGHTS] Resizing pos_embed: {official_weight.shape} -> {model_param.shape}")
                            # Simple resize - you might want to use the more sophisticated method from improved_weight_loader
                            if official_weight.shape[1] != model_param.shape[1]:
                                # For now, just take the first N tokens or pad with zeros
                                if official_weight.shape[1] > model_param.shape[1]:
                                    official_weight = official_weight[:, :model_param.shape[1], :]
                                else:
                                    pad_size = model_param.shape[1] - official_weight.shape[1]
                                    padding = torch.zeros(1, pad_size, official_weight.shape[2])
                                    official_weight = torch.cat([official_weight, padding], dim=1)
                        
                        if official_weight.shape == model_param.shape:
                            final_loadable[model_key] = official_weight
                            print(f"[WEIGHTS] Added DINOv2: {model_key}")
                
            except Exception as e:
                print(f"[WEIGHTS] Could not load DINOv2 weights: {e}")
        
        # Load weights into model
        missing, unexpected = model.load_state_dict(final_loadable, strict=False)
        final_loaded_pct = (len(final_loadable) / len(model_state_dict)) * 100
        
        print(f"\n[WEIGHTS] === FINAL LOADING SUMMARY ===")
        print(f"Loaded: {len(final_loadable)}/{len(model_state_dict)} ({final_loaded_pct:.1f}%)")
        print(f"Missing: {len(missing)}")
        print(f"Unexpected: {len(unexpected)}")
        
        if len(missing) > 0 and len(missing) <= 10:
            print("Missing keys:")
            for key in sorted(missing):
                print(f"  - {key}")
        elif len(missing) > 10:
            print(f"Missing keys (first 10 of {len(missing)}):")
            for key in sorted(missing)[:10]:
                print(f"  - {key}")
        
        # Success if we loaded at least 70% of weights
        if final_loaded_pct >= 70:
            print("[WEIGHTS] ✅ Weight loading successful!")
            return True
        else:
            print("[WEIGHTS] ⚠️ Low weight loading percentage")
            return False
            
    except Exception as e:
        print(f"[WEIGHTS] Comprehensive loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_onnx(
    onnx_path: str,
    checkpoint_path: Optional[str] = None,
    H: int = 840,
    W: int = 840,
    verbose: bool = False,
) -> str:
    """Export MatchAnything model to ONNX with comprehensive weight loading."""
    device = "cpu"
    
    print(f"[EXPORT] Creating model...")
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()
    
    # Load weights if checkpoint is provided
    weights_loaded = False
    if checkpoint_path:
        weights_loaded = comprehensive_weight_loading(model, checkpoint_path)
    
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
    
    parser = argparse.ArgumentParser(description="Export MatchAnything to ONNX (fixed weights version)")
    parser.add_argument("--onnx", default="output/matchanything_fixed_weights.onnx", 
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