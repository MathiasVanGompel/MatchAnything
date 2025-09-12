#!/usr/bin/env python3
"""
Memory-optimized ONNX export for MatchAnything with comprehensive weight loading.
This version includes memory optimizations and better error handling to prevent OOM during export.
"""

import os
import sys
import inspect
import torch
import onnx
import gc
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


def optimize_model_for_export(model):
    """Apply memory optimizations to the model before export."""
    print("[MEMORY] Applying memory optimizations...")
    
    # Set model to eval mode and disable gradients
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable mixed precision if available
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("[MEMORY] ✅ Memory optimizations applied")


def comprehensive_weight_loading(model, checkpoint_path: str) -> bool:
    """
    Comprehensive weight loading with multiple strategies and memory optimization.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[WEIGHTS] Checkpoint not found: {checkpoint_path}")
        return False
    
    # Fix patch size before loading weights
    fix_patch_size_in_encoder(model)
    
    print(f"[WEIGHTS] Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint with memory mapping for large files
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
                # Clear memory after loading
                del loadable, checkpoint_dict, raw_checkpoint
                gc.collect()
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
        
        # Clean up checkpoint data
        del checkpoint_dict, raw_checkpoint, final_mapped, mapped_checkpoint
        del direct_loadable, suffix_loadable, final_loadable
        gc.collect()
        
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
    """Export MatchAnything model to ONNX with comprehensive weight loading and memory optimization."""
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
    
    # Apply memory optimizations
    optimize_model_for_export(model)
    
    # Create dummy inputs (ensure dimensions are multiples of 14, not 16!)
    patch_size = getattr(model.encoder, 'patch', 14)
    H = ((H + patch_size - 1) // patch_size) * patch_size  # Round up to nearest multiple
    W = ((W + patch_size - 1) // patch_size) * patch_size
    
    print(f"[EXPORT] Using input dimensions: {H}x{W} (patch size: {patch_size})")
    x0 = torch.rand(1, 3, H, W, device=device, dtype=torch.float32)
    x1 = torch.rand(1, 3, H, W, device=device, dtype=torch.float32)
    
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
    
    print(f"[EXPORT] Output directory created: {onnx_path.parent}")
    
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
    
    # Memory optimization: Force garbage collection before export
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[EXPORT] Exporting to ONNX: {onnx_path}")
    print("[EXPORT] This may take several minutes and use significant memory...")
    
    try:
        # Use torch.no_grad() to further reduce memory usage
        with torch.no_grad():
            print("[EXPORT] Starting torch.onnx.export...")
            torch.onnx.export(model, (x0, x1), str(onnx_path), **export_kwargs)
            print("[EXPORT] torch.onnx.export completed")
        
        # Check if file was actually created
        if onnx_path.exists():
            file_size = onnx_path.stat().st_size
            print(f"[EXPORT] ✅ ONNX file created successfully: {file_size / (1024*1024):.1f} MB")
        else:
            raise FileNotFoundError(f"ONNX file was not created at {onnx_path}")
            
        print("[EXPORT] ✅ ONNX export successful")
        
    except Exception as e:
        print(f"[EXPORT] ❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if partial file exists
        if onnx_path.exists():
            print(f"[EXPORT] Partial file found, size: {onnx_path.stat().st_size} bytes")
            print("[EXPORT] Removing partial file...")
            onnx_path.unlink()
        
        raise
    
    # Clean up input tensors
    del x0, x1, outputs
    gc.collect()
    
    # Verify the exported model
    try:
        print("[EXPORT] Verifying ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"[EXPORT] ✅ ONNX model verification successful")
        print(f"[EXPORT] Model size: {onnx_path.stat().st_size / (1024*1024):.1f} MB")
        del onnx_model
        gc.collect()
    except Exception as e:
        print(f"[EXPORT] ⚠️ ONNX model verification failed: {e}")
        print("[EXPORT] The model file exists but may have issues")
    
    return str(onnx_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export MatchAnything to ONNX (memory-optimized debug version)")
    parser.add_argument("--onnx", default="output/matchanything_optimized.onnx", 
                       help="Output ONNX file path")
    parser.add_argument("--checkpoint", default="", 
                       help="Path to model checkpoint")
    parser.add_argument("--H", type=int, default=560, 
                       help="Input height (will be rounded to multiple of 14)")
    parser.add_argument("--W", type=int, default=560, 
                       help="Input width (will be rounded to multiple of 14)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose ONNX export")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ONNX Export Debug Version")
    print("="*60)
    print(f"Output path: {args.onnx}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'None (random weights)'}")
    print(f"Input size: {args.H}x{args.W}")
    print(f"Verbose: {args.verbose}")
    print("="*60)
    
    try:
        output_path = export_onnx(
            onnx_path=args.onnx,
            checkpoint_path=args.checkpoint if args.checkpoint else None,
            H=args.H,
            W=args.W,
            verbose=args.verbose,
        )
        print(f"\n{'='*60}")
        print(f"[SUCCESS] ONNX model exported to: {output_path}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[FAILURE] Export failed: {e}")
        print(f"{'='*60}")
        sys.exit(1)