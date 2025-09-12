#!/usr/bin/env python3
"""
Unified weight loading system for MatchAnything TensorRT conversion.

This module consolidates all weight loading logic into a single, robust function
that handles the complex mapping between MatchAnything checkpoint structure and
the TensorRT model architecture, including proper DINOv2 weight loading.

Based on the analysis in WEIGHT_LOADING_ANALYSIS.md, this replaces the multiple
weight adapter scripts with a single, comprehensive solution.
"""

import re
import torch
import timm
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import torch.nn.functional as F


def create_comprehensive_mapping_rules() -> List[Tuple[re.Pattern, str]]:
    """
    Create comprehensive mapping rules for MatchAnything checkpoint structure.
    
    These rules handle the complex nesting structure where DINOv2 weights are
    stored under 'embedding_decoder' in the checkpoint but expected under
    'encoder.dino' in the model.
    """
    return [
        # Remove common wrappers first
        (re.compile(r"^module\."), ""),
        (re.compile(r"^_orig_mod\."), ""),
        
        # MatchAnything specific structure - CRITICAL MAPPINGS
        # The embedding_decoder contains the DINOv2 ViT weights
        (re.compile(r"^matcher\.model\.decoder\.embedding_decoder\."), "encoder.dino."),
        (re.compile(r"^embedding_decoder\."), "encoder.dino."),
        (re.compile(r"^decoder\.embedding_decoder\."), "encoder.dino."),
        
        # CNN encoder mappings
        (re.compile(r"^matcher\.model\.encoder\.cnn\.layers\."), "encoder.layers."),
        (re.compile(r"^matcher\.model\.encoder\.cnn\."), "encoder."),
        (re.compile(r"^matcher\.model\.encoder\."), "encoder."),
        
        # Matcher/decoder mappings
        (re.compile(r"^matcher\.model\.decoder\."), "matcher."),
        (re.compile(r"^matcher\.model\."), ""),
        (re.compile(r"^matcher\."), ""),
        (re.compile(r"^model\."), ""),
        
        # Handle various DINOv2 naming patterns (fallbacks)
        (re.compile(r"^backbone\."), "encoder.dino."),
        (re.compile(r"^vit\."), "encoder.dino."),
        (re.compile(r"^dino\."), "encoder.dino."),
        (re.compile(r"^encoder\.vit\."), "encoder.dino."),
        (re.compile(r"^encoder\.backbone\."), "encoder.dino."),
        
        # Identity mappings (no change needed)
        (re.compile(r"^encoder\.dino\."), "encoder.dino."),
        (re.compile(r"^encoder\."), "encoder."),
    ]


def fix_dinov2_block_structure(weights_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Fix DINOv2 transformer block naming structure.
    
    The checkpoint has keys like: encoder.dino.blocks.5.attn.qkv.weight
    But the model expects: encoder.dino.blocks.5.0.attn.qkv.weight
    
    This function inserts the missing '.0' index after each block number.
    """
    fixed_weights = {}
    
    for key, value in weights_dict.items():
        if key.startswith('encoder.dino.blocks.'):
            # Pattern: encoder.dino.blocks.N.something -> encoder.dino.blocks.N.0.something
            parts = key.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                # Check if .0 is already present
                if len(parts) >= 5 and parts[4] == '0':
                    # Already has the .0, keep as is
                    fixed_weights[key] = value
                else:
                    # Insert .0 after the block number
                    new_parts = parts[:4] + ['0'] + parts[4:]
                    new_key = '.'.join(new_parts)
                    fixed_weights[new_key] = value
                    print(f"[UNIFIED] Fixed block structure: {key} -> {new_key}")
            else:
                fixed_weights[key] = value
        else:
            fixed_weights[key] = value
    
    return fixed_weights


def resize_positional_embedding(pos_embed: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Resize DINOv2 positional embedding from pre-training resolution to target resolution.
    
    DINOv2 was pre-trained at 518x518 (37x37 patches = 1369 spatial + 1 CLS = 1370 tokens).
    For 224x224 input (14x14 patches), we need 196 spatial + 1 CLS = 197 tokens.
    
    Args:
        pos_embed: Original positional embedding [1, N, D] where N=1370
        target_size: Target sequence length (e.g., 197 for 224x224)
    
    Returns:
        Resized positional embedding [1, target_size, D]
    """
    if pos_embed.shape[1] == target_size:
        print(f"[UNIFIED] Positional embedding already correct size: {pos_embed.shape}")
        return pos_embed
    
    print(f"[UNIFIED] Resizing positional embedding: {pos_embed.shape} -> [1, {target_size}, {pos_embed.shape[2]}]")
    
    # Extract CLS token (first token) and spatial tokens
    cls_token = pos_embed[:, 0:1, :]  # [1, 1, D]
    spatial_tokens = pos_embed[:, 1:, :]  # [1, N-1, D]
    
    # Calculate original spatial grid size
    num_spatial = spatial_tokens.shape[1]
    original_size = int(np.sqrt(num_spatial))  # Should be 37 for DINOv2
    
    if original_size * original_size != num_spatial:
        raise ValueError(f"Cannot reshape spatial tokens {num_spatial} into square grid")
    
    # Calculate target spatial grid size
    target_spatial = target_size - 1  # Subtract 1 for CLS token
    target_grid_size = int(np.sqrt(target_spatial))  # Should be 14 for 224x224
    
    if target_grid_size * target_grid_size != target_spatial:
        raise ValueError(f"Target size {target_size} does not correspond to square spatial grid")
    
    # Reshape spatial tokens to 2D grid and interpolate
    D = spatial_tokens.shape[2]
    spatial_2d = spatial_tokens.reshape(1, original_size, original_size, D)
    spatial_2d = spatial_2d.permute(0, 3, 1, 2)  # [1, D, H, W]
    
    # Interpolate to target size
    spatial_resized = F.interpolate(
        spatial_2d, 
        size=(target_grid_size, target_grid_size), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Reshape back to sequence format
    spatial_resized = spatial_resized.permute(0, 2, 3, 1)  # [1, H, W, D]
    spatial_resized = spatial_resized.reshape(1, target_spatial, D)  # [1, target_spatial, D]
    
    # Concatenate CLS token and resized spatial tokens
    resized_pos_embed = torch.cat([cls_token, spatial_resized], dim=1)
    
    print(f"[UNIFIED] Successfully resized positional embedding to {resized_pos_embed.shape}")
    return resized_pos_embed


def load_dinov2_components_func(model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Load missing DINOv2 components from official pre-trained weights.
    
    This ensures we have proper cls_token, pos_embed, mask_token, and patch_embed
    if they're missing or need to be resized.
    """
    dinov2_weights = {}
    
    try:
        print("[UNIFIED] Loading official DINOv2 weights for missing components...")
        dinov2_model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True)
        official_weights = dinov2_model.state_dict()
        
        # Check what components we need
        needs_pos_embed = 'encoder.dino.pos_embed' in model_state_dict
        needs_cls_token = 'encoder.dino.cls_token' in model_state_dict
        needs_mask_token = 'encoder.dino.mask_token' in model_state_dict
        needs_patch_embed = any(k.startswith('encoder.dino.patch_embed.') for k in model_state_dict.keys())
        
        if needs_pos_embed and 'pos_embed' in official_weights:
            pos_embed = official_weights['pos_embed']
            # Determine target size from model
            target_pos_embed = model_state_dict['encoder.dino.pos_embed']
            if pos_embed.shape != target_pos_embed.shape:
                pos_embed = resize_positional_embedding(pos_embed, target_pos_embed.shape[1])
            dinov2_weights['encoder.dino.pos_embed'] = pos_embed
            print(f"[UNIFIED] Added pos_embed: {pos_embed.shape}")
        
        if needs_cls_token and 'cls_token' in official_weights:
            dinov2_weights['encoder.dino.cls_token'] = official_weights['cls_token']
            print(f"[UNIFIED] Added cls_token: {official_weights['cls_token'].shape}")
        
        if needs_mask_token and 'mask_token' in official_weights:
            dinov2_weights['encoder.dino.mask_token'] = official_weights['mask_token']
            print(f"[UNIFIED] Added mask_token: {official_weights['mask_token'].shape}")
        
        if needs_patch_embed:
            for key, value in official_weights.items():
                if key.startswith('patch_embed.'):
                    new_key = f'encoder.dino.{key}'
                    if new_key in model_state_dict:
                        dinov2_weights[new_key] = value
                        print(f"[UNIFIED] Added {new_key}: {value.shape}")
        
    except Exception as e:
        print(f"[UNIFIED] Warning: Could not load official DINOv2 weights: {e}")
        print("[UNIFIED] Proceeding with checkpoint weights only...")
    
    return dinov2_weights


def apply_unified_weight_loading(
    checkpoint_path: str, 
    model_state_dict: Dict[str, torch.Tensor],
    load_dinov2_components: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Unified weight loading function that handles all MatchAnything checkpoint complexities.
    
    This function:
    1. Loads the checkpoint and applies comprehensive mapping rules
    2. Fixes DINOv2 block structure (inserts missing .0 indices)
    3. Handles positional embedding resizing
    4. Loads missing DINOv2 components from official weights
    5. Uses suffix matching for any remaining unmatched weights
    
    Args:
        checkpoint_path: Path to the MatchAnything checkpoint
        model_state_dict: Target model's state_dict
        load_dinov2_components: Whether to load missing components from official DINOv2
    
    Returns:
        Dictionary of weights that can be loaded into the model
    """
    print(f"[UNIFIED] Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint with error handling for different formats
    try:
        raw = torch.load(checkpoint_path, map_location="cpu")
        ckpt_state_dict = raw.get("state_dict", raw)
    except Exception as e:
        print(f"[UNIFIED] Error loading with torch.load: {e}")
        print("[UNIFIED] Trying alternative loading methods...")
        
        try:
            # Try loading with weights_only=False for older checkpoints
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_state_dict = raw.get("state_dict", raw)
            print("[UNIFIED] Successfully loaded with weights_only=False")
        except Exception as e2:
            print(f"[UNIFIED] Alternative loading also failed: {e2}")
            
            try:
                # Try loading as a pickle file directly
                import pickle
                with open(checkpoint_path, 'rb') as f:
                    raw = pickle.load(f)
                ckpt_state_dict = raw.get("state_dict", raw)
                print("[UNIFIED] Successfully loaded as pickle file")
            except Exception as e3:
                print(f"[UNIFIED] Pickle loading also failed: {e3}")
                print("[UNIFIED] Cannot load checkpoint - unsupported format")
                return {}
    
    print(f"[UNIFIED] Checkpoint has {len(ckpt_state_dict)} keys")
    print(f"[UNIFIED] Model expects {len(model_state_dict)} keys")
    
    # Analyze checkpoint structure
    prefixes = {}
    for key in ckpt_state_dict.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = f"{parts[0]}.{parts[1]}"
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print("[UNIFIED] Top checkpoint key prefixes:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {prefix}: {count} keys")
    
    # Step 1: Apply comprehensive mapping rules
    print("[UNIFIED] Step 1: Applying mapping rules...")
    rules = create_comprehensive_mapping_rules()
    mapped_weights = {}
    
    for ckpt_key, ckpt_value in ckpt_state_dict.items():
        mapped_key = ckpt_key
        
        # Apply each rule in sequence
        for pattern, replacement in rules:
            mapped_key = pattern.sub(replacement, mapped_key)
        
        mapped_weights[mapped_key] = ckpt_value
    
    print(f"[UNIFIED] After mapping: {len(mapped_weights)} keys")
    
    # Step 2: Fix DINOv2 block structure
    print("[UNIFIED] Step 2: Fixing DINOv2 block structure...")
    mapped_weights = fix_dinov2_block_structure(mapped_weights)
    
    # Step 3: Load missing DINOv2 components if requested
    dinov2_components = {}
    if load_dinov2_components:
        print("[UNIFIED] Step 3: Loading missing DINOv2 components...")
        dinov2_components = load_dinov2_components_func(model_state_dict)
        mapped_weights.update(dinov2_components)
    
    # Step 4: Direct matching
    print("[UNIFIED] Step 4: Attempting direct matches...")
    loadable = {}
    shape_mismatches = []
    
    for model_key in model_state_dict.keys():
        if model_key in mapped_weights:
            ckpt_tensor = mapped_weights[model_key]
            model_tensor = model_state_dict[model_key]
            
            if ckpt_tensor.shape == model_tensor.shape:
                loadable[model_key] = ckpt_tensor
            else:
                shape_mismatches.append((model_key, ckpt_tensor.shape, model_tensor.shape))
    
    print(f"[UNIFIED] Direct matches: {len(loadable)} weights")
    
    # Step 5: Suffix matching for remaining keys
    remaining_model_keys = set(model_state_dict.keys()) - set(loadable.keys())
    remaining_ckpt_keys = set(mapped_weights.keys()) - set(loadable.keys())
    
    if remaining_model_keys:
        print(f"[UNIFIED] Step 5: Trying suffix matching for {len(remaining_model_keys)} remaining keys...")
        
        for model_key in remaining_model_keys:
            # Use last 2 components as suffix for matching
            model_suffix = model_key.split('.')[-2:] if '.' in model_key else [model_key]
            model_suffix_str = '.'.join(model_suffix)
            
            for ckpt_key in remaining_ckpt_keys:
                if ckpt_key.endswith(model_suffix_str):
                    ckpt_tensor = mapped_weights[ckpt_key]
                    model_tensor = model_state_dict[model_key]
                    
                    if ckpt_tensor.shape == model_tensor.shape:
                        loadable[model_key] = ckpt_tensor
                        print(f"[UNIFIED] Suffix match: {ckpt_key} -> {model_key}")
                        break
    
    # Step 6: Report results
    print("[UNIFIED] === LOADING SUMMARY ===")
    print(f"Total weights loaded: {len(loadable)} / {len(model_state_dict)} ({100.0 * len(loadable) / len(model_state_dict):.1f}%)")
    
    # Analyze what was loaded
    dino_loaded = sum(1 for k in loadable.keys() if k.startswith('encoder.dino.'))
    dino_total = sum(1 for k in model_state_dict.keys() if k.startswith('encoder.dino.'))
    encoder_loaded = sum(1 for k in loadable.keys() if k.startswith('encoder.') and not k.startswith('encoder.dino.'))
    encoder_total = sum(1 for k in model_state_dict.keys() if k.startswith('encoder.') and not k.startswith('encoder.dino.'))
    matcher_loaded = sum(1 for k in loadable.keys() if k.startswith('matcher.'))
    matcher_total = sum(1 for k in model_state_dict.keys() if k.startswith('matcher.'))
    
    print(f"DINOv2 weights: {dino_loaded} / {dino_total} ({100.0 * dino_loaded / max(dino_total, 1):.1f}%)")
    print(f"CNN encoder weights: {encoder_loaded} / {encoder_total} ({100.0 * encoder_loaded / max(encoder_total, 1):.1f}%)")
    print(f"Matcher weights: {matcher_loaded} / {matcher_total} ({100.0 * matcher_loaded / max(matcher_total, 1):.1f}%)")
    
    # Report shape mismatches
    if shape_mismatches:
        print(f"\nShape mismatches: {len(shape_mismatches)}")
        for key, ckpt_shape, model_shape in shape_mismatches[:5]:  # Show first 5
            print(f"  {key}: checkpoint {ckpt_shape} vs model {model_shape}")
        if len(shape_mismatches) > 5:
            print(f"  ... and {len(shape_mismatches) - 5} more")
    
    # Report missing keys
    missing_keys = set(model_state_dict.keys()) - set(loadable.keys())
    if missing_keys:
        print(f"\nMissing keys: {len(missing_keys)}")
        missing_dino = [k for k in missing_keys if k.startswith('encoder.dino.')]
        if missing_dino:
            print(f"Missing DINOv2 keys: {len(missing_dino)}")
            for key in sorted(missing_dino)[:5]:  # Show first 5
                print(f"  {key}")
            if len(missing_dino) > 5:
                print(f"  ... and {len(missing_dino) - 5} more")
    
    # Success criteria
    if len(loadable) >= 0.95 * len(model_state_dict):
        print("\n✅ SUCCESS: >95% of weights loaded successfully!")
    elif len(loadable) >= 0.8 * len(model_state_dict):
        print("\n⚠️  WARNING: 80-95% of weights loaded. Some components may be missing.")
    else:
        print("\n❌ ERROR: <80% of weights loaded. Major components are missing!")
    
    return loadable


# Convenience function for backward compatibility
def remap_and_load(model, checkpoint_path: str, **kwargs) -> Dict[str, torch.Tensor]:
    """Backward compatibility wrapper for the old remap_and_load function."""
    return apply_unified_weight_loading(checkpoint_path, model.state_dict(), **kwargs)


if __name__ == "__main__":
    print("Unified Weight Loader for MatchAnything TensorRT Conversion")
    print("This module provides comprehensive weight loading with proper DINOv2 handling.")