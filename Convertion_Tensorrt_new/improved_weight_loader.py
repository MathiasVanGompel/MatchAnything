#!/usr/bin/env python3
"""
Improved weight loader for MatchAnything that handles proper key mapping.
Based on analysis of the MatchAnything ROMA architecture.
"""

import re
import torch
import timm
import numpy as np
from typing import Dict, List, Tuple, Set
import torch.nn.functional as F


def create_comprehensive_mapping_rules() -> List[Tuple[re.Pattern, str]]:
    """Create comprehensive mapping rules for MatchAnything checkpoint structure."""
    return [
        # Remove common prefixes
        (re.compile(r"^module\."), ""),
        (re.compile(r"^_orig_mod\."), ""),
        
        # MatchAnything specific mappings
        (re.compile(r"^matcher\.model\."), ""),
        (re.compile(r"^matcher\."), ""),
        (re.compile(r"^model\."), ""),
        
        # Encoder mappings - these are critical for MatchAnything
        (re.compile(r"^encoder\.cnn\."), "encoder."),  # CNN encoder parts
        (re.compile(r"^encoder\.backbone\."), "encoder.dino."),  # Backbone -> DINOv2
        (re.compile(r"^encoder\.vit\."), "encoder.dino."),  # ViT -> DINOv2
        (re.compile(r"^backbone\."), "encoder.dino."),  # Direct backbone -> DINOv2
        (re.compile(r"^vit\."), "encoder.dino."),  # Direct ViT -> DINOv2
        (re.compile(r"^dino\."), "encoder.dino."),  # Direct dino -> encoder.dino
        
        # Decoder/matcher mappings
        (re.compile(r"^decoder\.embedding_decoder\."), "encoder.dino."),  # Some decoders are actually encoders
        (re.compile(r"^embedding_decoder\."), "encoder.dino."),
        (re.compile(r"^decoder\."), "matcher."),  # True decoder -> matcher
        
        # Keep encoder.dino as is (already correct)
        (re.compile(r"^encoder\.dino\."), "encoder.dino."),
        (re.compile(r"^encoder\."), "encoder."),
    ]


def create_reverse_mapping_rules() -> List[Tuple[re.Pattern, str]]:
    """Create reverse mapping rules - try to map model keys to checkpoint keys."""
    return [
        # Try to find encoder.dino keys in various checkpoint locations
        (re.compile(r"^encoder\.dino\."), "backbone."),
        (re.compile(r"^encoder\.dino\."), "vit."),
        (re.compile(r"^encoder\.dino\."), "encoder.backbone."),
        (re.compile(r"^encoder\.dino\."), "encoder.vit."),
        (re.compile(r"^encoder\.dino\."), "dino."),
        (re.compile(r"^encoder\.dino\."), "embedding_decoder."),
        (re.compile(r"^encoder\.dino\."), "decoder.embedding_decoder."),
        
        # Try to find matcher keys
        (re.compile(r"^matcher\."), "decoder."),
        (re.compile(r"^matcher\."), "matcher."),
        (re.compile(r"^matcher\."), "matcher.model.decoder."),
    ]


def fix_dinov2_block_structure(weights_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Fix DINOv2 transformer block naming structure for BlockChunk architecture.
    MatchAnything uses a specific block structure that needs adjustment.
    """
    fixed = {}
    block_mappings = {}
    
    for key, value in weights_dict.items():
        if key.startswith("encoder.dino.blocks."):
            parts = key.split(".")
            if len(parts) >= 4 and parts[3].isdigit():
                block_num = int(parts[3])
                
                # Check if this follows the BlockChunk pattern (needs 0.X.Y structure)
                if len(parts) >= 5 and parts[4].isdigit():
                    # Already in BlockChunk format
                    fixed[key] = value
                else:
                    # Convert to BlockChunk format: blocks.N.X -> blocks.0.N.X
                    new_key = ".".join(parts[:3] + ["0", str(block_num)] + parts[4:])
                    fixed[new_key] = value
                    block_mappings[key] = new_key
                    print(f"[IMPROVED] Fixed BlockChunk structure: {key} -> {new_key}")
            else:
                fixed[key] = value
        else:
            fixed[key] = value
    
    return fixed


def resize_positional_embedding(pos_embed: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize positional embedding to match target size."""
    if pos_embed.shape[1] == target_size:
        return pos_embed
    
    print(f"[IMPROVED] Resizing positional embedding: {pos_embed.shape} -> [1, {target_size}, {pos_embed.shape[2]}]")
    
    # Separate CLS token and spatial tokens
    cls_token = pos_embed[:, 0:1, :]
    spatial_tokens = pos_embed[:, 1:, :]
    
    # Reshape spatial tokens to 2D grid
    num_spatial = spatial_tokens.shape[1]
    original_size = int(np.sqrt(num_spatial))
    
    target_spatial = target_size - 1  # -1 for CLS token
    target_grid = int(np.sqrt(target_spatial))
    
    # Reshape and interpolate
    spatial_2d = spatial_tokens.reshape(1, original_size, original_size, -1).permute(0, 3, 1, 2)
    resized = F.interpolate(spatial_2d, size=(target_grid, target_grid), mode="bilinear", align_corners=False)
    resized = resized.permute(0, 2, 3, 1).reshape(1, target_spatial, -1)
    
    return torch.cat([cls_token, resized], dim=1)


def load_dinov2_components(model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Load official DINOv2 weights for missing components."""
    dinov2_weights = {}
    
    try:
        print("[IMPROVED] Loading official DINOv2 weights for missing components...")
        dinov2_model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
        official = dinov2_model.state_dict()
        
        # Map official DINOv2 weights to model structure
        for model_key in model_state_dict.keys():
            if model_key.startswith("encoder.dino."):
                # Remove the encoder.dino prefix to get the original DINOv2 key
                dino_key = model_key[len("encoder.dino."):]
                
                if dino_key in official:
                    value = official[dino_key]
                    model_param = model_state_dict[model_key]
                    
                    # Handle special cases
                    if dino_key == "pos_embed" and value.shape != model_param.shape:
                        value = resize_positional_embedding(value, model_param.shape[1])
                    
                    if value.shape == model_param.shape:
                        dinov2_weights[model_key] = value
                        print(f"[IMPROVED] Added DINOv2 weight: {model_key} -> {value.shape}")
    
    except Exception as e:
        print(f"[IMPROVED] Warning: Could not load official DINOv2 weights: {e}")
    
    return dinov2_weights


def find_best_key_matches(
    model_keys: Set[str], 
    checkpoint_keys: Set[str], 
    checkpoint_dict: Dict[str, torch.Tensor],
    model_dict: Dict[str, torch.Tensor]
) -> Dict[str, str]:
    """Find the best matches between model keys and checkpoint keys."""
    matches = {}
    
    # Try exact suffix matching for unmatched keys
    for model_key in model_keys:
        if model_key in matches:
            continue
            
        model_param = model_dict[model_key]
        
        # Try different suffix lengths
        for suffix_len in [1, 2, 3, 4]:
            model_suffix = ".".join(model_key.split(".")[-suffix_len:])
            
            for checkpoint_key in checkpoint_keys:
                if checkpoint_key.endswith(model_suffix):
                    checkpoint_param = checkpoint_dict[checkpoint_key]
                    
                    # Check shape compatibility
                    if checkpoint_param.shape == model_param.shape:
                        matches[model_key] = checkpoint_key
                        print(f"[IMPROVED] Suffix match: {model_key} <- {checkpoint_key}")
                        break
            
            if model_key in matches:
                break
    
    return matches


def apply_improved_weight_loading(
    checkpoint_path: str,
    model_state_dict: Dict[str, torch.Tensor],
    load_dinov2_components: bool = True,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """Apply improved weight loading with better key mapping."""
    
    print(f"[IMPROVED] Loading checkpoint: {checkpoint_path}")
    
    try:
        raw = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_dict = raw.get("state_dict", raw)
    except Exception as e:
        print(f"[IMPROVED] Error loading checkpoint: {e}")
        return {}
    
    print(f"[IMPROVED] Checkpoint has {len(checkpoint_dict)} keys")
    
    # Step 1: Apply forward mapping rules
    mapping_rules = create_comprehensive_mapping_rules()
    forward_mapped = {}
    
    for checkpoint_key, checkpoint_value in checkpoint_dict.items():
        mapped_key = checkpoint_key
        
        # Apply all mapping rules
        for pattern, replacement in mapping_rules:
            mapped_key = pattern.sub(replacement, mapped_key)
        
        forward_mapped[mapped_key] = checkpoint_value
    
    print(f"[IMPROVED] After forward mapping: {len(forward_mapped)} keys")
    
    # Step 2: Fix DINOv2 block structure
    forward_mapped = fix_dinov2_block_structure(forward_mapped)
    
    # Step 3: Direct matching
    loadable = {}
    model_keys = set(model_state_dict.keys())
    checkpoint_keys = set(forward_mapped.keys())
    
    for model_key in model_keys:
        if model_key in forward_mapped:
            checkpoint_value = forward_mapped[model_key]
            model_param = model_state_dict[model_key]
            
            if checkpoint_value.shape == model_param.shape:
                loadable[model_key] = checkpoint_value
            else:
                if verbose:
                    print(f"[IMPROVED] Shape mismatch for {model_key}: {checkpoint_value.shape} vs {model_param.shape}")
    
    print(f"[IMPROVED] After direct matching: {len(loadable)}/{len(model_keys)} ({100.0*len(loadable)/len(model_keys):.1f}%)")
    
    # Step 4: Try reverse mapping for unmatched keys
    unmatched_model_keys = model_keys - set(loadable.keys())
    reverse_rules = create_reverse_mapping_rules()
    
    for model_key in unmatched_model_keys:
        model_param = model_state_dict[model_key]
        
        for pattern, replacement in reverse_rules:
            if pattern.match(model_key):
                # Try to find this key with the reverse mapping
                candidate_key = pattern.sub(replacement, model_key)
                
                # Look for this candidate in the original checkpoint
                for orig_key, orig_value in checkpoint_dict.items():
                    if orig_key == candidate_key or orig_key.endswith(candidate_key):
                        if orig_value.shape == model_param.shape:
                            loadable[model_key] = orig_value
                            print(f"[IMPROVED] Reverse match: {model_key} <- {orig_key}")
                            break
                
                if model_key in loadable:
                    break
    
    print(f"[IMPROVED] After reverse matching: {len(loadable)}/{len(model_keys)} ({100.0*len(loadable)/len(model_keys):.1f}%)")
    
    # Step 5: Try suffix matching for remaining keys
    still_unmatched = model_keys - set(loadable.keys())
    suffix_matches = find_best_key_matches(still_unmatched, checkpoint_keys, forward_mapped, model_state_dict)
    
    for model_key, checkpoint_key in suffix_matches.items():
        loadable[model_key] = forward_mapped[checkpoint_key]
    
    print(f"[IMPROVED] After suffix matching: {len(loadable)}/{len(model_keys)} ({100.0*len(loadable)/len(model_keys):.1f}%)")
    
    # Step 6: Load DINOv2 components for still missing keys
    if load_dinov2_components:
        dinov2_weights = load_dinov2_components(model_state_dict)
        
        # Only add DINOv2 weights for keys that are still missing
        for key, value in dinov2_weights.items():
            if key not in loadable:
                loadable[key] = value
    
    # Final statistics
    final_loaded = len(loadable)
    total_params = len(model_state_dict)
    loading_percentage = 100.0 * final_loaded / total_params
    
    print(f"\n[IMPROVED] === FINAL LOADING SUMMARY ===")
    print(f"Total weights loaded: {final_loaded} / {total_params} ({loading_percentage:.1f}%)")
    
    # Analyze what's still missing
    still_missing = model_keys - set(loadable.keys())
    if still_missing and verbose:
        print(f"[IMPROVED] Still missing {len(still_missing)} keys:")
        for key in sorted(list(still_missing)[:10]):  # Show first 10
            print(f"  - {key}")
        if len(still_missing) > 10:
            print(f"  ... and {len(still_missing) - 10} more")
    
    # Analyze loaded components
    dino_loaded = sum(1 for k in loadable if k.startswith("encoder.dino."))
    dino_total = sum(1 for k in model_state_dict if k.startswith("encoder.dino."))
    matcher_loaded = sum(1 for k in loadable if k.startswith("matcher."))
    matcher_total = sum(1 for k in model_state_dict if k.startswith("matcher."))
    
    print(f"DINOv2 encoder: {dino_loaded} / {dino_total} ({100.0*dino_loaded/max(dino_total,1):.1f}%)")
    print(f"Matcher: {matcher_loaded} / {matcher_total} ({100.0*matcher_loaded/max(matcher_total,1):.1f}%)")
    
    return loadable


if __name__ == "__main__":
    print("Improved Weight Loader for MatchAnything")