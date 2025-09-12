#!/usr/bin/env python3
"""
Improved weight adapter specifically for MatchAnything checkpoints.
"""
import re
import torch
from typing import Dict, List, Tuple
from collections import Counter

def create_matchanything_mapping_rules():
    """Create comprehensive mapping rules for MatchAnything checkpoint structure."""
    return [
        # Remove common wrappers
        (re.compile(r"^module\."), ""),
        (re.compile(r"^_orig_mod\."), ""),
        
        # MatchAnything specific structure
        (re.compile(r"^matcher\.model\.encoder\.cnn\."), "encoder.layers."),
        (re.compile(r"^matcher\.model\.encoder\."), "encoder."),
        (re.compile(r"^matcher\.model\.decoder\.embedding_decoder\."), "encoder.dino."),
        (re.compile(r"^matcher\.model\.decoder\."), "matcher."),
        (re.compile(r"^matcher\.model\."), ""),
        (re.compile(r"^matcher\."), ""),
        (re.compile(r"^model\."), ""),
        
        # DINOv2 patterns - the checkpoint already has DINOv2 weights!
        (re.compile(r"^embedding_decoder\."), "encoder.dino."),
        (re.compile(r"^decoder\.embedding_decoder\."), "encoder.dino."),
        
        # Generic patterns
        (re.compile(r"^encoder\."), "encoder."),  # identity
    ]

def apply_improved_mapping(checkpoint_path: str, model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply improved mapping for MatchAnything weights."""
    
    print(f"[IMPROVED] Loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt_state_dict = raw.get("state_dict", raw)
    
    print(f"[IMPROVED] Checkpoint has {len(ckpt_state_dict)} keys")
    print(f"[IMPROVED] Model expects {len(model_state_dict)} keys")
    
    # Analyze checkpoint structure
    prefixes = {}
    for key in ckpt_state_dict.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = f"{parts[0]}.{parts[1]}"
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print("[IMPROVED] Checkpoint key prefixes:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {prefix}: {count} keys")
    
    # Apply mapping rules
    rules = create_matchanything_mapping_rules()
    mapped_weights = {}
    
    for ckpt_key, ckpt_value in ckpt_state_dict.items():
        mapped_key = ckpt_key
        
        # Apply each rule in sequence
        for pattern, replacement in rules:
            mapped_key = pattern.sub(replacement, mapped_key)
        
        mapped_weights[mapped_key] = ckpt_value
    
    print(f"[IMPROVED] After mapping: {len(mapped_weights)} keys")
    
    # Try direct matches first
    loadable = {}
    for model_key in model_state_dict.keys():
        if model_key in mapped_weights:
            ckpt_tensor = mapped_weights[model_key]
            model_tensor = model_state_dict[model_key]
            
            if ckpt_tensor.shape == model_tensor.shape:
                loadable[model_key] = ckpt_tensor
            else:
                print(f"[IMPROVED] Shape mismatch: {model_key}")
                print(f"  Checkpoint: {ckpt_tensor.shape}")
                print(f"  Model:      {model_tensor.shape}")
    
    # Try suffix matching for remaining keys
    remaining_model_keys = set(model_state_dict.keys()) - set(loadable.keys())
    remaining_ckpt_keys = set(mapped_weights.keys()) - set(loadable.keys())
    
    print(f"[IMPROVED] Trying suffix matching for {len(remaining_model_keys)} remaining keys...")
    
    for model_key in remaining_model_keys:
        model_suffix = model_key.split('.')[-2:] if '.' in model_key else [model_key]
        model_suffix_str = '.'.join(model_suffix)
        
        for ckpt_key in remaining_ckpt_keys:
            if ckpt_key.endswith(model_suffix_str):
                ckpt_tensor = mapped_weights[ckpt_key]
                model_tensor = model_state_dict[model_key]
                
                if ckpt_tensor.shape == model_tensor.shape:
                    loadable[model_key] = ckpt_tensor
                    print(f"[IMPROVED] Suffix match: {ckpt_key} -> {model_key}")
                    break
    
    print(f"[IMPROVED] Final result: {len(loadable)} / {len(model_state_dict)} weights loaded")
    
    return loadable

# Test the improved mapping
if __name__ == "__main__":
    # This would be called from the main conversion script
    pass
