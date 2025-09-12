#!/usr/bin/env python3
"""
Simple checkpoint inspection script.
Run this in your environment with: python3 inspect_checkpoint.py
"""

import torch
import sys
from collections import defaultdict

def inspect_checkpoint(checkpoint_path):
    """Inspect the structure of a checkpoint file."""
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        
        print(f"\nCheckpoint has {len(state_dict)} keys")
        
        # Categorize keys by prefixes
        categories = defaultdict(list)
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) >= 2:
                prefix = parts[0] + '.' + parts[1]
            else:
                prefix = parts[0]
            categories[prefix].append(key)
        
        print(f"\nKey categories:")
        for prefix, keys in sorted(categories.items()):
            print(f"  {prefix}: {len(keys)} keys")
        
        print(f"\nFirst 30 keys (sorted):")
        for i, key in enumerate(sorted(state_dict.keys())[:30]):
            shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
            print(f"  {i+1:2d}: {key} -> {shape}")
        
        # Look for specific patterns
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k.lower()]
        dino_keys = [k for k in state_dict.keys() if 'dino' in k.lower()]
        blocks_keys = [k for k in state_dict.keys() if 'blocks' in k.lower()]
        matcher_keys = [k for k in state_dict.keys() if 'matcher' in k.lower()]
        
        print(f"\nPattern analysis:")
        print(f"  Keys with 'encoder': {len(encoder_keys)}")
        print(f"  Keys with 'dino': {len(dino_keys)}")
        print(f"  Keys with 'blocks': {len(blocks_keys)}")
        print(f"  Keys with 'matcher': {len(matcher_keys)}")
        
        if matcher_keys:
            print(f"\nSample matcher keys:")
            for i, key in enumerate(matcher_keys[:10]):
                shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
                print(f"  {i+1}: {key} -> {shape}")
                
        if encoder_keys:
            print(f"\nSample encoder keys:")
            for i, key in enumerate(encoder_keys[:10]):
                shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
                print(f"  {i+1}: {key} -> {shape}")
        
        # Check for common prefixes that might need remapping
        common_prefixes = set()
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) >= 1:
                common_prefixes.add(parts[0])
            if len(parts) >= 2:
                common_prefixes.add(parts[0] + '.' + parts[1])
        
        print(f"\nCommon prefixes:")
        for prefix in sorted(common_prefixes):
            count = len([k for k in state_dict.keys() if k.startswith(prefix + '.')])
            if count > 0:
                print(f"  {prefix}: {count} keys")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    checkpoint_path = "/home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    inspect_checkpoint(checkpoint_path)