#!/usr/bin/env python3
"""
Model architecture inspection script.
Run this in your environment with: python3 inspect_model.py
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add the path to import the model
sys.path.insert(0, str(Path(__file__).parent / "Convertion_Tensorrt_new"))
sys.path.insert(0, str(Path(__file__).parent / "Convertion_Tensorrt"))

def inspect_model():
    """Inspect the model architecture."""
    try:
        from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT
        
        print("Creating model...")
        model = AccurateMatchAnythingTRT(amp=False)
        
        # Get state dict
        state_dict = model.state_dict()
        print(f"\nModel has {len(state_dict)} parameters")
        
        # Categorize keys by prefixes
        categories = defaultdict(list)
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) >= 2:
                prefix = parts[0] + '.' + parts[1]
            else:
                prefix = parts[0]
            categories[prefix].append(key)
        
        print(f"\nModel parameter categories:")
        for prefix, keys in sorted(categories.items()):
            print(f"  {prefix}: {len(keys)} parameters")
        
        print(f"\nFirst 30 model parameters:")
        for i, (key, param) in enumerate(list(state_dict.items())[:30]):
            shape = tuple(param.shape) if hasattr(param, 'shape') else 'scalar'
            print(f"  {i+1:2d}: {key} -> {shape}")
        
        # Look for specific patterns
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k.lower()]
        dino_keys = [k for k in state_dict.keys() if 'dino' in k.lower()]
        blocks_keys = [k for k in state_dict.keys() if 'blocks' in k.lower()]
        matcher_keys = [k for k in state_dict.keys() if 'matcher' in k.lower()]
        
        print(f"\nPattern analysis:")
        print(f"  Parameters with 'encoder': {len(encoder_keys)}")
        print(f"  Parameters with 'dino': {len(dino_keys)}")
        print(f"  Parameters with 'blocks': {len(blocks_keys)}")
        print(f"  Parameters with 'matcher': {len(matcher_keys)}")
        
        if encoder_keys:
            print(f"\nSample encoder parameters:")
            for i, key in enumerate(encoder_keys[:15]):
                shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
                print(f"  {i+1}: {key} -> {shape}")
        
        if matcher_keys:
            print(f"\nSample matcher parameters:")
            for i, key in enumerate(matcher_keys[:10]):
                shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
                print(f"  {i+1}: {key} -> {shape}")
        
        # Check for common prefixes
        common_prefixes = set()
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) >= 1:
                common_prefixes.add(parts[0])
            if len(parts) >= 2:
                common_prefixes.add(parts[0] + '.' + parts[1])
        
        print(f"\nModel prefixes:")
        for prefix in sorted(common_prefixes):
            count = len([k for k in state_dict.keys() if k.startswith(prefix + '.') or k == prefix])
            if count > 0:
                print(f"  {prefix}: {count} parameters")
        
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model()