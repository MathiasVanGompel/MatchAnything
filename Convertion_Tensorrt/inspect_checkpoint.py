#!/usr/bin/env python3
"""
Inspect the checkpoint to understand its structure and create a compatible model.
"""
import torch
from collections import defaultdict

def inspect_checkpoint(ckpt_path):
    print("="*60)
    print("CHECKPOINT INSPECTION")
    print("="*60)
    
    # Load checkpoint
    print(f"Loading: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw)
    
    print(f"Total keys: {len(state)}")
    print(f"Checkpoint structure: {list(raw.keys())}")
    
    # Analyze key patterns
    prefixes = defaultdict(int)
    for key in state.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = '.'.join(parts[:2])
            prefixes[prefix] += 1
        if len(parts) >= 3:
            prefix = '.'.join(parts[:3])
            prefixes[prefix] += 1
    
    print("\nKey prefixes (showing structure):")
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix:<40} x{count}")
    
    # Show sample keys for each major section
    print("\nSample keys by section:")
    sections = defaultdict(list)
    for key in state.keys():
        section = key.split('.')[0] if '.' in key else 'root'
        sections[section].append(key)
    
    for section, keys in sections.items():
        print(f"\n[{section}] - {len(keys)} keys:")
        for key in sorted(keys)[:10]:  # Show first 10
            tensor = state[key]
            print(f"  {key:<50} {list(tensor.shape)}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")
    
    # Try to identify model type
    print("\n" + "="*60)
    print("MODEL TYPE ANALYSIS")
    print("="*60)
    
    has_cnn = any('cnn' in key for key in state.keys())
    has_dino = any('dino' in key for key in state.keys())
    has_transformer = any('transformer' in key or 'attn' in key for key in state.keys())
    has_patch_embed = any('patch_embed' in key for key in state.keys())
    
    print(f"Has CNN layers: {has_cnn}")
    print(f"Has DINO layers: {has_dino}")  
    print(f"Has Transformer: {has_transformer}")
    print(f"Has Patch Embed: {has_patch_embed}")
    
    if has_cnn and not has_dino:
        print("\n🔍 CONCLUSION: This appears to be a CNN-based model (LoFTR-style)")
        print("   Our DINOv2 TensorRT implementation won't be compatible.")
        print("   Need to create a CNN-based TensorRT implementation instead.")
    elif has_dino:
        print("\n🔍 CONCLUSION: This appears to be a DINOv2-based model")
        print("   Our TensorRT implementation should be compatible.")
    else:
        print("\n🔍 CONCLUSION: Model type unclear from checkpoint keys")

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    inspect_checkpoint(ckpt_path)

if __name__ == "__main__":
    main()