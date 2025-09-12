#!/usr/bin/env python3
"""
Comprehensive diagnostic script for MatchAnything weight loading issues.
This script analyzes both the checkpoint and model structure to identify mapping problems.
"""

import sys
import torch
from pathlib import Path
from collections import defaultdict, Counter
import re

# Add paths for imports
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent / "Convertion_Tensorrt"
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT


def analyze_key_patterns(keys, name):
    """Analyze patterns in a set of keys."""
    print(f"\n=== {name.upper()} ANALYSIS ===")
    print(f"Total keys: {len(keys)}")
    
    # Analyze prefixes
    prefixes = defaultdict(int)
    for key in keys:
        parts = key.split('.')
        if len(parts) >= 1:
            prefixes[parts[0]] += 1
        if len(parts) >= 2:
            prefixes[parts[0] + '.' + parts[1]] += 1
        if len(parts) >= 3:
            prefixes[parts[0] + '.' + parts[1] + '.' + parts[2]] += 1
    
    print("\nTop prefixes:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
        print(f"  {prefix}: {count}")
    
    # Analyze patterns
    patterns = {
        'encoder': len([k for k in keys if 'encoder' in k.lower()]),
        'dino': len([k for k in keys if 'dino' in k.lower()]),
        'blocks': len([k for k in keys if 'blocks' in k.lower()]),
        'matcher': len([k for k in keys if 'matcher' in k.lower()]),
        'decoder': len([k for k in keys if 'decoder' in k.lower()]),
        'backbone': len([k for k in keys if 'backbone' in k.lower()]),
        'vit': len([k for k in keys if 'vit' in k.lower()]),
        'patch_embed': len([k for k in keys if 'patch_embed' in k.lower()]),
        'pos_embed': len([k for k in keys if 'pos_embed' in k.lower()]),
        'cls_token': len([k for k in keys if 'cls_token' in k.lower()]),
        'norm': len([k for k in keys if 'norm' in k.lower()]),
        'attn': len([k for k in keys if 'attn' in k.lower()]),
        'mlp': len([k for k in keys if 'mlp' in k.lower()]),
    }
    
    print("\nPattern counts:")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {pattern}: {count}")


def find_potential_mappings(checkpoint_keys, model_keys, checkpoint_dict, model_dict):
    """Find potential mappings between checkpoint and model keys."""
    print(f"\n=== POTENTIAL MAPPINGS ===")
    
    # Group keys by suffix patterns
    def get_suffix_patterns(keys, suffix_len=2):
        patterns = defaultdict(list)
        for key in keys:
            parts = key.split('.')
            if len(parts) >= suffix_len:
                suffix = '.'.join(parts[-suffix_len:])
                patterns[suffix].append(key)
        return patterns
    
    checkpoint_patterns = get_suffix_patterns(checkpoint_keys)
    model_patterns = get_suffix_patterns(model_keys)
    
    # Find matching patterns
    common_suffixes = set(checkpoint_patterns.keys()) & set(model_patterns.keys())
    
    print(f"Found {len(common_suffixes)} common suffix patterns")
    
    potential_mappings = {}
    for suffix in sorted(common_suffixes):
        checkpoint_candidates = checkpoint_patterns[suffix]
        model_candidates = model_patterns[suffix]
        
        print(f"\nSuffix: {suffix}")
        print(f"  Checkpoint candidates: {len(checkpoint_candidates)}")
        print(f"  Model candidates: {len(model_candidates)}")
        
        # Try to match by shape
        for model_key in model_candidates:
            model_shape = model_dict[model_key].shape
            
            for checkpoint_key in checkpoint_candidates:
                checkpoint_shape = checkpoint_dict[checkpoint_key].shape
                
                if checkpoint_shape == model_shape:
                    potential_mappings[model_key] = checkpoint_key
                    print(f"    MATCH: {model_key} <- {checkpoint_key} {model_shape}")
                    break
    
    return potential_mappings


def analyze_transformer_blocks(checkpoint_keys, model_keys):
    """Analyze transformer block structure."""
    print(f"\n=== TRANSFORMER BLOCKS ANALYSIS ===")
    
    # Find block patterns in checkpoint
    checkpoint_blocks = [k for k in checkpoint_keys if 'blocks.' in k and ('norm' in k or 'attn' in k or 'mlp' in k)]
    model_blocks = [k for k in model_keys if 'blocks.' in k and ('norm' in k or 'attn' in k or 'mlp' in k)]
    
    print(f"Checkpoint block keys: {len(checkpoint_blocks)}")
    print(f"Model block keys: {len(model_blocks)}")
    
    # Analyze block numbering patterns
    def extract_block_numbers(keys):
        patterns = []
        for key in keys:
            match = re.search(r'blocks\.(\d+)(?:\.(\d+))?\.', key)
            if match:
                if match.group(2):  # BlockChunk pattern (blocks.X.Y)
                    patterns.append(f"blocks.{match.group(1)}.{match.group(2)}")
                else:  # Simple pattern (blocks.X)
                    patterns.append(f"blocks.{match.group(1)}")
        return sorted(set(patterns))
    
    checkpoint_block_patterns = extract_block_numbers(checkpoint_blocks)
    model_block_patterns = extract_block_numbers(model_blocks)
    
    print(f"\nCheckpoint block patterns: {checkpoint_block_patterns[:10]}...")
    print(f"Model block patterns: {model_block_patterns[:10]}...")
    
    # Check if we need BlockChunk conversion
    has_blockchunk_checkpoint = any('.' in p.split('blocks.')[1] for p in checkpoint_block_patterns if 'blocks.' in p)
    has_blockchunk_model = any('.' in p.split('blocks.')[1] for p in model_block_patterns if 'blocks.' in p)
    
    print(f"\nCheckpoint uses BlockChunk pattern: {has_blockchunk_checkpoint}")
    print(f"Model uses BlockChunk pattern: {has_blockchunk_model}")


def main():
    """Main diagnostic function."""
    checkpoint_path = "/home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt"
    
    print("=== MATCHANYTHING WEIGHT LOADING DIAGNOSTIC ===")
    
    # Load checkpoint
    try:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        raw_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_dict = raw_checkpoint.get('state_dict', raw_checkpoint)
        checkpoint_keys = set(checkpoint_dict.keys())
        print(f"Checkpoint loaded successfully with {len(checkpoint_keys)} keys")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint: {e}")
        return
    
    # Create model
    try:
        print(f"\nCreating model...")
        model = AccurateMatchAnythingTRT(amp=False)
        model_dict = model.state_dict()
        model_keys = set(model_dict.keys())
        print(f"Model created successfully with {len(model_keys)} parameters")
    except Exception as e:
        print(f"ERROR: Could not create model: {e}")
        return
    
    # Analyze both structures
    analyze_key_patterns(checkpoint_keys, "checkpoint")
    analyze_key_patterns(model_keys, "model")
    
    # Find exact matches
    exact_matches = checkpoint_keys & model_keys
    print(f"\n=== EXACT MATCHES ===")
    print(f"Found {len(exact_matches)} exact matches ({100.0*len(exact_matches)/len(model_keys):.1f}% of model)")
    
    # Analyze transformer blocks
    analyze_transformer_blocks(checkpoint_keys, model_keys)
    
    # Find potential mappings
    potential_mappings = find_potential_mappings(checkpoint_keys, model_keys, checkpoint_dict, model_dict)
    
    print(f"\n=== SUMMARY ===")
    print(f"Exact matches: {len(exact_matches)}")
    print(f"Potential suffix mappings: {len(potential_mappings)}")
    print(f"Total mappable: {len(exact_matches) + len(potential_mappings)}")
    print(f"Model parameters: {len(model_keys)}")
    print(f"Potential loading percentage: {100.0*(len(exact_matches) + len(potential_mappings))/len(model_keys):.1f}%")
    
    # Show some sample mappings
    if potential_mappings:
        print(f"\nSample potential mappings:")
        for i, (model_key, checkpoint_key) in enumerate(list(potential_mappings.items())[:10]):
            print(f"  {model_key} <- {checkpoint_key}")
    
    # Identify missing categories
    unmatched_model = model_keys - exact_matches - set(potential_mappings.keys())
    if unmatched_model:
        print(f"\nUnmatched model keys by category:")
        unmatched_patterns = defaultdict(list)
        for key in unmatched_model:
            parts = key.split('.')
            category = parts[0] if len(parts) >= 1 else 'root'
            if len(parts) >= 2:
                category = parts[0] + '.' + parts[1]
            unmatched_patterns[category].append(key)
        
        for category, keys in sorted(unmatched_patterns.items(), key=lambda x: -len(x[1])):
            print(f"  {category}: {len(keys)} keys")
            if len(keys) <= 3:
                for key in keys:
                    print(f"    - {key}")


if __name__ == "__main__":
    main()