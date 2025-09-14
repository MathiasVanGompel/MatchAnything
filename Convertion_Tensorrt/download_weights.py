#!/usr/bin/env python3
"""
Download and verify MatchAnything pretrained weights from Google Drive.
This script handles the weights from https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from typing import Optional
import gdown
import torch


def download_matchanything_weights(
    output_dir: str = "/workspace/imcui/third_party/MatchAnything/weights",
    force_download: bool = False
) -> str:
    """
    Download MatchAnything pretrained weights from Google Drive.
    
    Args:
        output_dir: Directory to save the weights
        force_download: Whether to re-download if file exists
        
    Returns:
        Path to the downloaded weights file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Google Drive file ID from the URL
    file_id = "12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d"
    output_path = os.path.join(output_dir, "matchanything_roma.ckpt")
    
    # Check if file already exists
    if os.path.exists(output_path) and not force_download:
        print(f"[WEIGHTS] File already exists: {output_path}")
        # Verify it's a valid checkpoint
        try:
            checkpoint = torch.load(output_path, map_location="cpu")
            print(f"[WEIGHTS] Checkpoint verified, contains {len(checkpoint.get('state_dict', checkpoint))} keys")
            return output_path
        except Exception as e:
            print(f"[WEIGHTS] Existing file corrupted: {e}, re-downloading...")
    
    print(f"[WEIGHTS] Downloading MatchAnything weights from Google Drive...")
    print(f"[WEIGHTS] File ID: {file_id}")
    print(f"[WEIGHTS] Output: {output_path}")
    
    try:
        # Use gdown to download from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        
        # Verify the download
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"[WEIGHTS] Downloaded successfully: {file_size:.1f} MB")
            
            # Try to load and verify it's a valid checkpoint
            try:
                checkpoint = torch.load(output_path, map_location="cpu")
                state_dict = checkpoint.get('state_dict', checkpoint)
                print(f"[WEIGHTS] Checkpoint verified, contains {len(state_dict)} keys")
                
                # Print some key information
                sample_keys = list(state_dict.keys())[:10]
                print(f"[WEIGHTS] Sample keys: {sample_keys}")
                
                return output_path
                
            except Exception as e:
                print(f"[WEIGHTS] Error loading checkpoint: {e}")
                print(f"[WEIGHTS] File may be corrupted, but will proceed...")
                return output_path
        else:
            raise FileNotFoundError("Download failed - file not found")
            
    except Exception as e:
        print(f"[WEIGHTS] Download failed: {e}")
        print(f"[WEIGHTS] Please manually download from:")
        print(f"[WEIGHTS] https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view")
        print(f"[WEIGHTS] And save as: {output_path}")
        raise

def verify_weights_compatibility(weights_path: str) -> dict:
    """
    Verify that the weights are compatible with the model architecture.
    
    Args:
        weights_path: Path to the checkpoint file
        
    Returns:
        Dictionary with verification results
    """
    print(f"[VERIFY] Checking weights compatibility: {weights_path}")
    
    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Analyze the structure
        all_keys = list(state_dict.keys())
        
        # Look for key patterns
        patterns = {
            'dino': [k for k in all_keys if 'dino' in k.lower()],
            'encoder': [k for k in all_keys if 'encoder' in k.lower()],
            'matcher': [k for k in all_keys if 'matcher' in k.lower()],
            'backbone': [k for k in all_keys if 'backbone' in k.lower()],
            'model': [k for k in all_keys if k.startswith('model.')],
        }
        
        result = {
            'total_keys': len(all_keys),
            'patterns': {k: len(v) for k, v in patterns.items()},
            'sample_keys': all_keys[:20],
            'has_dino': len(patterns['dino']) > 0,
            'has_encoder': len(patterns['encoder']) > 0,
            'has_matcher': len(patterns['matcher']) > 0,
        }
        
        print(f"[VERIFY] Total parameters: {result['total_keys']}")
        print(f"[VERIFY] Key patterns found: {result['patterns']}")
        print(f"[VERIFY] Has DINOv2 weights: {result['has_dino']}")
        print(f"[VERIFY] Has encoder weights: {result['has_encoder']}")
        print(f"[VERIFY] Has matcher weights: {result['has_matcher']}")
        
        return result
        
    except Exception as e:
        print(f"[VERIFY] Error verifying weights: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MatchAnything pretrained weights")
    parser.add_argument(
        "--output_dir", 
        default="/workspace/imcui/third_party/MatchAnything/weights",
        help="Output directory for weights"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify weights compatibility after download"
    )
    
    args = parser.parse_args()
    
    try:
        # Download weights
        weights_path = download_matchanything_weights(
            output_dir=args.output_dir,
            force_download=args.force
        )
        
        # Verify if requested
        if args.verify:
            verify_weights_compatibility(weights_path)
            
        print(f"[SUCCESS] Weights ready at: {weights_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to download weights: {e}")
        sys.exit(1)
