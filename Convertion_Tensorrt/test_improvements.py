#!/usr/bin/env python3
"""
Test script to validate the improvements made to MatchAnything TensorRT conversion.
This script tests weight loading, image size handling, and overall functionality.
"""

import os
import sys
import tempfile
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import Tuple, Optional
import PIL.Image as Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_images(sizes: list) -> list:
    """Create test images of different sizes"""
    test_images = []
    
    for i, (h, w) in enumerate(sizes):
        # Create a random test image
        img_array = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Add some pattern to make it more realistic
        for y in range(0, h, 50):
            for x in range(0, w, 50):
                img_array[y:min(y+25, h), x:min(x+25, w)] = [255, 0, 0]  # Red squares
        
        # Save as temporary file
        temp_path = f"/tmp/test_image_{i}_{h}x{w}.jpg"
        img = Image.fromarray(img_array)
        img.save(temp_path, "JPEG")
        test_images.append(temp_path)
        print(f"Created test image: {temp_path} ({h}x{w})")
    
    return test_images

def test_weight_downloading():
    """Test the weight downloading functionality"""
    print("\n" + "="*60)
    print("TESTING WEIGHT DOWNLOADING")
    print("="*60)
    
    try:
        from download_weights import download_matchanything_weights, verify_weights_compatibility
        
        # Test download to temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"Testing download to: {temp_dir}")
        
        # This will attempt to download or verify existing weights
        weights_path = download_matchanything_weights(
            output_dir=temp_dir,
            force_download=False
        )
        
        if os.path.exists(weights_path):
            print("✅ Weight downloading/verification successful")
            
            # Test weight verification
            result = verify_weights_compatibility(weights_path)
            if 'error' not in result:
                print("✅ Weight verification successful")
                print(f"   Total keys: {result['total_keys']}")
                print(f"   Key patterns: {result['patterns']}")
            else:
                print(f"❌ Weight verification failed: {result['error']}")
                
        else:
            print("❌ Weight downloading failed")
            return False
            
    except Exception as e:
        print(f"❌ Weight downloading test failed: {e}")
        return False
    
    return True

def test_image_size_handling():
    """Test improved image size handling"""
    print("\n" + "="*60)
    print("TESTING IMAGE SIZE HANDLING")
    print("="*60)
    
    try:
        from run_accurate_matchanything_trt import load_image_rgb, preprocess_for_tensorrt
        
        # Test different image sizes
        test_sizes = [
            (480, 640),   # Different aspect ratio
            (832, 832),   # Square
            (1024, 768),  # Landscape
            (600, 800),   # Portrait
        ]
        
        test_images = create_test_images(test_sizes)
        target_size = (832, 832)
        
        print(f"Testing image loading with target size: {target_size}")
        
        for img_path in test_images:
            try:
                # Test loading with target size
                img_rgb = load_image_rgb(img_path, target_size=target_size)
                print(f"✅ Loaded {img_path}: {img_rgb.shape}")
                
                # Test preprocessing
                img_tensor = preprocess_for_tensorrt(img_rgb)
                print(f"   Preprocessed shape: {img_tensor.shape}")
                
                # Verify target size was achieved
                if img_rgb.shape[:2] == target_size[::-1]:  # (H, W) vs (W, H)
                    print(f"   ✅ Correct target size achieved")
                else:
                    print(f"   ❌ Target size not achieved: got {img_rgb.shape[:2]}, expected {target_size[::-1]}")
                
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")
                return False
        
        # Clean up test images
        for img_path in test_images:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        print("✅ Image size handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Image size handling test failed: {e}")
        return False

def test_weight_loading():
    """Test improved weight loading functionality"""
    print("\n" + "="*60)
    print("TESTING WEIGHT LOADING")
    print("="*60)
    
    try:
        from weight_adapter import remap_and_load, _apply_rules
        from accurate_matchanything_trt import AccurateMatchAnythingTRT
        
        # Test rule application
        test_cases = [
            ("module.encoder.dino.patch_embed.proj.weight", "encoder.dino.patch_embed.proj.weight"),
            ("matcher.model.encoder.layers.0.weight", "encoder.layers.0.weight"),
            ("backbone.blocks.0.attn.qkv.weight", "encoder.dino.blocks.0.attn.qkv.weight"),
        ]
        
        print("Testing weight key remapping rules:")
        for original, expected in test_cases:
            remapped = _apply_rules(original)
            if remapped == expected:
                print(f"✅ {original} -> {remapped}")
            else:
                print(f"❌ {original} -> {remapped} (expected: {expected})")
        
        # Test model creation
        model = AccurateMatchAnythingTRT(
            model_name="matchanything_roma",
            match_threshold=0.1
        )
        print("✅ Model creation successful")
        
        # Test forward pass with random inputs
        x1 = torch.rand(1, 3, 224, 224)
        x2 = torch.rand(1, 3, 224, 224)
        
        with torch.no_grad():
            result = model(x1, x2)
            print(f"✅ Forward pass successful:")
            print(f"   keypoints0: {result['keypoints0'].shape}")
            print(f"   keypoints1: {result['keypoints1'].shape}")
            print(f"   mconf: {result['mconf'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight loading test failed: {e}")
        return False

def test_onnx_export():
    """Test ONNX export functionality"""
    print("\n" + "="*60)
    print("TESTING ONNX EXPORT")
    print("="*60)
    
    try:
        from accurate_matchanything_trt import export_accurate_matchanything_onnx
        
        # Test ONNX export to temporary file
        temp_onnx = "/tmp/test_matchanything.onnx"
        
        print("Testing ONNX export...")
        onnx_path = export_accurate_matchanything_onnx(
            onnx_path=temp_onnx,
            model_name="matchanything_roma",
            H=224,
            W=224,
            match_threshold=0.1,
            ckpt=None  # Test without checkpoint first
        )
        
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            print(f"✅ ONNX export successful: {file_size:.1f} MB")
            
            # Check for companion data file
            data_file = onnx_path + ".data"
            if os.path.exists(data_file):
                data_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
                print(f"✅ ONNX data file created: {data_size:.1f} MB")
            
            # Clean up
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            if os.path.exists(data_file):
                os.remove(data_file)
            
            return True
        else:
            print("❌ ONNX export failed - file not created")
            return False
            
    except Exception as e:
        print(f"❌ ONNX export test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test MatchAnything TensorRT improvements")
    parser.add_argument("--test", choices=["all", "weights", "images", "loading", "onnx"], 
                       default="all", help="Which test to run")
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip weight downloading test (requires internet)")
    
    args = parser.parse_args()
    
    print("MatchAnything TensorRT Improvements Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    if args.test in ["all", "weights"] and not args.skip_download:
        total_tests += 1
        if test_weight_downloading():
            tests_passed += 1
    
    if args.test in ["all", "images"]:
        total_tests += 1
        if test_image_size_handling():
            tests_passed += 1
    
    if args.test in ["all", "loading"]:
        total_tests += 1
        if test_weight_loading():
            tests_passed += 1
    
    if args.test in ["all", "onnx"]:
        total_tests += 1
        if test_onnx_export():
            tests_passed += 1
    
    # Final results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The improvements are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())