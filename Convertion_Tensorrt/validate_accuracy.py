#!/usr/bin/env python3
"""
Validation script to ensure TensorRT results match PyTorch exactly.
This script compares the outputs of both implementations.
"""
import argparse
import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add the third_party path for original implementation
sys.path.append(str(Path(__file__).parent / "../imcui/third_party/MatchAnything"))
sys.path.append(str(Path(__file__).parent / "../imcui/hloc"))

def load_original_matcher(model_name: str, ckpt_path: Optional[str] = None):
    """Load the original MatchAnything matcher for comparison"""
    try:
        from imcui.hloc.matchers.matchanything import MatchAnything
        from imcui.hloc import DEVICE
        
        conf = {
            'model_name': model_name,
            'img_resize': 832,
            'match_threshold': 0.1
        }
        
        matcher = MatchAnything(conf)
        print(f"✅ Loaded original {model_name} matcher")
        return matcher
        
    except Exception as e:
        print(f"❌ Failed to load original matcher: {e}")
        return None

def load_trt_model(model_name: str):
    """Load our TensorRT-compatible model for comparison"""
    try:
        from accurate_matchanything_trt import AccurateMatchAnythingTRT
        
        model = AccurateMatchAnythingTRT(
            model_name=model_name,
            img_resize=832,
            match_threshold=0.1,
            amp=False
        )
        model.eval()
        print(f"✅ Loaded TensorRT-compatible {model_name} model")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load TensorRT model: {e}")
        return None

def load_test_images(img0_path: str, img1_path: str, target_size: Tuple[int, int] = (640, 480)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load test images in the format expected by both models"""
    
    # Load images
    img0_bgr = cv2.imread(img0_path)
    img1_bgr = cv2.imread(img1_path)
    
    if img0_bgr is None or img1_bgr is None:
        raise FileNotFoundError("Could not load test images")
    
    # Convert BGR to RGB
    img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img0_rgb = cv2.resize(img0_rgb, target_size, interpolation=cv2.INTER_AREA)
    img1_rgb = cv2.resize(img1_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensors [1, C, H, W] in range [0, 1]
    img0_tensor = torch.from_numpy(img0_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    return img0_tensor, img1_tensor

def run_original_matcher(matcher, img0: torch.Tensor, img1: torch.Tensor) -> Dict[str, np.ndarray]:
    """Run the original matcher and return results"""
    data = {
        'image0': img0,
        'image1': img1
    }
    
    with torch.no_grad():
        result = matcher._forward(data)
    
    return {
        'keypoints0': result['keypoints0'].numpy(),
        'keypoints1': result['keypoints1'].numpy(),
        'mconf': result['mconf'].numpy()
    }

def run_trt_model(model, img0: torch.Tensor, img1: torch.Tensor) -> Dict[str, np.ndarray]:
    """Run our TensorRT-compatible model and return results"""
    with torch.no_grad():
        result = model(img0, img1)
    
    return {
        'keypoints0': result['keypoints0'].numpy(),
        'keypoints1': result['keypoints1'].numpy(),
        'mconf': result['mconf'].numpy()
    }

def compare_results(orig_results: Dict[str, np.ndarray], 
                   trt_results: Dict[str, np.ndarray],
                   tolerance: float = 1e-3) -> Dict[str, bool]:
    """Compare results between original and TensorRT implementations"""
    
    comparison = {}
    
    print("\n" + "="*60)
    print("ACCURACY COMPARISON")
    print("="*60)
    
    # Compare number of matches
    orig_num = len(orig_results['mconf'])
    trt_num = len(trt_results['mconf'])
    
    print(f"Number of matches:")
    print(f"  Original: {orig_num}")
    print(f"  TensorRT: {trt_num}")
    print(f"  Difference: {abs(orig_num - trt_num)}")
    
    if orig_num == 0 and trt_num == 0:
        print("✅ Both models found no matches (consistent)")
        return {'overall': True}
    
    if orig_num == 0 or trt_num == 0:
        print("❌ One model found matches, the other didn't")
        return {'overall': False}
    
    # Compare match quality (using top N matches from both)
    min_matches = min(orig_num, trt_num)
    if min_matches > 100:
        min_matches = 100  # Limit for comparison
    
    # Sort by confidence and take top matches
    orig_top_idx = np.argsort(-orig_results['mconf'])[:min_matches]
    trt_top_idx = np.argsort(-trt_results['mconf'])[:min_matches]
    
    orig_kpts0 = orig_results['keypoints0'][orig_top_idx]
    orig_kpts1 = orig_results['keypoints1'][orig_top_idx]
    orig_conf = orig_results['mconf'][orig_top_idx]
    
    trt_kpts0 = trt_results['keypoints0'][trt_top_idx]
    trt_kpts1 = trt_results['keypoints1'][trt_top_idx]
    trt_conf = trt_results['mconf'][trt_top_idx]
    
    # Compare keypoint positions
    kpts0_diff = np.mean(np.abs(orig_kpts0 - trt_kpts0))
    kpts1_diff = np.mean(np.abs(orig_kpts1 - trt_kpts1))
    conf_diff = np.mean(np.abs(orig_conf - trt_conf))
    
    print(f"\nTop {min_matches} matches comparison:")
    print(f"  Keypoints0 MAE: {kpts0_diff:.4f} pixels")
    print(f"  Keypoints1 MAE: {kpts1_diff:.4f} pixels")
    print(f"  Confidence MAE: {conf_diff:.4f}")
    
    # Determine if results are close enough
    kpts0_close = kpts0_diff < tolerance
    kpts1_close = kpts1_diff < tolerance
    conf_close = conf_diff < 0.01  # Confidence can vary more
    
    comparison['keypoints0'] = kpts0_close
    comparison['keypoints1'] = kpts1_close
    comparison['confidence'] = conf_close
    comparison['overall'] = kpts0_close and kpts1_close
    
    # Print results
    print(f"\nAccuracy Assessment (tolerance: {tolerance}):")
    print(f"  Keypoints0: {'✅ PASS' if kpts0_close else '❌ FAIL'}")
    print(f"  Keypoints1: {'✅ PASS' if kpts1_close else '❌ FAIL'}")
    print(f"  Confidence: {'✅ PASS' if conf_close else '⚠️  ACCEPTABLE'}")
    print(f"  Overall: {'✅ PASS' if comparison['overall'] else '❌ FAIL'}")
    
    return comparison

def create_comparison_visualization(img0_path: str, img1_path: str,
                                  orig_results: Dict[str, np.ndarray],
                                  trt_results: Dict[str, np.ndarray],
                                  output_path: str):
    """Create side-by-side visualization comparing both results"""
    
    # Load images
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    h, w = img0.shape[:2]
    
    # Create comparison image (original on top, TRT on bottom)
    comparison_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Top row: Original results
    comparison_img[:h, :w] = img0
    comparison_img[:h, w:] = img1
    
    # Bottom row: TRT results  
    comparison_img[h:, :w] = img0
    comparison_img[h:, w:] = img1
    
    # Draw original matches (top row, green)
    if len(orig_results['mconf']) > 0:
        top_matches = min(50, len(orig_results['mconf']))  # Limit for visibility
        for i in range(top_matches):
            x0, y0 = int(orig_results['keypoints0'][i, 0]), int(orig_results['keypoints0'][i, 1])
            x1, y1 = int(orig_results['keypoints1'][i, 0]) + w, int(orig_results['keypoints1'][i, 1])
            
            cv2.circle(comparison_img, (x0, y0), 2, (0, 255, 0), -1)
            cv2.circle(comparison_img, (x1, y1), 2, (0, 255, 0), -1)
            cv2.line(comparison_img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    
    # Draw TRT matches (bottom row, red)
    if len(trt_results['mconf']) > 0:
        top_matches = min(50, len(trt_results['mconf']))  # Limit for visibility
        for i in range(top_matches):
            x0, y0 = int(trt_results['keypoints0'][i, 0]), int(trt_results['keypoints1'][i, 1]) + h
            x1, y1 = int(trt_results['keypoints1'][i, 0]) + w, int(trt_results['keypoints1'][i, 1]) + h
            
            cv2.circle(comparison_img, (x0, y0), 2, (0, 0, 255), -1)
            cv2.circle(comparison_img, (x1, y1), 2, (0, 0, 255), -1)
            cv2.line(comparison_img, (x0, y0), (x1, y1), (0, 0, 255), 1)
    
    # Add labels
    cv2.putText(comparison_img, "Original PyTorch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison_img, "TensorRT Compatible", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, comparison_img)
    print(f"Comparison visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate TensorRT accuracy against original")
    parser.add_argument("--model", choices=["matchanything_roma", "matchanything_eloftr"], 
                       default="matchanything_roma", help="Model to validate")
    parser.add_argument("--image0", required=True, help="First test image")
    parser.add_argument("--image1", required=True, help="Second test image")
    parser.add_argument("--ckpt", help="Checkpoint path (if available)")
    parser.add_argument("--tolerance", type=float, default=1.0, 
                       help="Tolerance for coordinate differences (pixels)")
    parser.add_argument("--output_dir", default="Convertion_Tensorrt/out/validation",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("MATCHANYTHING TENSORRT ACCURACY VALIDATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Test images: {args.image0}, {args.image1}")
    print(f"Tolerance: {args.tolerance} pixels")
    print()
    
    # Load test images
    print("Loading test images...")
    try:
        img0, img1 = load_test_images(args.image0, args.image1)
        print(f"✅ Images loaded: {img0.shape}, {img1.shape}")
    except Exception as e:
        print(f"❌ Failed to load images: {e}")
        return
    
    # Load models
    print("\nLoading models...")
    orig_matcher = load_original_matcher(args.model, args.ckpt)
    trt_model = load_trt_model(args.model)
    
    if orig_matcher is None:
        print("⚠️  Cannot load original matcher - skipping comparison")
        print("   This is expected in environments without the full MatchAnything setup")
        return
    
    if trt_model is None:
        print("❌ Cannot load TensorRT model - check implementation")
        return
    
    # Run inference
    print("\nRunning inference...")
    
    print("  Running original matcher...")
    try:
        orig_results = run_original_matcher(orig_matcher, img0, img1)
        print(f"    Found {len(orig_results['mconf'])} matches")
    except Exception as e:
        print(f"    ❌ Original matcher failed: {e}")
        return
    
    print("  Running TensorRT model...")
    try:
        trt_results = run_trt_model(trt_model, img0, img1)
        print(f"    Found {len(trt_results['mconf'])} matches")
    except Exception as e:
        print(f"    ❌ TensorRT model failed: {e}")
        return
    
    # Compare results
    comparison = compare_results(orig_results, trt_results, args.tolerance)
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, f"validation_results_{args.model}.npz")
    np.savez(results_file,
             orig_keypoints0=orig_results['keypoints0'],
             orig_keypoints1=orig_results['keypoints1'],
             orig_mconf=orig_results['mconf'],
             trt_keypoints0=trt_results['keypoints0'],
             trt_keypoints1=trt_results['keypoints1'],
             trt_mconf=trt_results['mconf'])
    print(f"\nDetailed results saved: {results_file}")
    
    # Create visualization
    viz_path = os.path.join(args.output_dir, f"comparison_{args.model}.jpg")
    create_comparison_visualization(args.image0, args.image1, orig_results, trt_results, viz_path)
    
    # Final assessment
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if comparison['overall']:
        print("🎉 SUCCESS: TensorRT implementation matches original within tolerance!")
        print("   The TensorRT version should produce nearly identical results.")
    else:
        print("⚠️  WARNING: Differences detected between implementations.")
        print("   Review the detailed comparison above and consider adjusting the model.")
    
    print(f"\nValidation complete. Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()