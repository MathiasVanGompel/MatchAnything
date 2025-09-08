#!/usr/bin/env python3
"""
Improved MatchAnything TensorRT inference script with proper size handling.
This version fixes the reshape errors and dimension mismatches.
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from typing import Dict, Tuple, Optional
import PIL.Image as Image

def ensure_dino_compatible_size(h: int, w: int) -> Tuple[int, int]:
    """
    Ensure dimensions are compatible with DINOv2 requirements.
    DINOv2 requires dimensions to be multiples of 14 (patch size).
    """
    # Round to nearest multiple of 14
    h_new = ((h + 13) // 14) * 14
    w_new = ((w + 13) // 14) * 14
    
    # Ensure minimum size
    h_new = max(h_new, 224)
    w_new = max(w_new, 224)
    
    return h_new, w_new

def load_image_rgb(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image as RGB numpy array with improved resizing"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load with PIL to ensure RGB format
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    
    if target_size:
        # Use high-quality resizing that preserves aspect ratio
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with high quality interpolation
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to target size if needed
        if new_w != target_w or new_h != target_h:
            # Create black canvas
            padded = np.zeros((target_h, target_w, 3), dtype=img.dtype)
            # Center the image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
            img = padded
    
    return img

def preprocess_for_tensorrt(img: np.ndarray) -> np.ndarray:
    """Preprocess image for TensorRT inference."""
    img_float = img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch

class ImprovedTensorRTEngine:
    """Improved TensorRT engine wrapper with proper size handling"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.INFO)
        
        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        print(f"Engine loaded: {engine_path}")
        print(f"Inputs: {self.input_names}")
        print(f"Outputs: {self.output_names}")
    
    def infer(self, image0: np.ndarray, image1: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference with proper size handling"""
        
        # Ensure both images have DINOv2-compatible sizes
        _, _, h0, w0 = image0.shape
        _, _, h1, w1 = image1.shape
        
        # Find common DINOv2-compatible size
        target_h, target_w = ensure_dino_compatible_size(max(h0, h1), max(w0, w1))
        
        print(f"Original shapes: {image0.shape}, {image1.shape}")
        print(f"Target DINOv2-compatible size: {target_h}x{target_w}")
        
        # Resize both images to the same size
        if image0.shape[2] != target_h or image0.shape[3] != target_w:
            image0 = torch.nn.functional.interpolate(
                torch.from_numpy(image0), 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            ).numpy()
        
        if image1.shape[2] != target_h or image1.shape[3] != target_w:
            image1 = torch.nn.functional.interpolate(
                torch.from_numpy(image1), 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            ).numpy()
        
        print(f"Resized shapes: {image0.shape}, {image1.shape}")
        
        # Set input shapes for dynamic engine
        for name in self.input_names:
            if name == "image0":
                self.context.set_input_shape(name, image0.shape)
            elif name == "image1":
                self.context.set_input_shape(name, image1.shape)
        
        # Allocate GPU memory
        stream = cuda.Stream()
        
        # Input bindings
        d_image0 = cuda.mem_alloc(image0.nbytes)
        d_image1 = cuda.mem_alloc(image1.nbytes)
        
        # Output bindings with conservative estimates
        d_outputs = {}
        h_outputs = {}
        
        for name in self.output_names:
            if "keypoints" in name:
                shape = (5000, 2)  # Conservative estimate
                dtype = np.float32
            elif "mconf" in name:
                shape = (5000,)    # Conservative estimate  
                dtype = np.float32
            else:
                shape = (1000,)    # Generic fallback
                dtype = np.float32
            
            h_outputs[name] = np.empty(shape, dtype=dtype)
            d_outputs[name] = cuda.mem_alloc(h_outputs[name].nbytes)
            self.context.set_tensor_address(name, int(d_outputs[name]))
        
        # Set input addresses
        self.context.set_tensor_address("image0", int(d_image0))
        self.context.set_tensor_address("image1", int(d_image1))
        
        # Copy inputs to GPU
        cuda.memcpy_htod_async(d_image0, image0, stream)
        cuda.memcpy_htod_async(d_image1, image1, stream)
        
        # Execute
        success = self.context.execute_async_v3(stream.handle)
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Copy outputs back
        for name in self.output_names:
            cuda.memcpy_dtoh_async(h_outputs[name], d_outputs[name], stream)
        
        stream.synchronize()
        
        # Trim outputs to actual size
        if "mconf" in h_outputs:
            mconf_data = h_outputs["mconf"]
            nonzero_indices = np.nonzero(mconf_data)[0]
            if len(nonzero_indices) > 0:
                actual_size = nonzero_indices[-1] + 1
                for name in self.output_names:
                    if "keypoints" in name:
                        h_outputs[name] = h_outputs[name][:actual_size]
                    elif "mconf" in name:
                        h_outputs[name] = h_outputs[name][:actual_size]
        
        return h_outputs

def extract_matches_from_results(keypoints0: np.ndarray, keypoints1: np.ndarray, 
                                mconf: np.ndarray, confidence_threshold: float = 0.1) -> np.ndarray:
    """Extract high-confidence matches"""
    if len(mconf) == 0:
        return np.empty((0, 4), dtype=np.float32)
    
    conf_mask = mconf >= confidence_threshold
    if not np.any(conf_mask):
        return np.empty((0, 4), dtype=np.float32)
    
    kpts0_conf = keypoints0[conf_mask]
    kpts1_conf = keypoints1[conf_mask]
    
    matches = np.column_stack([
        kpts0_conf[:, 0], kpts0_conf[:, 1],
        kpts1_conf[:, 0], kpts1_conf[:, 1]
    ])
    
    return matches.astype(np.float32)

def draw_matches_visualization(img0_path: str, img1_path: str, matches: np.ndarray, 
                              output_path: str, max_matches: int = 1000):
    """Create visualization of matches"""
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
    if img0 is None or img1 is None:
        print("Warning: Could not load images for visualization")
        return
    
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    h_max = max(h0, h1)
    combined = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
    combined[:h0, :w0] = img0
    combined[:h1, w0:w0+w1] = img1
    
    if len(matches) > max_matches:
        indices = np.random.choice(len(matches), max_matches, replace=False)
        matches_vis = matches[indices]
    else:
        matches_vis = matches
    
    np.random.seed(42)
    for i, (x0, y0, x1, y1) in enumerate(matches_vis):
        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1)) + w0, int(round(y1)))
        
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        cv2.circle(combined, pt0, 3, color, -1)
        cv2.circle(combined, pt1, 3, color, -1)
        cv2.line(combined, pt0, pt1, color, 1, cv2.LINE_AA)
    
    cv2.imwrite(output_path, combined)
    print(f"Visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Improved MatchAnything TensorRT Inference")
    parser.add_argument("--engine", required=True, help="TensorRT engine file")
    parser.add_argument("--image0", required=True, help="First image path")
    parser.add_argument("--image1", required=True, help="Second image path")
    parser.add_argument("--output_dir", default="out/improved_results", help="Output directory")
    parser.add_argument("--confidence_threshold", type=float, default=0.1, help="Match confidence threshold")
    parser.add_argument("--max_matches_viz", type=int, default=1000, help="Max matches to visualize")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("IMPROVED MATCHANYTHING TENSORRT INFERENCE")
    print("="*60)
    print(f"Engine: {args.engine}")
    print(f"Image 0: {args.image0}")
    print(f"Image 1: {args.image1}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print()
    
    # Load TensorRT engine
    engine = ImprovedTensorRTEngine(args.engine)
    
    # Use a DINOv2-compatible default size
    target_size = (840, 840)  # Multiple of 14
    print(f"Using DINOv2-compatible target size: {target_size}")
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    img0_rgb = load_image_rgb(args.image0, target_size)
    img1_rgb = load_image_rgb(args.image1, target_size)
    
    img0_tensor = preprocess_for_tensorrt(img0_rgb)
    img1_tensor = preprocess_for_tensorrt(img1_rgb)
    
    print(f"Preprocessed shapes: {img0_tensor.shape}, {img1_tensor.shape}")
    
    # Run inference
    print("Running improved TensorRT inference...")
    try:
        results = engine.infer(img0_tensor, img1_tensor)
        
        keypoints0 = results['keypoints0']
        keypoints1 = results['keypoints1'] 
        mconf = results['mconf']
        
        print(f"Raw results:")
        print(f"  Keypoints0: {keypoints0.shape}")
        print(f"  Keypoints1: {keypoints1.shape}")
        print(f"  Confidences: {mconf.shape}")
        
        matches = extract_matches_from_results(keypoints0, keypoints1, mconf, args.confidence_threshold)
        
        print(f"\nFiltered matches: {len(matches)}")
        if len(matches) > 0:
            print(f"  Confidence range: {mconf.min():.3f} - {mconf.max():.3f}")
            print(f"  Mean confidence: {mconf.mean():.3f}")
        
        # Save results
        results_file = os.path.join(args.output_dir, "matches.npz")
        np.savez(results_file, matches=matches, keypoints0=keypoints0, keypoints1=keypoints1, mconf=mconf)
        print(f"Results saved: {results_file}")
        
        # Create visualization
        if len(matches) > 0:
            viz_path = os.path.join(args.output_dir, "matches_visualization.jpg")
            draw_matches_visualization(args.image0, args.image1, matches, viz_path, args.max_matches_viz)
        else:
            print("No matches found for visualization")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
