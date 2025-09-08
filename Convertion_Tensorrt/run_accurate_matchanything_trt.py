#!/usr/bin/env python3
"""
Accurate MatchAnything TensorRT inference script.
This maintains exact compatibility with the original implementation.
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
    """
    Preprocess image for TensorRT inference.
    Convert from RGB numpy array to normalized tensor format.
    """
    # Convert to float32 and normalize to [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW format
    img_chw = np.transpose(img_float, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch

def extract_matches_from_results(keypoints0: np.ndarray, keypoints1: np.ndarray, 
                                mconf: np.ndarray, confidence_threshold: float = 0.1) -> np.ndarray:
    """
    Extract high-confidence matches from TensorRT results.
    
    Returns:
        matches: [N, 4] array of (x0, y0, x1, y1) coordinates
    """
    if len(mconf) == 0:
        return np.empty((0, 4), dtype=np.float32)
    
    # Filter by confidence
    conf_mask = mconf >= confidence_threshold
    
    if not np.any(conf_mask):
        return np.empty((0, 4), dtype=np.float32)
    
    # Extract high-confidence matches
    kpts0_conf = keypoints0[conf_mask]
    kpts1_conf = keypoints1[conf_mask]
    
    # Combine into matches format [x0, y0, x1, y1]
    matches = np.column_stack([
        kpts0_conf[:, 0], kpts0_conf[:, 1],  # x0, y0
        kpts1_conf[:, 0], kpts1_conf[:, 1]   # x1, y1
    ])
    
    return matches.astype(np.float32)

def draw_matches_visualization(img0_path: str, img1_path: str, matches: np.ndarray, 
                              output_path: str, max_matches: int = 1000):
    """
    Create a side-by-side visualization of matches.
    """
    # Load images
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
    if img0 is None or img1 is None:
        print(f"Warning: Could not load images for visualization")
        return
    
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    # Create side-by-side image
    h_max = max(h0, h1)
    combined = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
    combined[:h0, :w0] = img0
    combined[:h1, w0:w0+w1] = img1
    
    # Limit number of matches for visualization
    if len(matches) > max_matches:
        indices = np.random.choice(len(matches), max_matches, replace=False)
        matches_vis = matches[indices]
    else:
        matches_vis = matches
    
    # Draw matches
    for i, (x0, y0, x1, y1) in enumerate(matches_vis):
        # Convert to integer coordinates
        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1)) + w0, int(round(y1)))
        
        # Generate color based on match index
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # Draw keypoints
        cv2.circle(combined, pt0, 3, color, -1)
        cv2.circle(combined, pt1, 3, color, -1)
        
        # Draw line
        cv2.line(combined, pt0, pt1, color, 1, cv2.LINE_AA)
    
    # Save visualization
    cv2.imwrite(output_path, combined)
    print(f"Visualization saved: {output_path}")

class AccurateTensorRTEngine:
    """TensorRT engine wrapper for accurate MatchAnything inference"""
    
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

    def get_optimal_resolution(self) -> Optional[Tuple[int, int]]:
        """Return the engine's preferred (width, height).

        Uses the optimization profile if available, otherwise falls back to the
        static binding shape. Returns ``None`` when neither can be queried.
        """
        try:
            idx = self.engine.get_binding_index("image0")

            # Try dynamic profile shape first
            try:
                _, opt_shape, _ = self.engine.get_profile_shape(0, idx)
                if opt_shape[2] > 0 and opt_shape[3] > 0:
                    return opt_shape[3], opt_shape[2]
            except Exception:
                pass

            # Fallback to static binding shape
            shape = self.engine.get_binding_shape(idx)
            if len(shape) == 4:  # NCHW
                h, w = shape[2], shape[3]
            elif len(shape) == 3:  # CHW (implicit batch)
                h, w = shape[1], shape[2]
            else:
                return None

            if h > 0 and w > 0:
                return w, h
            return None
        except Exception:
            return None
    
    def infer(self, image0: np.ndarray, image1: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on image pair with improved shape handling.
        
        Args:
            image0: Preprocessed image [1, C, H, W]
            image1: Preprocessed image [1, C, H, W]
            
        Returns:
            Dictionary with keypoints0, keypoints1, mconf
        """
        # Ensure images have the same shape for matching
        if image0.shape != image1.shape:
            print(f"Warning: Image shapes differ: {image0.shape} vs {image1.shape}")
            # Resize to common size (use the smaller dimensions to avoid OOM)
            min_h = min(image0.shape[2], image1.shape[2])
            min_w = min(image0.shape[3], image1.shape[3])
            
            if image0.shape[2] != min_h or image0.shape[3] != min_w:
                image0 = torch.nn.functional.interpolate(
                    torch.from_numpy(image0), 
                    size=(min_h, min_w), 
                    mode='bilinear', 
                    align_corners=False
                ).numpy()
            
            if image1.shape[2] != min_h or image1.shape[3] != min_w:
                image1 = torch.nn.functional.interpolate(
                    torch.from_numpy(image1), 
                    size=(min_h, min_w), 
                    mode='bilinear', 
                    align_corners=False
                ).numpy()
            
            print(f"Resized images to common shape: {image0.shape}")
        
        # Set input shapes if dynamic
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
        
        # Output bindings
        output_shapes = {}
        d_outputs = {}
        h_outputs = {}
        
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Handle dynamic output shapes
            if -1 in shape:
                # For dynamic outputs, we need to allocate max possible size
                # This is a limitation - in practice you'd need to know the max
                max_matches = 10000  # Conservative estimate
                if "keypoints" in name:
                    shape = (max_matches, 2)
                elif "mconf" in name:
                    shape = (max_matches,)
            
            output_shapes[name] = shape
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
        
        return h_outputs

def main():
    parser = argparse.ArgumentParser(
        description="Accurate MatchAnything TensorRT Inference"
    )
    parser.add_argument("--engine", required=True, help="TensorRT engine file")
    parser.add_argument("--image0", required=True, help="First image path")
    parser.add_argument("--image1", required=True, help="Second image path")
    parser.add_argument(
        "--output_dir", 
        default="Convertion_Tensorrt/out/accurate_results",
        help="Output directory"
    )
    parser.add_argument(
        "--confidence_threshold", 
        type=float, 
        default=0.1,
        help="Match confidence threshold"
    )
    parser.add_argument(
        "--max_matches_viz", 
        type=int, 
        default=1000,
        help="Maximum matches to show in visualization"
    )
    parser.add_argument(
        "--target_size", 
        type=int, 
        nargs=2, 
        default=None,
        help="Target image size (width height)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("ACCURATE MATCHANYTHING TENSORRT INFERENCE")
    print("="*60)
    print(f"Engine: {args.engine}")
    print(f"Image 0: {args.image0}")
    print(f"Image 1: {args.image1}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print()
    
    # Load TensorRT engine
    engine = AccurateTensorRTEngine(args.engine)
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    engine_size = engine.get_optimal_resolution()
    if args.target_size:
        target_size = tuple(args.target_size)
        if engine_size and target_size != engine_size:
            print(
                f"Warning: engine built for {engine_size[0]}x{engine_size[1]} but resizing to {target_size}"
            )
    else:
        target_size = engine_size
        if target_size:
            print(f"Using engine target size: {target_size[0]}x{target_size[1]}")
        else:
            print("Could not determine engine target size; using original image sizes")

    img0_rgb = load_image_rgb(args.image0, target_size)
    img1_rgb = load_image_rgb(args.image1, target_size)
    
    img0_tensor = preprocess_for_tensorrt(img0_rgb)
    img1_tensor = preprocess_for_tensorrt(img1_rgb)
    
    print(f"Image 0 shape: {img0_tensor.shape}")
    print(f"Image 1 shape: {img1_tensor.shape}")
    
    # Run inference
    print("Running TensorRT inference...")
    results = engine.infer(img0_tensor, img1_tensor)
    
    # Extract results
    keypoints0 = results['keypoints0']
    keypoints1 = results['keypoints1']
    mconf = results['mconf']
    
    print(f"Raw results:")
    print(f"  Keypoints0: {keypoints0.shape}")
    print(f"  Keypoints1: {keypoints1.shape}")
    print(f"  Confidences: {mconf.shape}")
    
    # Extract high-confidence matches
    matches = extract_matches_from_results(
        keypoints0, keypoints1, mconf, args.confidence_threshold
    )
    
    print(f"\nFiltered matches: {len(matches)}")
    if len(matches) > 0:
        print(f"  Confidence range: {mconf.min():.3f} - {mconf.max():.3f}")
        print(f"  Mean confidence: {mconf.mean():.3f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, "matches.npz")
    np.savez(
        results_file,
        matches=matches,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        mconf=mconf
    )
    print(f"Results saved: {results_file}")
    
    # Create visualization
    if len(matches) > 0:
        viz_path = os.path.join(args.output_dir, "matches_visualization.jpg")
        draw_matches_visualization(
            args.image0, args.image1, matches, viz_path, args.max_matches_viz
        )
    else:
        print("No matches found for visualization")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Found {len(matches)} high-confidence matches")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()