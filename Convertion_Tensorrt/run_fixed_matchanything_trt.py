#!/usr/bin/env python3
import argparse
import os
import numpy as np
import cv2
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from typing import Dict, Tuple, Optional
import PIL.Image as Image

def preprocess_for_tensorrt(img: np.ndarray) -> np.ndarray:
    """Preprocess image ensuring contiguous arrays."""
    img_float = img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    
    # CRITICAL: Ensure the array is contiguous in memory
    if not img_batch.flags['C_CONTIGUOUS']:
        img_batch = np.ascontiguousarray(img_batch)
    
    return img_batch

def load_image_rgb(path: str, target_size: Tuple[int, int] = (448, 448)) -> np.ndarray:
    """Load and resize image to target size."""
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    
    # Resize to exact target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    return img

class FixedTensorRTEngine:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.INFO)
        
        with open(engine_path, "rb") as f:
            engine_data = f.read()
            engine_size_mb = len(engine_data) / (1024*1024)
            print(f"Engine file size: {engine_size_mb:.2f} MB")
            
            if engine_size_mb < 10:
                print(f"⚠️  WARNING: Engine is very small ({engine_size_mb:.2f} MB)")
                print("   Expected size with DINOv2 weights: 300-500 MB")
                print("   This suggests weights weren't properly loaded during export.")
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get I/O info
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        print(f"Inputs: {self.input_names}")
        print(f"Outputs: {self.output_names}")
    
    def infer(self, image0: np.ndarray, image1: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference with proper contiguous array handling."""
        
        # Ensure arrays are contiguous (CRITICAL FIX)
        if not image0.flags['C_CONTIGUOUS']:
            image0 = np.ascontiguousarray(image0)
        if not image1.flags['C_CONTIGUOUS']:
            image1 = np.ascontiguousarray(image1)
        
        print(f"Input shapes: {image0.shape}, {image1.shape}")
        print(f"Contiguous: {image0.flags['C_CONTIGUOUS']}, {image1.flags['C_CONTIGUOUS']}")
        
        # Set input shapes
        for name in self.input_names:
            if name == "image0":
                self.context.set_input_shape(name, image0.shape)
            elif name == "image1":
                self.context.set_input_shape(name, image1.shape)
        
        # Allocate memory
        stream = cuda.Stream()
        d_image0 = cuda.mem_alloc(image0.nbytes)
        d_image1 = cuda.mem_alloc(image1.nbytes)
        
        # Conservative output estimates
        d_outputs = {}
        h_outputs = {}
        
        for name in self.output_names:
            if "keypoints" in name:
                shape = (1000, 2)
                dtype = np.float32
            elif "mconf" in name:
                shape = (1000,)
                dtype = np.float32
            else:
                shape = (1000,)
                dtype = np.float32
            
            h_outputs[name] = np.ascontiguousarray(np.empty(shape, dtype=dtype))
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
        
        # Trim to actual size
        if "mconf" in h_outputs and len(h_outputs["mconf"]) > 0:
            mconf_data = h_outputs["mconf"]
            nonzero_indices = np.where(mconf_data > 0)[0]
            if len(nonzero_indices) > 0:
                actual_size = nonzero_indices[-1] + 1
                for name in self.output_names:
                    if "keypoints" in name:
                        h_outputs[name] = h_outputs[name][:actual_size]
                    elif "mconf" in name:
                        h_outputs[name] = h_outputs[name][:actual_size]
        
        return h_outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--output_dir", default="out/fixed_results")
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("FIXED MATCHANYTHING TENSORRT INFERENCE")
    print("="*60)
    
    # Load engine
    engine = FixedTensorRTEngine(args.engine)
    
    # Load images with fixed size
    target_size = (448, 448)  # DINOv2 compatible
    print(f"Loading images with target size: {target_size}")
    
    img0_rgb = load_image_rgb(args.image0, target_size)
    img1_rgb = load_image_rgb(args.image1, target_size)
    
    img0_tensor = preprocess_for_tensorrt(img0_rgb)
    img1_tensor = preprocess_for_tensorrt(img1_rgb)
    
    print("Running inference...")
    try:
        results = engine.infer(img0_tensor, img1_tensor)
        
        keypoints0 = results.get('keypoints0', np.array([]))
        keypoints1 = results.get('keypoints1', np.array([]))
        mconf = results.get('mconf', np.array([]))
        
        print(f"Results:")
        print(f"  Keypoints0: {keypoints0.shape}")
        print(f"  Keypoints1: {keypoints1.shape}")
        print(f"  Confidences: {mconf.shape}")
        
        # Filter by confidence
        if len(mconf) > 0:
            conf_mask = mconf >= args.confidence_threshold
            matches_count = np.sum(conf_mask)
            print(f"  Matches above threshold: {matches_count}")
            
            if matches_count > 0:
                print(f"  Confidence range: {mconf.min():.3f} - {mconf.max():.3f}")
                
                # Save results
                results_file = os.path.join(args.output_dir, "matches.npz")
                np.savez(results_file, keypoints0=keypoints0, keypoints1=keypoints1, mconf=mconf)
                print(f"✅ Results saved: {results_file}")
            else:
                print("⚠️  No matches found above threshold")
        else:
            print("❌ No confidence scores returned - model may not be working properly")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔍 DIAGNOSIS:")
        print("The engine is too small (2 MB), indicating missing DINOv2 weights.")
        print("Try rebuilding with the complete checkpoint.")

if __name__ == "__main__":
    main()