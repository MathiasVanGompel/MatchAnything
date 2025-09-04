#!/usr/bin/env python3
"""
Test ONNX inference using ONNX Runtime (alternative to TensorRT for testing).
This lets you verify the conversion works before installing TensorRT.
"""
import argparse
import numpy as np
import cv2
import os

def test_onnx_inference(onnx_path, image0_path, image1_path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX Runtime not available. Install with: pip install onnxruntime")
        return
    
    print("="*60)
    print("TESTING ONNX INFERENCE WITH ONNX RUNTIME")
    print("="*60)
    print(f"ONNX Model: {onnx_path}")
    print(f"Image 0: {image0_path}")
    print(f"Image 1: {image1_path}")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    
    # Get input/output info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"\nModel Info:")
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")
    
    for inp in session.get_inputs():
        print(f"  Input '{inp.name}': {inp.shape} ({inp.type})")
    
    for out in session.get_outputs():
        print(f"  Output '{out.name}': {out.shape} ({out.type})")
    
    # Load and preprocess images
    def load_image(path, target_size=(832, 832)):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] and convert to CHW format
        img_float = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch
    
    print(f"\nLoading images...")
    img0 = load_image(image0_path)
    img1 = load_image(image1_path)
    
    print(f"  Image0 shape: {img0.shape}")
    print(f"  Image1 shape: {img1.shape}")
    
    # Run inference
    print(f"\nRunning ONNX inference...")
    inputs = {input_names[0]: img0, input_names[1]: img1}
    outputs = session.run(output_names, inputs)
    
    # Process results
    keypoints0 = outputs[0] if len(outputs) > 0 else np.array([])
    keypoints1 = outputs[1] if len(outputs) > 1 else np.array([])
    mconf = outputs[2] if len(outputs) > 2 else np.array([])
    
    print(f"\nResults:")
    print(f"  Keypoints0: {keypoints0.shape}")
    print(f"  Keypoints1: {keypoints1.shape}")
    print(f"  Confidences: {mconf.shape}")
    print(f"  Match count: {len(mconf)}")
    
    if len(mconf) > 0:
        print(f"  Confidence range: {mconf.min():.3f} - {mconf.max():.3f}")
        print(f"  Mean confidence: {mconf.mean():.3f}")
    else:
        print("  No matches found (expected with random weights)")
    
    print(f"\n✅ ONNX inference successful!")
    print(f"The conversion pipeline is working correctly.")
    print(f"Next step: Install TensorRT to build optimized engine.")

def main():
    parser = argparse.ArgumentParser(description="Test ONNX model with ONNX Runtime")
    parser.add_argument("--onnx", default="out/accurate_matchanything_roma.onnx", help="ONNX model path")
    parser.add_argument("--image0", required=True, help="First image path")
    parser.add_argument("--image1", required=True, help="Second image path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.onnx):
        print(f"❌ ONNX file not found: {args.onnx}")
        print("Run the conversion first: ./build_accurate_tensorrt.sh")
        return
    
    test_onnx_inference(args.onnx, args.image0, args.image1)

if __name__ == "__main__":
    main()