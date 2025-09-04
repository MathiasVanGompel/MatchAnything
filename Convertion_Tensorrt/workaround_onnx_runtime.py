#!/usr/bin/env python3
"""
Workaround: Use ONNX Runtime for inference while we fix the TensorRT export.
This provides the same speed benefits for testing.
"""
import argparse
import numpy as np
import cv2
import os
import time

def run_onnx_runtime_inference(onnx_path, image0_path, image1_path, output_dir):
    """Run inference using ONNX Runtime as a workaround"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
        return
    
    print("="*60)
    print("MATCHANYTHING INFERENCE WITH ONNX RUNTIME")
    print("="*60)
    print(f"ONNX Model: {onnx_path}")
    print(f"Image 0: {image0_path}")
    print(f"Image 1: {image1_path}")
    
    # Create ONNX Runtime session with GPU support
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    print(f"Execution Provider: {session.get_providers()[0]}")
    
    # Get input/output info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"\nModel Info:")
    for inp in session.get_inputs():
        print(f"  Input '{inp.name}': {inp.shape} ({inp.type})")
    for out in session.get_outputs():
        print(f"  Output '{out.name}': {out.shape} ({out.type})")
    
    # Load and preprocess images
    def load_and_preprocess(path, target_size=(832, 832)):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] and convert to CHW format
        img_float = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        
        # Add batch dimension
        return np.expand_dims(img_chw, axis=0)
    
    print(f"\nLoading images...")
    img0 = load_and_preprocess(image0_path)
    img1 = load_and_preprocess(image1_path)
    
    print(f"  Image0 shape: {img0.shape}")
    print(f"  Image1 shape: {img1.shape}")
    
    # Run inference with timing
    print(f"\nRunning inference...")
    start_time = time.time()
    
    inputs = {input_names[0]: img0, input_names[1]: img1}
    outputs = session.run(output_names, inputs)
    
    inference_time = time.time() - start_time
    
    # Process results
    keypoints0 = outputs[0] if len(outputs) > 0 else np.array([])
    keypoints1 = outputs[1] if len(outputs) > 1 else np.array([])
    mconf = outputs[2] if len(outputs) > 2 else np.array([])
    
    print(f"\n🎉 Inference Results:")
    print(f"  ⏱️  Inference time: {inference_time*1000:.1f}ms")
    print(f"  🎯 Keypoints0: {keypoints0.shape}")
    print(f"  🎯 Keypoints1: {keypoints1.shape}")
    print(f"  📊 Confidences: {mconf.shape}")
    print(f"  ✨ Match count: {len(mconf)}")
    
    if len(mconf) > 0:
        print(f"  📈 Confidence range: {mconf.min():.3f} - {mconf.max():.3f}")
        print(f"  📊 Mean confidence: {mconf.mean():.3f}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "onnx_runtime_results.npz")
        np.savez(results_file, 
                keypoints0=keypoints0, 
                keypoints1=keypoints1, 
                mconf=mconf,
                inference_time=inference_time)
        print(f"  💾 Results saved: {results_file}")
        
        # Create visualization
        create_match_visualization(image0_path, image1_path, keypoints0, keypoints1, 
                                 os.path.join(output_dir, "matches.jpg"))
        
    else:
        print("  ⚠️  No matches found")
    
    print(f"\n✅ ONNX Runtime inference successful!")
    return inference_time, len(mconf)

def create_match_visualization(img0_path, img1_path, kpts0, kpts1, output_path):
    """Create match visualization"""
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h_max = max(h0, h1)
    
    # Create side-by-side image
    combined = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
    combined[:h0, :w0] = img0
    combined[:h1, w0:w0+w1] = img1
    
    # Draw matches (limit for visibility)
    max_matches = min(100, len(kpts0))
    for i in range(max_matches):
        x0, y0 = int(kpts0[i, 0]), int(kpts0[i, 1])
        x1, y1 = int(kpts1[i, 0]) + w0, int(kpts1[i, 1])
        
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        cv2.circle(combined, (x0, y0), 3, color, -1)
        cv2.circle(combined, (x1, y1), 3, color, -1)
        cv2.line(combined, (x0, y0), (x1, y1), color, 1)
    
    cv2.imwrite(output_path, combined)
    print(f"  🖼️  Visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime inference workaround")
    parser.add_argument("--onnx", default="out/accurate_matchanything_roma.onnx")
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--output_dir", default="out/onnx_runtime_results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.onnx):
        print(f"❌ ONNX file not found: {args.onnx}")
        print("Run: ./build_accurate_tensorrt.sh first (ONNX export should work)")
        return
    
    inference_time, match_count = run_onnx_runtime_inference(
        args.onnx, args.image0, args.image1, args.output_dir
    )
    
    print(f"\n🎯 Summary:")
    print(f"  Found {match_count} matches in {inference_time*1000:.1f}ms")
    print(f"  ONNX Runtime provides good performance while we fix TensorRT export")

if __name__ == "__main__":
    main()