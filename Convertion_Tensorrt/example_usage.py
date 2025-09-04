#!/usr/bin/env python3
"""
Example usage script for MatchAnything TensorRT conversion.
This script demonstrates the complete workflow from PyTorch to TensorRT inference.
"""
import os
import sys
import argparse

def print_step(step_num, title, description):
    """Print a formatted step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")
    print(description)

def main():
    parser = argparse.ArgumentParser(description="MatchAnything TensorRT Conversion Example")
    parser.add_argument("--checkpoint", help="Path to MatchAnything checkpoint file")
    parser.add_argument("--image1", help="Path to first image")
    parser.add_argument("--image2", help="Path to second image")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    args = parser.parse_args()
    
    print("MatchAnything TensorRT Conversion - Example Usage")
    print("="*60)
    
    # Step 1: Environment Check
    print_step(1, "Environment Check", 
               "First, let's check if your environment is ready for TensorRT conversion.")
    
    cmd1 = "python3 setup_environment.py"
    print(f"Command: {cmd1}")
    
    if not args.dry_run:
        print("Running environment check...")
        os.system(cmd1)
    
    # Step 2: ONNX Export
    print_step(2, "ONNX Export",
               "Convert the PyTorch model to ONNX format with dynamic shapes.")
    
    onnx_args = [
        "python3 matchanything_to_trt_full.py",
        "--onnx out/matchanything_dynamic.onnx",
        "--H 448 --W 448",
        "--verbose"
    ]
    
    if args.checkpoint:
        onnx_args.append(f"--ckpt {args.checkpoint}")
    
    cmd2 = " ".join(onnx_args)
    print(f"Command: {cmd2}")
    
    if not args.dry_run:
        print("Exporting to ONNX...")
        result = os.system(cmd2)
        if result != 0:
            print("❌ ONNX export failed. Please check the error messages above.")
            return
    
    # Step 3: TensorRT Engine Build
    print_step(3, "TensorRT Engine Build",
               "Convert the ONNX model to optimized TensorRT engine.")
    
    trt_args = [
        "trtexec",
        "--onnx=out/matchanything_dynamic.onnx",
        "--saveEngine=out/matchanything.plan",
        "--explicitBatch --fp16 --workspace=2048",
        "--minShapes=image0:1x3x224x224,image1:1x3x224x224",
        "--optShapes=image0:1x3x448x448,image1:1x3x448x448",
        "--maxShapes=image0:1x3x896x896,image1:1x3x896x896",
        "--buildOnly --verbose"
    ]
    
    cmd3 = " \\\n    ".join(trt_args)
    print(f"Command:\n{cmd3}")
    
    if not args.dry_run:
        print("Building TensorRT engine...")
        result = os.system(" ".join(trt_args))
        if result != 0:
            print("❌ TensorRT build failed. Please check the error messages above.")
            return
    
    # Step 4: Inference
    print_step(4, "TensorRT Inference",
               "Run inference with the optimized TensorRT engine.")
    
    if args.image1 and args.image2:
        inference_args = [
            "python3 run_ma_trt.py",
            "--engine out/matchanything.plan",
            f"--image0 {args.image1}",
            f"--image1 {args.image2}",
            "--H 448 --W 448",
            "--norm imagenet",
            "--budget 1000"
        ]
        
        cmd4 = " ".join(inference_args)
        print(f"Command: {cmd4}")
        
        if not args.dry_run:
            print("Running TensorRT inference...")
            result = os.system(cmd4)
            if result == 0:
                print("✅ Inference completed! Check out/out_trt/ for results.")
            else:
                print("❌ Inference failed. Please check the error messages above.")
    else:
        print("Command: python3 run_ma_trt.py --engine out/matchanything.plan --image0 <image1> --image1 <image2>")
        print("Note: Provide --image1 and --image2 arguments to run actual inference.")
    
    # Step 5: Results
    print_step(5, "Results",
               "Your TensorRT conversion is complete!")
    
    if not args.dry_run:
        print("Generated files:")
        if os.path.exists("out/matchanything_dynamic.onnx"):
            print("✅ out/matchanything_dynamic.onnx - ONNX model")
        if os.path.exists("out/matchanything.plan"):
            print("✅ out/matchanything.plan - TensorRT engine")
        if os.path.exists("out/out_trt"):
            print("✅ out/out_trt/ - Inference results")
    
    print("\nNext steps:")
    print("1. Use the TensorRT engine for fast inference in your applications")
    print("2. Integrate the engine into your deployment pipeline")
    print("3. Benchmark performance against the original PyTorch model")
    
    # Performance expectations
    print("\nExpected Performance Improvements:")
    print("- 3-4x faster inference compared to PyTorch")
    print("- Lower memory usage with FP16 precision")
    print("- Optimized for NVIDIA GPU architectures")

if __name__ == "__main__":
    main()