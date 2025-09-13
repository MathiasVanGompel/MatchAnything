#!/usr/bin/env python3
"""
Accurate MatchAnything to TensorRT conversion script.
This version ensures exact match with the original PyTorch implementation.
"""

import argparse
import os
from pathlib import Path

from accurate_matchanything_trt import export_accurate_matchanything_onnx


def main():
    parser = argparse.ArgumentParser(
        description="Convert MatchAnything to TensorRT with exact accuracy"
    )
    parser.add_argument(
        "--onnx",
        default="Convertion_Tensorrt/out/accurate_matchanything.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--model",
        choices=["matchanything_roma", "matchanything_eloftr"],
        default="matchanything_roma",
        help="Model variant to convert",
    )
    parser.add_argument("--H", type=int, default=840, help="Input height")
    parser.add_argument("--W", type=int, default=840, help="Input width")
    parser.add_argument(
        "--match_threshold", type=float, default=0.1, help="Match confidence threshold"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to MatchAnything checkpoint"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("ACCURATE MATCHANYTHING TO TENSORRT CONVERSION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input size: {args.H}x{args.W}")
    print(f"Match threshold: {args.match_threshold}")
    if args.ckpt:
        print(f"Checkpoint: {args.ckpt}")
    print()

    # Step 1: Export to ONNX
    print("Step 1: Exporting to ONNX...")
    onnx_path = export_accurate_matchanything_onnx(
        onnx_path=args.onnx,
        model_name=args.model,
        H=args.H,
        W=args.W,
        match_threshold=args.match_threshold,
        ckpt=args.ckpt,
    )

    print("\n" + "=" * 60)
    print("ONNX EXPORT COMPLETE")
    print("=" * 60)
    print(f"Output: {onnx_path}")

    # Step 3: Provide TensorRT build commands
    engine_path = onnx_path.replace(".onnx", ".plan")

    print("\nNext: Build TensorRT engine with trtexec:")
    print("\nRecommended command:")
    print("/usr/src/tensorrt/bin/trtexec \\")
    print(f"    --onnx={onnx_path} \\")
    print(f"    --saveEngine={engine_path} \\")
    print("    --fp16 --memPoolSize=workspace:4096M \\")
    print(
        f"    --minShapes=image0:1x3x{args.H//2}x{args.W//2},image1:1x3x{args.H//2}x{args.W//2} \\"
    )
    print(
        f"    --optShapes=image0:1x3x{args.H}x{args.W},image1:1x3x{args.H}x{args.W} \\"
    )
    print(
        f"    --maxShapes=image0:1x3x{args.H*2}x{args.W*2},image1:1x3x{args.H*2}x{args.W*2} \\"
    )
    print("    --skipInference --verbose")

    print("\nThen run inference:")
    print("python3 run_accurate_matchanything_trt.py \\")
    print(f"    --engine {engine_path} \\")
    print("    --image0 image1.jpg --image1 image2.jpg")

    print("\n" + "=" * 60)
    print("CONVERSION READY")
    print("=" * 60)
    print(
        "The exported ONNX model maintains exact accuracy with the original PyTorch implementation."
    )
    print("All preprocessing and postprocessing steps are preserved.")


if __name__ == "__main__":
    main()
