#!/usr/bin/env python3
import argparse
import os
import torch

from accurate_matchanything_trt import AccurateMatchAnythingTRT, export_accurate_matchanything_onnx

def parse_args():
    ap = argparse.ArgumentParser("Accurate MatchAnything → ONNX/TensorRT exporter")
    ap.add_argument("--ckpt", default="", type=str,
                    help="(optional) Path to MatchAnything_roma ckpt. This exporter uses official DINOv2 weights for the encoder; custom heads would need mapping.")
    ap.add_argument("--onnx", default="Convertion_Tensorrt/out/accurate_matchanything_encoder.onnx", type=str)
    ap.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    ap.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16"])
    ap.add_argument("--H", default=518, type=int)
    ap.add_argument("--W", default=518, type=int)
    ap.add_argument("--opset", default=17, type=int)
    return ap.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("ACCURATE MATCHANYTHING TO TENSORRT CONVERSION")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision.upper()}")
    print(f"Input size: {args.H}x{args.W}")
    print("=" * 60)

    if args.ckpt and os.path.exists(args.ckpt):
        print("[INFO] (optional) Found checkpoint at", args.ckpt,
              "This exporter currently builds DINOv2 from official weights; custom heads from the ckpt would need explicit mapping if used.")
    else:
        print("[INFO] No custom ckpt provided; using official DINOv2 weights for encoder.")

    amp = (args.precision.lower() == "fp16")
    device = torch.device(args.device)

    model = AccurateMatchAnythingTRT(input_hw=(args.H, args.W), amp=amp).to(device)

    print("[ONNX] Exporting...")
    onnx_path = export_accurate_matchanything_onnx(
        model, args.onnx, H=args.H, W=args.W, opset=args.opset
    )
    print(f"[OK] Saved: {onnx_path}")

if __name__ == "__main__":
    main()
