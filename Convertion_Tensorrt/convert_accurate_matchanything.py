# Convertion_Tensorrt/convert_accurate_matchanything.py
import argparse
import torch
import warnings
from pathlib import Path

# allow running from repo root
import sys, os
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from accurate_matchanything_trt import AccurateMatchAnythingTRT, export_accurate_matchanything_onnx
from unified_weight_loader import apply_unified_weight_loading  # <— use your loader

def load_matchanything_ckpt(model: AccurateMatchAnythingTRT, ckpt_path: str) -> None:
    if not ckpt_path or not Path(ckpt_path).is_file():
        warnings.warn(f"[CKPT] Not found or not provided: {ckpt_path}")
        return

    # 1) Use unified loader to map checkpoint -> model keys (includes encoder.dino.*)
    mapped = apply_unified_weight_loading(
        checkpoint_path=ckpt_path,
        model_state_dict=model.state_dict(),
        load_dinov2_components=True,
    )

    # 2) Load what we could map
    msg = model.load_state_dict(mapped, strict=False)
    print(f"[LOAD] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

def main():
    ap = argparse.ArgumentParser("Accurate MatchAnything -> ONNX")
    ap.add_argument("--ckpt", type=str, required=True, help="matchanything_roma.ckpt path")
    ap.add_argument("--onnx", type=str, default="Convertion_Tensorrt/out/accurate_matchanything.onnx")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--H", type=int, default=840)
    ap.add_argument("--W", type=int, default=840)
    args = ap.parse_args()

    print("=" * 60)
    print("ACCURATE MATCHANYTHING TO TENSORRT CONVERSION")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")

    device = torch.device(args.device)
    model = AccurateMatchAnythingTRT(amp=True).to(device).eval()

    # 🔑 Load ROMA backbone + head from the ckpt via your mapper
    load_matchanything_ckpt(model, args.ckpt)

    onnx_path = export_accurate_matchanything_onnx(model, args.onnx, H=args.H, W=args.W)
    print("\n============================================================")
    print("ONNX EXPORT COMPLETE")
    print("============================================================")
    print(f"Output: {onnx_path}\n")
    print("Next: Build TensorRT engine with trtexec:\n")
    print("/usr/src/tensorrt/bin/trtexec \\")
    print(f"    --onnx={onnx_path} \\")
    print("    --saveEngine=Convertion_Tensorrt/out/accurate_matchanything.plan \\")
    print("    --fp16 --memPoolSize=workspace:4096M \\")
    print("    --minShapes=image0:1x3x420x420,image1:1x3x420x420 \\")
    print("    --optShapes=image0:1x3x840x840,image1:1x3x840x840 \\")
    print("    --maxShapes=image0:1x3x1680x1680,image1:1x3x1680x1680 \\")
    print("    --skipInference --verbose")
    print("\nThen run inference:")
    print("python Convertion_Tensorrt/run_accurate_matchanything_trt.py \\")
    print("  --engine Convertion_Tensorrt/out/accurate_matchanything.plan \\")
    print("  --image0 /path/to/img0.jpg --image1 /path/to/img1.jpg")

if __name__ == "__main__":
    main()
