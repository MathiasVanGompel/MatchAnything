# Convertion_Tensorrt/convert_accurate_matchanything.py
import argparse
import torch
import warnings
from pathlib import Path

from accurate_matchanything_trt import AccurateMatchAnythingTRT, export_accurate_matchanything_onnx

def load_matchanything_ckpt_head_only(model: AccurateMatchAnythingTRT, ckpt_path: str) -> None:
    """
    Load ONLY matcher/head weights (strip 'matcher.'), leave ROMA DINOv2 backbone to its own official weights.
    This mirrors the HF Space behavior. See roma/matchanything_roma_model.py::load_state_dict. :contentReference[oaicite:4]{index=4}
    """
    if not ckpt_path or not Path(ckpt_path).is_file():
        warnings.warn(f"[CKPT] Not found or not provided: {ckpt_path}")
        return

    msg = torch.load(ckpt_path, map_location="cpu")
    state = msg.get("state_dict", msg)

    # Strip 'matcher.' prefix as in HF Space
    cleaned = {}
    for k, v in state.items():
        if k.startswith("matcher."):
            cleaned[k.replace("matcher.", "", 1)] = v

    # Only apply to gp_head.* (and any auxiliary layers you placed there)
    head_state = {k.replace("gp_head.", "", 1): v for k, v in cleaned.items() if k.startswith("gp_head.")}
    missing, unexpected = model.gp_head.load_state_dict(head_state, strict=False)
    if missing or unexpected:
        warnings.warn(f"[CKPT head] missing={len(missing)} unexpected={len(unexpected)} (ok if keys differ)")

def main():
    ap = argparse.ArgumentParser("Accurate MatchAnything -> ONNX")
    ap.add_argument("--ckpt", type=str, required=False, help="matchanything_roma.ckpt path")
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

    # Load only head weights from the ckpt (ROMA DINOv2 backbone uses its own official weights)
    if args.ckpt:
        load_matchanything_ckpt_head_only(model, args.ckpt)

    onnx_path = export_accurate_matchanything_onnx(model, args.onnx, H=args.H, W=args.W)
    print("\n============================================================")
    print("ONNX EXPORT COMPLETE")
    print("============================================================")
    print(f"Output: {onnx_path}\n")
    print("Next: Build TensorRT engine with trtexec:\n")
    print("Recommended command:")
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
    print("\n============================================================")
    print("CONVERSION READY")
    print("============================================================")

if __name__ == "__main__":
    main()
