# ONNX exporter mirroring your original CLI, with a --head flag,
# and a PYTHONPATH bootstrap so RoMa's package resolves.

import os, sys
import argparse
import torch

# --- BEGIN: RoMa path bootstrap (fixes ImportError) ---
THIS = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(THIS, "..", ".."))
ROMA_CANDIDATES = [
    os.path.join(REPO, "imcui", "third_party", "MatchAnything", "third_party", "ROMA"),
    os.path.join(REPO, "imcui", "third_party", "MatchAnything", "third_party", "RoMa"),
    os.path.join(REPO, "imcui", "third_party", "ROMA"),
    os.path.join(REPO, "third_party", "ROMA"),
]
for p in ROMA_CANDIDATES:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
# --- END: RoMa path bootstrap ---

from full_matchanything_trt_plus import FullMatchAnythingTRTPlus
from unified_weight_loader_plus import build_mapped_state_dict

def _flatten_state_dict(raw):
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
    return raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--onnx", required=True, type=str)
    ap.add_argument("--H", required=True, type=int)
    ap.add_argument("--W", required=True, type=int)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--opset", default=17, type=int)
    ap.add_argument("--head", default="gp", choices=["gp", "decoder", "decoder_refine"])
    ap.add_argument("--beta", default=14.285714285714286, type=float)  # 1/0.07
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = args.precision == "fp16"

    model = FullMatchAnythingTRTPlus(
        use_head=args.head,
        gp_beta=args.beta,
        encoder_kwargs={"input_hw": (args.H, args.W), "amp": amp},
        upsample_res=(args.H, args.W)
    ).to(device).eval()

    # Only try ckpt mapping for decoder paths. GP uses your baseline init anyway.
    if args.head != "gp" and os.path.isfile(args.ckpt):
        raw = torch.load(args.ckpt, map_location="cpu")
        ckpt = _flatten_state_dict(raw)
        mapped = build_mapped_state_dict(ckpt, model.state_dict())
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        print(f"[loader] loaded: {len(mapped)} tensors | missing: {len(missing)} | unexpected: {len(unexpected)}")
    elif args.head != "gp":
        print(f"[warn] ckpt not found at {args.ckpt} — exporting with current weights.")
    else:
        print("[info] head='gp': skipping external ckpt load (not needed).")

    if amp:
        model.half()

    dtype = torch.float16 if amp else torch.float32
    imgA = torch.zeros(1, 3, args.H, args.W, device=device, dtype=dtype)
    imgB = torch.zeros_like(imgA)

    os.makedirs(os.path.dirname(args.onnx), exist_ok=True)
    torch.onnx.export(
        model, (imgA, imgB), args.onnx,
        input_names=["imgA", "imgB"], output_names=["warp", "certainty"],
        opset_version=int(args.opset), dynamic_axes=None
    )
    print(f"[ok] ONNX written to {args.onnx}")

if __name__ == "__main__":
    main()
