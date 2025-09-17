# ONNX exporter mirroring original, with a --head flag,
# PYTHONPATH bootstrap for RoMa, SAME unified loader as original,
# and optional extra decoder/refiner remap via ckpt_tools_plus.

import os, sys
import argparse
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- BEGIN: RoMa path bootstrap ---
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
# same unified loader as your original converter
from full.unified_weight_loader import apply_unified_weight_loading

# optional local helper for extra decoder/refiner weights
try:
    # local import (file in the same folder)
    from ckpt_tools_plus import try_load_extra_heads
except Exception:  # fallback if someone runs from repo root
    try:
        from Convertion_Tensorrt.full.ckpt_tools_plus import try_load_extra_heads
    except Exception as e:
        print(f"[loader+heads] optional ckpt_tools_plus not available: {e}")
        try_load_extra_heads = None

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
    # optional: a second ckpt to mine decoder/refiner weights from (e.g., official RoMa)
    ap.add_argument("--extra_ckpt", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = args.precision == "fp16"

    model = FullMatchAnythingTRTPlus(
        use_head=args.head,
        gp_beta=args.beta,
        encoder_kwargs={"input_hw": (args.H, args.W), "amp": amp},
        upsample_res=(args.H, args.W)
    ).to(device).eval()

    # --- Load weights like the original (skip when head='gp' to keep your good path unchanged) ---
    if args.head == "gp":
        print("[info] head='gp': skipping external ckpt load (not needed).")
    elif os.path.isfile(args.ckpt):
        model_state = model.state_dict()

        # 1) unified loader (your original mapping + TIMM DINOv2 supplementation)
        loadable = apply_unified_weight_loading(args.ckpt, model_state, load_dinov2_components=True)

        # keep only keys present in model with matching shapes
        filtered = {k: v for k, v in loadable.items()
                    if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)

        # 2) OPTIONAL: try to also load decoder/refiner weights if helper is available
        if try_load_extra_heads is not None and args.head in ("decoder", "decoder_refine"):
            # from main ckpt
            loaded1, cands1 = try_load_extra_heads(model, args.ckpt, verbose=True)
            # optionally from a second ckpt (e.g., a RoMa checkpoint)
            loaded2 = 0
            if args.extra_ckpt and os.path.isfile(args.extra_ckpt):
                loaded2, cands2 = try_load_extra_heads(model, args.extra_ckpt, verbose=True)
            # recalc summary based on what we actually put into the state
            total = len(model.state_dict())
            base_loaded = len(filtered)
            total_loaded = base_loaded + loaded1 + loaded2  # small overcount unlikely; fine for summary
            print(f"[loader] loaded: {total_loaded} / {total} tensors "
                  f"({(total_loaded / max(total,1))*100:.1f}%) | "
                  f"base(enc/gp): {base_loaded}, extra(dec/ref): {loaded1 + loaded2}")
        else:
            # pretty-print base summary (encoder + gp)
            total = len(model.state_dict())
            print(f"[loader] loaded: {len(filtered)} / {total} tensors "
                  f"({(len(filtered)/max(total,1))*100:.1f}%) | "
                  f"missing: {total - len(filtered)}")
    else:
        print(f"[warn] ckpt not found at {args.ckpt} — exporting with current weights.")

    # precision
    if amp:
        model.half()

    # dummy inputs
    dtype = torch.float16 if amp else torch.float32
    imgA = torch.zeros(1, 3, args.H, args.W, device=device, dtype=dtype)
    imgB = torch.zeros_like(imgA)

    # export
    out_dir = os.path.dirname(args.onnx)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.onnx.export(
        model, (imgA, imgB), args.onnx,
        input_names=["imgA", "imgB"], output_names=["warp", "certainty"],
        opset_version=int(args.opset), dynamic_axes=None
    )
    print(f"[ok] ONNX written to {args.onnx}")

if __name__ == "__main__":
    main()
