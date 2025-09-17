# ONNX exporter that mirrors the original CLI, adds a --head flag,
# It also bootstraps PYTHONPATH so the RoMa package resolves with the same weight loader.

import os, sys
import argparse
import torch

# RoMa path bootstrap to avoid import errors.
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
# End of the RoMa path bootstrap adjustments.

from full_matchanything_trt_plus import FullMatchAnythingTRTPlus
# Import the same weight loader used by the original script.
from unified_weight_loader import apply_unified_weight_loading

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--onnx", required=True, type=str)
    ap.add_argument("--H", required=True, type=int)
    ap.add_argument("--W", required=True, type=int)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--opset", default=17, type=int)
    ap.add_argument("--head", default="gp", choices=["gp", "decoder", "decoder_refine"])
    ap.add_argument("--beta", default=14.285714285714286, type=float)  # Beta defaults to 1/0.07.
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

    # Load weights just like the original converter (encoder and GP fully mapped).
    if os.path.isfile(args.ckpt):
        model_state = model.state_dict()
        # Let the shared loader handle the mapping and TIMM supplementation.
        loadable = apply_unified_weight_loading(args.ckpt, model_state, load_dinov2_components=True)
        # Keep only entries that exist in the model and match shapes.
        filtered = {k: v for k, v in loadable.items()
                    if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
        model_state.update(filtered)
        missing_keys = [k for k in model.state_dict().keys() if k not in filtered]
        model.load_state_dict(model_state, strict=False)
        # Print a summary consistent with the original script.
        total = len(model.state_dict())
        print(f"[loader] loaded: {len(filtered)} / {total} tensors "
              f"({(len(filtered)/max(total,1))*100:.1f}%) | missing: {len(missing_keys)}")
    else:
        print(f"[warn] ckpt not found at {args.ckpt} — exporting with current weights.")

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
