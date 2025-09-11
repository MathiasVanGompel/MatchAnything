#!/usr/bin/env python3
"""
Export MatchAnything (ROMA) to ONNX with dense outputs (warp_c, cert_c).

Notes:
  • Inputs:  image0, image1   -> [B,3,H,W], float32 in [0,1]
  • Outputs: warp_c            -> [B,Ha,Wa,2]  (coarse grid coords in image1, Ha=H/14, Wa=W/14)
            cert_c            -> [B,Ha,Wa]
  • Dynamic axes: B, H, W, Ha, Wa
  • No in-graph thresholding, no variable-length lists.
"""

import os
import inspect
import torch
import onnx

from Convertion_Tensorrt_new.accurate_matchanything_trt_dyn import (
    AccurateMatchAnythingTRT,
)


# Optional: your weight mapping loader (if available)
def _try_load_weights(model, ckpt_path: str):
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            # Import the unified weight loader from the parent directory
            import sys
            from pathlib import Path
            parent_dir = Path(__file__).parent.parent / "Convertion_Tensorrt"
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            from unified_weight_loader_fixed import apply_unified_weight_loading

            model_state = model.state_dict()
            loadable = apply_unified_weight_loading(
                ckpt_path, model_state, load_dinov2_components=True
            )
            missing, unexpected = model.load_state_dict(loadable, strict=False)
            print(
                f"[WEIGHTS] Loaded={len(loadable)}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
        except Exception as e:
            print(f"[WEIGHTS] Failed unified loader: {e}")
            print(
                "[WEIGHTS] Proceeding WITHOUT weights (random init) — expect low scores."
            )
    else:
        print("[WEIGHTS] No checkpoint file given; continuing without weights.")


def export_onnx(onnx_path: str, ckpt: str = "", H: int = 840, W: int = 840):
    device = "cpu"
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()
    _try_load_weights(model, ckpt)

    # Dummy inputs — pick multiples of 14 (e.g., 840 = 60*14)
    x0 = torch.rand(1, 3, H, W, device=device)
    x1 = torch.rand(1, 3, H, W, device=device)

    # Dry run
    with torch.inference_mode():
        out = model(x0, x1)
        print("Dry run:", out["warp_c"].shape, out["cert_c"].shape)

    dynamic_axes = {
        "image0": {"0": "B", "2": "H", "3": "W"},
        "image1": {"0": "B", "2": "H", "3": "W"},
        "warp_c": {"0": "B", "1": "Ha", "2": "Wa"},
        "cert_c": {"0": "B", "1": "Ha", "2": "Wa"},
    }

    export_kwargs = dict(
        input_names=["image0", "image1"],
        output_names=["warp_c", "cert_c"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["use_external_data_format"] = True

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(model, (x0, x1), onnx_path, **export_kwargs)
    print(f"[ONNX] Saved -> {onnx_path}")

    # (Optional) consolidate external data if created by exporter
    try:
        model_proto = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(model_proto, onnx_path, save_as_external_data=True)
    except Exception as e:
        print(f"[ONNX] External data consolidation skipped: {e}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="out/matchanything_dense.onnx")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--H", type=int, default=840, help="Multiple of 14")
    ap.add_argument("--W", type=int, default=840, help="Multiple of 14")
    args = ap.parse_args()
    export_onnx(args.onnx, args.ckpt, args.H, args.W)
