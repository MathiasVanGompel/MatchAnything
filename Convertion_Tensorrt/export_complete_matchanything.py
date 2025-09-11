#!/usr/bin/env python3
"""Complete MatchAnything ONNX export with unified weight loading."""

from pathlib import Path
import os
from typing import Optional

import onnx
import torch

from accurate_matchanything_trt import AccurateMatchAnythingTRT
from unified_weight_loader import apply_unified_weight_loading


def export_complete_matchanything_onnx(
    onnx_path: str,
    checkpoint_path: Optional[str],
    H: int = 832,
    W: int = 832,
    match_threshold: float = 0.1,
    use_external_data: bool = True,
) -> str:
    """Export the full MatchAnything model with weights to ONNX."""
    device = "cpu"
    model = (
        AccurateMatchAnythingTRT(
            model_name="matchanything_roma",
            match_threshold=match_threshold,
            amp=False,
        )
        .to(device)
        .eval()
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        model_state = model.state_dict()
        loadable = apply_unified_weight_loading(
            checkpoint_path, model_state, load_dinov2_components=True
        )
        missing, unexpected = model.load_state_dict(loadable, strict=False)
        print(
            f"[EXPORT] Model loading: {len(loadable)} loaded, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    else:
        print("[EXPORT] No checkpoint provided - using random initialization")

    # Dummy forward for shape discovery
    x0 = torch.rand(1, 3, H, W, device=device)
    x1 = torch.rand(1, 3, H, W, device=device)
    with torch.no_grad():
        out = model(x0, x1)
        print(
            f"[EXPORT] Forward pass OK: {out['keypoints0'].shape} "
            f"{out['keypoints1'].shape}"
        )

    dynamic_axes = {
        "image0": {0: "B", 2: "H", 3: "W"},
        "image1": {0: "B", 2: "H", 3: "W"},
        "keypoints0": {0: "num_matches"},
        "keypoints1": {0: "num_matches"},
        "mconf": {0: "num_matches"},
    }

    export_kwargs = dict(
        input_names=["image0", "image1"],
        output_names=["keypoints0", "keypoints1", "mconf"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (x0, x1), str(onnx_path), **export_kwargs)
    print(f"[EXPORT] Saved -> {onnx_path}")

    if use_external_data:
        try:
            mp = onnx.load(str(onnx_path), load_external_data=False)
            data_file = onnx_path.with_suffix(onnx_path.suffix + ".data")
            onnx.external_data_utils.convert_model_to_external_data(
                mp,
                all_tensors_to_one_file=True,
                location=data_file.name,
                size_threshold=0,
            )
            onnx.save_model(mp, str(onnx_path), save_as_external_data=True)
            print(f"[EXPORT] External data -> {onnx_path} (+{data_file.name})")
        except Exception as e:
            print(f"[EXPORT] External data conversion failed: {e}")

    return str(onnx_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Export complete MatchAnything ONNX")
    ap.add_argument("--onnx", default="out/matchanything_complete_export.onnx")
    ap.add_argument("--checkpoint", default="", help="Checkpoint path")
    ap.add_argument("--size", type=int, default=832)
    ap.add_argument("--match_threshold", type=float, default=0.1)
    ap.add_argument("--no_external_data", action="store_true")
    args = ap.parse_args()

    export_complete_matchanything_onnx(
        args.onnx,
        args.checkpoint or None,
        H=args.size,
        W=args.size,
        match_threshold=args.match_threshold,
        use_external_data=not args.no_external_data,
    )
