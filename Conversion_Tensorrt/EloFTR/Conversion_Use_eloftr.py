#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal, reproducible E-LoFTR → TensorRT export pipeline."""

from pathlib import Path

from convert_eloftr_to_trt import (
    build_trt_engine_from_onnx,
    export_eloftr_onnx,
    load_eloftr_model,
)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIR = REPO_ROOT / "Conversion_Tensorrt" / "out"
ONNX_PATH = OUT_DIR / "eloftr_dynamic.onnx"
ENGINE_PATH = OUT_DIR / "eloftr_dynamic_fp16.plan"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load the checkpoint from the vendorised repository.
    model, _cfg = load_eloftr_model(device="cuda")

    # 2) Export ONNX with dynamic batch/height/width dimensions.
    export_eloftr_onnx(model, str(ONNX_PATH), opset=17, sample_hw=(480, 640))

    # 3) Build a TensorRT engine with FP16 enabled.
    build_trt_engine_from_onnx(
        onnx_path=str(ONNX_PATH),
        engine_path=str(ENGINE_PATH),
        min_hw=(240, 320),
        opt_hw=(480, 640),
        max_hw=(640, 960),
        fp16=True,
        workspace_mb=512,
        builder_optimization_level=1,
    )


if __name__ == "__main__":
    main()
