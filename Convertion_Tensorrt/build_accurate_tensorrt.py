#!/usr/bin/env python3
"""Export MatchAnything to ONNX and build a TensorRT engine.

This helper mirrors the behaviour of ``build_accurate_tensorrt.sh`` but is
implemented in Python so it can be executed on systems without a POSIX shell.

By default it uses the repository's bundled ``matchanything_roma.ckpt``
checkpoint. The script writes the ONNX model and its consolidated weight file to
``Convertion_Tensorrt/out`` and immediately invokes ``trtexec`` to generate a
TensorRT engine.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from accurate_matchanything_trt import export_accurate_matchanything_onnx


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_ckpt = (
        repo_root
        / "imcui"
        / "third_party"
        / "MatchAnything"
        / "weights"
        / "matchanything_roma.ckpt"
    )

    parser = argparse.ArgumentParser(
        description="Export MatchAnything to ONNX and build a TensorRT engine",
    )
    parser.add_argument(
        "--ckpt",
        default=str(default_ckpt),
        help="Path to matchanything_roma.ckpt",
    )
    parser.add_argument("--height", type=int, default=832, help="Input height")
    parser.add_argument("--width", type=int, default=832, help="Input width")
    parser.add_argument(
        "--workspace",
        type=int,
        default=4096,
        help="TensorRT workspace size in MB",
    )
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "accurate_matchanything_roma.onnx"
    engine_path = out_dir / "accurate_matchanything_roma.plan"

    export_accurate_matchanything_onnx(
        str(onnx_path),
        model_name="matchanything_roma",
        H=args.height,
        W=args.width,
        ckpt=args.ckpt,
    )

    trt_cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace}M",
        f"--minShapes=image0:1x3x{args.height//2}x{args.width//2},image1:1x3x{args.height//2}x{args.width//2}",
        f"--optShapes=image0:1x3x{args.height}x{args.width},image1:1x3x{args.height}x{args.width}",
        f"--maxShapes=image0:1x3x{args.height*2}x{args.width*2},image1:1x3x{args.height*2}x{args.width*2}",
        "--skipInference",
        "--verbose",
    ]
    if not args.no_fp16:
        trt_cmd.append("--fp16")

    subprocess.run(trt_cmd, check=True)

    print(f"✅ TensorRT engine written to {engine_path}")
    print(f"   (Keep {onnx_path} and {onnx_path}.data together)")


if __name__ == "__main__":
    main()
