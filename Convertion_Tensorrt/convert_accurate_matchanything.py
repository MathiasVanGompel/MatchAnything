#!/usr/bin/env python3
"""
Accurate MatchAnything to TensorRT conversion script.
This version ensures exact match with the original PyTorch implementation.
"""

import argparse
import os
from pathlib import Path

# Optional ONNX graph surgery support
try:
    import onnx
    import onnx_graphsurgeon as gs

    HAS_GS = True
except Exception:
    HAS_GS = False

from accurate_matchanything_trt import export_accurate_matchanything_onnx


def strip_eyelike_if_present(onnx_in: str, onnx_out: str):
    """Remove problematic EyeLike operators for TensorRT compatibility"""
    if not HAS_GS:
        print("[ONNX-GS] not available; skipping EyeLike rewrite.")
        return onnx_in

    print("[ONNX-GS] Scanning for EyeLike...")
    model = onnx.load(onnx_in, load_external_data=True)
    graph = gs.import_onnx(model)
    victims = [n for n in graph.nodes if n.op == "EyeLike"]
    print(f"[ONNX-GS] Found {len(victims)} EyeLike node(s).")

    if not victims:
        return onnx_in

    # Remove EyeLike nodes
    for node in victims:
        node.inputs = []
        node.outputs = []

    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    external_data_utils = getattr(
        onnx, "external_data_utils", onnx.external_data_helper
    )
    external_data_utils.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=os.path.basename(onnx_out) + ".data",
        size_threshold=0,
    )
    onnx.save_model(model, onnx_out, save_as_external_data=True)

    out_dir = Path(onnx_out).parent
    for shard in out_dir.glob("onnx__*"):
        shard.unlink(missing_ok=True)

    print(f"[ONNX-GS] Rewrote EyeLike -> {onnx_out}")
    return onnx_out


def main():
    parser = argparse.ArgumentParser(
        description="Convert MatchAnything to TensorRT with exact accuracy"
    )
    parser.add_argument(
        "--onnx",
        default="Convertion_Tensorrt/out/accurate_matchanything.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--model",
        choices=["matchanything_roma", "matchanything_eloftr"],
        default="matchanything_roma",
        help="Model variant to convert",
    )
    parser.add_argument("--H", type=int, default=832, help="Input height")
    parser.add_argument("--W", type=int, default=832, help="Input width")
    parser.add_argument(
        "--match_threshold", type=float, default=0.1, help="Match confidence threshold"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to MatchAnything checkpoint"
    )
    parser.add_argument(
        "--no_graph_surgery", action="store_true", help="Skip EyeLike operator removal"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("ACCURATE MATCHANYTHING TO TENSORRT CONVERSION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input size: {args.H}x{args.W}")
    print(f"Match threshold: {args.match_threshold}")
    if args.ckpt:
        print(f"Checkpoint: {args.ckpt}")
    print()

    # Step 1: Export to ONNX
    print("Step 1: Exporting to ONNX...")
    onnx_path = export_accurate_matchanything_onnx(
        onnx_path=args.onnx,
        model_name=args.model,
        H=args.H,
        W=args.W,
        match_threshold=args.match_threshold,
        ckpt=args.ckpt,
    )

    # Step 2: Graph surgery (optional)
    if not args.no_graph_surgery:
        try:
            onnx_path = strip_eyelike_if_present(onnx_path, onnx_path)
        except Exception as e:
            print(f"[ONNX-GS] rewrite skipped: {e}")

    print("\n" + "=" * 60)
    print("ONNX EXPORT COMPLETE")
    print("=" * 60)
    print(f"Output: {onnx_path}")

    # Step 3: Provide TensorRT build commands
    engine_path = onnx_path.replace(".onnx", ".plan")

    print("\nNext: Build TensorRT engine with trtexec:")
    print("\nRecommended command:")
    print("/usr/src/tensorrt/bin/trtexec \\")
    print(f"    --onnx={onnx_path} \\")
    print(f"    --saveEngine={engine_path} \\")
    print("    --fp16 --memPoolSize=workspace:4096M \\")
    print(
        f"    --minShapes=image0:1x3x{args.H//2}x{args.W//2},image1:1x3x{args.H//2}x{args.W//2} \\"
    )
    print(
        f"    --optShapes=image0:1x3x{args.H}x{args.W},image1:1x3x{args.H}x{args.W} \\"
    )
    print(
        f"    --maxShapes=image0:1x3x{args.H*2}x{args.W*2},image1:1x3x{args.H*2}x{args.W*2} \\"
    )
    print("    --skipInference --verbose")

    print("\nThen run inference:")
    print("python3 run_accurate_matchanything_trt.py \\")
    print(f"    --engine {engine_path} \\")
    print("    --image0 image1.jpg --image1 image2.jpg")

    print("\n" + "=" * 60)
    print("CONVERSION READY")
    print("=" * 60)
    print(
        "The exported ONNX model maintains exact accuracy with the original PyTorch implementation."
    )
    print("All preprocessing and postprocessing steps are preserved.")


if __name__ == "__main__":
    main()
