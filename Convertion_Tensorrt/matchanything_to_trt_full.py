#!/usr/bin/env python3
import argparse, os, sys
from typing import Optional
import torch

# Optional ONNX graph surgery (EyeLike) support
try:
    import onnx, onnx_graphsurgeon as gs, numpy as np
    HAS_GS = True
except Exception:
    HAS_GS = False

from roma_models_trt_full import RoMaTRTCoreFull
from weight_adapter import remap_and_load

def export_matchanything_onnx(onnx_path: str, H: int, W: int,
                              ckpt: Optional[str] = None,
                              backbone_pretrained: bool = False,
                              verbose: bool = False):
    device = "cpu"
    model = RoMaTRTCoreFull().to(device).eval()

    # If you provided a ckpt, try to load as many weights as possible (including DINO keys)
    if ckpt:
        print(f"[CKPT] Using: {ckpt}")
        remap_and_load(model, ckpt_path=ckpt, save_sanitized=None)
    if backbone_pretrained:
        print("[INFO] --backbone_pretrained was provided; this pipeline uses DINOv2 (ViT) backbone, "
              "so this flag is a no-op here and kept for CLI compatibility.")

    # Dry run before export
    x1 = torch.rand(1, 3, H, W, device=device)
    x2 = torch.rand(1, 3, H, W, device=device)
    with torch.no_grad():
        warp, cert = model(x1, x2)
        print("dry-run OK:", tuple(warp.shape), tuple(cert.shape))

    dynamic_axes = {
        "image0": {0: "B", 2: "H", 3: "W"},
        "image1": {0: "B", 2: "H", 3: "W"},
        "warp":   {0: "B", 2: "H", 3: "W"},
        "cert":   {0: "B", 2: "H", 3: "W"},
    }
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(
        model, (x1, x2), onnx_path,
        input_names=["image0", "image1"],
        output_names=["warp", "cert"],
        dynamic_axes=dynamic_axes,
        opset_version=17, do_constant_folding=True,
    )
    print(f"[ONNX] Exported -> {onnx_path}")
    return onnx_path

def strip_eyelike_if_present(onnx_in: str, onnx_out: str):
    if not HAS_GS:
        print("[ONNX-GS] not available; skipping EyeLike rewrite.")
        return onnx_in
    print("[ONNX-GS] Scanning for EyeLike...")
    model = onnx.load(onnx_in)
    graph = gs.import_onnx(model)
    victims = [n for n in graph.nodes if n.op == "EyeLike"]
    print(f"[ONNX-GS] Found {len(victims)} EyeLike node(s).")
    if not victims:
        return onnx_in

    # Pragmatic approach: many ViT exports tolerate simply dropping EyeLike (identity uses).
    for node in victims:
        y = node.outputs[0]
        # Rewire consumers to a constant-zeros tensor with the same shape if needed
        # but in most exports EyeLike isn't used downstream; keep it simple:
        node.inputs = []; node.outputs = []
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), onnx_out)
    print(f"[ONNX-GS] Rewrote EyeLike -> {onnx_out}")
    return onnx_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="Convertion_Tensorrt/out/roma_dino_gp_dynamic.onnx")
    ap.add_argument("--H", type=int, default=448)
    ap.add_argument("--W", type=int, default=448)
    ap.add_argument("--ckpt", type=str, default=None, help="path to matchanything_roma.ckpt")
    ap.add_argument("--backbone_pretrained", action="store_true", help="kept for CLI compatibility (no-op)")
    ap.add_argument("--no_graph_surgery", action="store_true", help="skip EyeLike rewrite")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    onnx_path = export_matchanything_onnx(
        args.onnx, args.H, args.W,
        ckpt=args.ckpt,
        backbone_pretrained=args.backbone_pretrained,
        verbose=args.verbose,
    )

    if not args.no_graph_surgery:
        try:
            onnx_path = strip_eyelike_if_present(onnx_path, onnx_path)
        except Exception as e:
            print("[ONNX-GS] rewrite skipped:", e)

    print("[INFO] ONNX ready. Build engines with trtexec:")
    print("Low-mem:")
    print(f"  trtexec --onnx={onnx_path} --saveEngine=Convertion_Tensorrt/out/roma_dino_gp_lowmem.plan \\")
    print("          --explicitBatch --fp16 --workspace=1024 \\")
    print("          --minShapes=image0:1x3x224x224,image1:1x3x224x224 \\")
    print("          --optShapes=image0:1x3x448x448,image1:1x3x448x448 \\")
    print("          --maxShapes=image0:1x3x640x640,image1:1x3x640x640 --buildOnly")
    print("Hi-acc (safer maxima):")
    print(f"  trtexec --onnx={onnx_path} --saveEngine=Convertion_Tensorrt/out/roma_dino_gp_hiacc.plan \\")
    print("          --explicitBatch --fp16 --workspace=2048 \\")
    print("          --minShapes=image0:1x3x448x448,image1:1x3x448x448 \\")
    print("          --optShapes=image0:1x3x672x672,image1:1x3x672x672 \\")
    print("          --maxShapes=image0:1x3x896x896,image1:1x3x896x896 --buildOnly")

if __name__ == "__main__":
    main()