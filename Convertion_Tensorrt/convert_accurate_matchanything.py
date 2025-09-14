#!/usr/bin/env python3
"""Convert MatchAnything model to ONNX with deterministic weight loading."""
import argparse
import os
import re
import warnings
import torch
from pathlib import Path

from accurate_matchanything_trt import AccurateMatchAnythingTRT, export_accurate_matchanything_onnx

def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

def load_ma_roma_ckpt_strict(model, ckpt_path, backbone_key="encoder.dino", ckpt_backbone_prefix=("matcher.model.encoder.dino", "encoder.dino")):
    """Load MatchAnything checkpoint strictly for non-backbone parts."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    stripped = {}
    for p in ckpt_backbone_prefix:
        stripped.update(_strip_prefix(state, p + "."))

    target_sd = model.state_dict()
    to_load = {}
    missing_dbg, unexpected_dbg = [], []

    for k, v in state.items():
        kk = k
        if kk.startswith("matcher.model."):
            kk = kk[len("matcher.model."):]
        if kk.startswith(backbone_key + "."):
            continue
        if kk in target_sd and target_sd[kk].shape == v.shape:
            to_load[kk] = v
        else:
            if kk not in target_sd:
                unexpected_dbg.append(kk)
            else:
                missing_dbg.append((kk, v.shape, target_sd[kk].shape))

    msg = model.load_state_dict(to_load, strict=False)
    if msg.missing_keys:
        warnings.warn(f"[CKPT non-backbone] Missing keys: {len(msg.missing_keys)} -> e.g. {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        warnings.warn(f"[CKPT non-backbone] Unexpected keys: {len(msg.unexpected_keys)} -> e.g. {msg.unexpected_keys[:5]}")
    if missing_dbg:
        warnings.warn(f"[CKPT non-backbone] Shape mismatches: {len(missing_dbg)}")

def load_official_dinov2_backbone(model, backbone_key="encoder.dino", use_timm_first=True):
    """Fill the DINOv2 backbone with canonical weights."""
    off = None
    if use_timm_first:
        try:
            import timm
            m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, features_only=False)
            off = m.state_dict()
        except Exception:
            off = None
    if off is None:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        off = dinov2.state_dict()

    tgt_sd = model.state_dict()
    mapped = {}
    ok, skip = 0, 0
    for k, v in off.items():
        k2 = f"{backbone_key}.{k}"
        if k2 in tgt_sd and tgt_sd[k2].shape == v.shape:
            mapped[k2] = v
            ok += 1
        else:
            skip += 1

    msg = model.load_state_dict(mapped, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        warnings.warn(f"[DINOv2 backbone] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    print(f"[DINOv2 backbone] loaded {ok} tensors, skipped {skip} (shape/name mismatch).")
    return ok

def main():
    ap = argparse.ArgumentParser(description="Convert MatchAnything to ONNX")
    ap.add_argument("--onnx", default="Convertion_Tensorrt/out/accurate_matchanything.onnx", help="Output ONNX path")
    ap.add_argument("--H", type=int, default=840)
    ap.add_argument("--W", type=int, default=840)
    ap.add_argument("--ckpt", type=str, default=None, help="Path to MatchAnything checkpoint")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.onnx) or ".", exist_ok=True)

    model = AccurateMatchAnythingTRT()
    if args.ckpt:
        load_ma_roma_ckpt_strict(model, args.ckpt, backbone_key="encoder.dino")
    loaded = load_official_dinov2_backbone(model, backbone_key="encoder.dino")
    assert loaded > 0, "Failed to load any DINOv2 backbone weights"

    onnx_path = export_accurate_matchanything_onnx(model, args.onnx, H=args.H, W=args.W)
    print(f"[ONNX] Exported accurate model -> {onnx_path}")

    engine_path = args.onnx.replace(".onnx", ".plan")
    print("\nNext: build TensorRT engine with trtexec:")
    print("/usr/src/tensorrt/bin/trtexec \\")
    print(f"  --onnx={onnx_path} \\")
    print(f"  --saveEngine={engine_path} \\")
    print("  --fp16 --memPoolSize=workspace:4096M \\")
    print(f"  --minShapes=image0:1x3x{args.H//2}x{args.W//2},image1:1x3x{args.H//2}x{args.W//2} \\")
    print(f"  --optShapes=image0:1x3x{args.H}x{args.W},image1:1x3x{args.H}x{args.W} \\")
    print(f"  --maxShapes=image0:1x3x{args.H*2}x{args.W*2},image1:1x3x{args.H*2}x{args.W*2} \\")
    print("  --skipInference --verbose")

if __name__ == "__main__":
    main()

