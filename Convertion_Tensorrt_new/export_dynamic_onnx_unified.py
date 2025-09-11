#!/usr/bin/env python3
"""
Export dynamic ONNX for MatchAnything with unified weight loading & full stats.

Outputs (dense, no filtering):
  - warp_c        [B,Ha,Wa,2]
  - cert_c        [B,Ha,Wa]
  - valid_mask    [B,Ha,Wa]
  - coarse_stride [1]  (float)

Dynamic axes for B,H,W,Ha,Wa. Compatible with TensorRT dynamic profiles.
"""

import os, re, inspect, sys, tempfile
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import onnx

# Optional timm (for DINOv2 backfill). If missing, we continue.
try:
    import timm
except Exception:
    timm = None

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_THIS_DIR))

from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT

# ---------------------------------------------------------------------
# Low-level (C/C++) stdout/stderr silencer (captures libtorch/ONNX prints)
# ---------------------------------------------------------------------
@contextmanager
def silence_stdio_c(stdout=True, stderr=True):
    """
    Temporarily redirect OS-level file descriptors for stdout/stderr to a
    temporary file so C/C++ prints (libtorch/JIT/ONNX) are hidden.

    NOTE: Python-level redirect_* only captures Python writes; C++ writes need
    fd redirection (os.dup2) as described by pybind11 & others.
    """
    fds = []
    backups = []
    tmp = None
    try:
        if not (stdout or stderr):
            yield
            return

        # Flush Python streams so we don't lose ordering
        sys.stdout.flush()
        sys.stderr.flush()

        tmp = tempfile.TemporaryFile(mode="w+b")
        if stdout:
            backups.append((1, os.dup(1)))
            fds.append(1)
        if stderr:
            backups.append((2, os.dup(2)))
            fds.append(2)

        for fd in fds:
            os.dup2(tmp.fileno(), fd)

        yield
    finally:
        try:
            # Restore original fds
            for fd, bk in backups:
                os.dup2(bk, fd)
                os.close(bk)
        finally:
            if tmp is not None:
                tmp.close()

# -----------------------------
# Unified weight loading
# -----------------------------
def create_comprehensive_mapping_rules() -> List[Tuple[re.Pattern, str]]:
    return [
        (re.compile(r"^module\\."), ""),
        (re.compile(r"^_orig_mod\\."), ""),
        (re.compile(r"^matcher\\.model\\.decoder\\.embedding_decoder\\."), "encoder.dino."),
        (re.compile(r"^embedding_decoder\\."), "encoder.dino."),
        (re.compile(r"^decoder\\.embedding_decoder\\."), "encoder.dino."),
        (re.compile(r"^matcher\\.model\\.encoder\\.cnn\\.layers\\."), "encoder.layers."),
        (re.compile(r"^matcher\\.model\\.encoder\\.cnn\\."), "encoder."),
        (re.compile(r"^matcher\\.model\\.encoder\\."), "encoder."),
        (re.compile(r"^matcher\\.model\\.decoder\\."), "matcher."),
        (re.compile(r"^matcher\\.model\\."), ""),
        (re.compile(r"^matcher\\."), ""),
        (re.compile(r"^model\\."), ""),
        (re.compile(r"^backbone\\."), "encoder.dino."),
        (re.compile(r"^vit\\."), "encoder.dino."),
        (re.compile(r"^dino\\."), "encoder.dino."),
        (re.compile(r"^encoder\\.vit\\."), "encoder.dino."),
        (re.compile(r"^encoder\\.backbone\\."), "encoder.dino."),
        (re.compile(r"^encoder\\.dino\\."), "encoder.dino."),
        (re.compile(r"^encoder\\."), "encoder."),
    ]

def fix_dinov2_block_structure(weights_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert 'encoder.dino.blocks.N.*' -> 'encoder.dino.blocks.0.N.*' for BlockChunk."""
    fixed = {}
    for k, v in weights_dict.items():
        if k.startswith("encoder.dino.blocks."):
            parts = k.split(".")
            if len(parts) >= 4 and parts[3].isdigit():
                blk = parts[3]
                if len(parts) >= 5 and parts[4].isdigit():
                    fixed[k] = v
                else:
                    new_key = ".".join(parts[:3] + ["0", blk] + parts[4:])
                    fixed[new_key] = v
                    print(f"[LOAD] BlockChunk fix: {k} -> {new_key}")
            else:
                fixed[k] = v
        else:
            fixed[k] = v
    return fixed

def resize_positional_embedding(pos_embed: torch.Tensor, target_size: int) -> torch.Tensor:
    if pos_embed.shape[1] == target_size:
        print(f"[LOAD] pos_embed already {tuple(pos_embed.shape)}")
        return pos_embed
    print(f"[LOAD] Resize pos_embed {tuple(pos_embed.shape)} -> [1,{target_size},{pos_embed.shape[2]}]")
    cls = pos_embed[:, 0:1, :]
    spatial = pos_embed[:, 1:, :]
    nsp = spatial.shape[1]
    og = int(np.sqrt(nsp))
    assert og * og == nsp, f"pos_embed spatial tokens not square: {nsp}"
    tg_spatial = target_size - 1
    tg = int(np.sqrt(tg_spatial))
    assert tg * tg == tg_spatial, f"target size {target_size} not square+1"
    D = spatial.shape[2]
    spatial_2d = spatial.reshape(1, og, og, D).permute(0, 3, 1, 2)  # [1,D,H,W]
    resized = F.interpolate(spatial_2d, size=(tg, tg), mode="bilinear", align_corners=False)
    resized = resized.permute(0, 2, 3, 1).reshape(1, tg_spatial, D)
    out = torch.cat([cls, resized], dim=1)
    return out

def load_dinov2_components(model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Optional fill-ins from official DINOv2 timm weights."""
    out = {}
    if timm is None:
        print("[LOAD] timm not available; skipping DINOv2 backfill")
        return out
    try:
        print("[LOAD] Loading DINOv2 (timm) for missing components...")
        dinom = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
        official = dinom.state_dict()

        if "encoder.dino.pos_embed" in model_state and "pos_embed" in official:
            pos = official["pos_embed"]
            tgt = model_state["encoder.dino.pos_embed"]
            if pos.shape != tgt.shape:
                pos = resize_positional_embedding(pos, tgt.shape[1])
            out["encoder.dino.pos_embed"] = pos
            print(f"[LOAD] +pos_embed {tuple(pos.shape)}")

        # If the ckpt didn't have these at all, synthesize or copy from timm
        if "encoder.dino.cls_token" in model_state:
            out["encoder.dino.cls_token"] = official.get("cls_token", torch.zeros_like(model_state["encoder.dino.cls_token"]))
            print(f"[LOAD] +cls_token {tuple(out['encoder.dino.cls_token'].shape)}")
        if "encoder.dino.mask_token" in model_state:
            out["encoder.dino.mask_token"] = official.get("mask_token", torch.zeros_like(model_state["encoder.dino.mask_token"]))
            src = "timm" if "mask_token" in official else "synthesized zeros"
            print(f"[LOAD] +mask_token ({src}) {tuple(out['encoder.dino.mask_token'].shape)}")

        need_patch = any(k.startswith("encoder.dino.patch_embed.") for k in model_state.keys())
        if need_patch:
            for k, v in official.items():
                if k.startswith("patch_embed."):
                    nk = f"encoder.dino.{k}"
                    if nk in model_state:
                        out[nk] = v
                        print(f"[LOAD] +{nk} {tuple(v.shape)}")

        # If final LayerNorm params are totally missing, mirror from official or zeros
        if "encoder.dino.norm.weight" in model_state and "encoder.dino.norm.weight" not in out:
            out["encoder.dino.norm.weight"] = official.get("norm.weight", torch.ones_like(model_state["encoder.dino.norm.weight"]))
            print(f"[LOAD] +encoder.dino.norm.weight {tuple(out['encoder.dino.norm.weight'].shape)}")
        if "encoder.dino.norm.bias" in model_state and "encoder.dino.norm.bias" not in out:
            out["encoder.dino.norm.bias"] = official.get("norm.bias", torch.zeros_like(model_state["encoder.dino.norm.bias"]))
            print(f"[LOAD] +encoder.dino.norm.bias {tuple(out['encoder.dino.norm.bias'].shape)}")

    except Exception as e:
        print(f"[LOAD] DINOv2 backfill failed: {e}")
    return out

def load_missing_dinov2_blocks(model_state: Dict[str, torch.Tensor], existing: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fill missing transformer blocks from official DINOv2 (wrap-around if needed)."""
    out = {}
    if timm is None:
        print("[LOAD] timm not available; skipping missing-block fill")
        return out
    try:
        print("[LOAD] Filling missing DINOv2 blocks from timm...")
        dinom = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
        official = dinom.state_dict()

        need, have = set(), set()
        for k in model_state.keys():
            if k.startswith("encoder.dino.blocks.0.") and len(k.split(".")) >= 5:
                blk = k.split(".")[4]
                if blk.isdigit():
                    need.add(int(blk))
        for k in existing.keys():
            if k.startswith("encoder.dino.blocks.0.") and len(k.split(".")) >= 5:
                blk = k.split(".")[4]
                if blk.isdigit():
                    have.add(int(blk))
        miss = need - have
        print(f"[LOAD] Need blocks: {sorted(need)}")
        print(f"[LOAD] Have blocks: {sorted(have)}")
        print(f"[LOAD] Missing:     {sorted(miss)}")

        for mb in sorted(miss):
            src = mb % 24  # wrap if needed
            for ok, ov in official.items():
                if ok.startswith(f"blocks.{src}."):
                    comps = ok.split(".")[2:]  # drop 'blocks.N'
                    nk = f"encoder.dino.blocks.0.{mb}." + ".".join(comps)
                    if nk in model_state:
                        out[nk] = ov
        print(f"[LOAD] Added {len(out)} missing block params")
    except Exception as e:
        print(f"[LOAD] Missing-block fill failed: {e}")
    return out

def apply_unified_weight_loading(checkpoint_path: str, model_state: Dict[str, torch.Tensor], use_dinov2_backfill: bool = True) -> Dict[str, torch.Tensor]:
    print(f"[LOAD] Reading checkpoint: {checkpoint_path}")
    try:
        raw = torch.load(checkpoint_path, map_location="cpu")
        ckpt_state = raw.get("state_dict", raw)
    except Exception as e:
        print(f"[LOAD] ERROR loading checkpoint: {e}")
        return {}

    print(f"[LOAD] CKPT keys: {len(ckpt_state)}")
    print(f"[LOAD] Model expects keys: {len(model_state)}")

    # Step 1: map keys
    rules = create_comprehensive_mapping_rules()
    mapped = {}
    for k, v in ckpt_state.items():
        nk = k
        for pat, rep in rules:
            nk = pat.sub(rep, nk)
        mapped[nk] = v
    print(f"[LOAD] After mapping: {len(mapped)}")

    # Step 2: DINO BlockChunk structure
    mapped = fix_dinov2_block_structure(mapped)

    # Step 3: optional backfill from official DINOv2
    if use_dinov2_backfill:
        back = load_dinov2_components(model_state)
        mapped.update(back)

    # Step 4: direct matches
    loadable: Dict[str, torch.Tensor] = {}
    for mk, mv in model_state.items():
        if mk in mapped and mapped[mk].shape == mv.shape:
            loadable[mk] = mapped[mk]
    print(f"[LOAD] Direct matches: {len(loadable)}")

    # Step 5: fill missing blocks
    if use_dinov2_backfill:
        add_blocks = load_missing_dinov2_blocks(model_state, loadable)
        loadable.update(add_blocks)
        print(f"[LOAD] After missing-block fill: {len(loadable)}")

    # Step 6: final backfill for qkv.bias (paramwise copy by block index)
    if use_dinov2_backfill and timm is not None:
        try:
            dinom = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
            official = dinom.state_dict()
            for b in range(24):
                mk = f"encoder.dino.blocks.0.{b}.attn.qkv.bias"
                ok = f"blocks.{b}.attn.qkv.bias"
                if mk in model_state and mk not in loadable and ok in official and official[ok].shape == model_state[mk].shape:
                    loadable[mk] = official[ok]
                    print(f"[LOAD] +paramwise {mk}  (from {ok})")
            print(f"[LOAD] After paramwise fill: {len(loadable)}")
        except Exception as e:
            print(f"[LOAD] Paramwise fill failed: {e}")

    # Report
    loaded_cnt = len(loadable)
    total_cnt  = len(model_state)
    pct = 100.0 * loaded_cnt / max(1, total_cnt)
    print("[LOAD] === SUMMARY ===")
    print(f"[LOAD] Loaded tensors: {loaded_cnt} / {total_cnt}  ({pct:.1f}%)")

    # DINO block coverage
    blocks_loaded = set()
    for k in loadable.keys():
        if k.startswith("encoder.dino.blocks.0.") and len(k.split(".")) >= 5:
            b = k.split(".")[4]
            if b.isdigit():
                blocks_loaded.add(int(b))
    print(f"[LOAD] DINOv2 blocks loaded: {len(blocks_loaded)}/24")

    if loaded_cnt >= 0.95 * total_cnt:
        print("[LOAD] ✅ SUCCESS: >95% loaded")
    elif loaded_cnt >= 0.8 * total_cnt:
        print("[LOAD] ⚠️  WARNING: 80–95% loaded")
    else:
        print("[LOAD] ❌ ERROR: <80% loaded")

    return loadable

# -----------------------------
# Export
# -----------------------------
def export_onnx(onnx_path: str, ckpt: str = "", H: int = 768, W: int = 1024,
                use_dinov2_backfill: bool = True, verbose_export: bool = False,
                silence_cpp: bool = True):
    # Keep logs quieter unless asked
    os.environ.pop("PYTORCH_ONNX_VERBOSE", None)
    os.environ.pop("PYTORCH_JIT_LOG_LEVEL", None)
    os.environ.pop("TORCH_LOGS", None)

    device = "cpu"
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()

    # Load weights
    if ckpt and os.path.exists(ckpt):
        ms = model.state_dict()
        loadable = apply_unified_weight_loading(ckpt, ms, use_dinov2_backfill)
        missing, unexpected = model.load_state_dict(loadable, strict=False)
        print(f"[LOAD] load_state_dict -> loaded={len(loadable)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("[LOAD] --- Missing keys ---")
            for k in sorted(missing):
                print("   ", k)
        if unexpected:
            print("[LOAD] --- Unexpected keys ---")
            for k in sorted(unexpected):
                print("   ", k)
    else:
        print("[LOAD] No checkpoint provided or file missing; continuing with random init (low scores).")

    # Dummy input
    x0 = torch.rand(1, 3, H, W, device=device)
    x1 = torch.rand(1, 3, H, W, device=device)

    # Dry run
    with torch.inference_mode():
        out = model(x0, x1)
        print("Dry run:",
              "warp_c", tuple(out["warp_c"].shape),
              "cert_c", tuple(out["cert_c"].shape),
              "valid_mask", tuple(out["valid_mask"].shape),
              "coarse_stride", tuple(out["coarse_stride"].shape))

    out_path = Path(onnx_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "image0": {"0": "B", "2": "H", "3": "W"},
        "image1": {"0": "B", "2": "H", "3": "W"},
        "warp_c": {"0": "B", "1": "Ha", "2": "Wa"},
        "cert_c": {"0": "B", "1": "Ha", "2": "Wa"},
        "valid_mask": {"0": "B", "1": "Ha", "2": "Wa"},
        "coarse_stride": {"0": "one"},
    }

    kwargs = dict(
        input_names=["image0", "image1"],
        output_names=["warp_c", "cert_c", "valid_mask", "coarse_stride"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        verbose=bool(verbose_export),
        training=torch.onnx.TrainingMode.EVAL,
    )
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
        kwargs["use_external_data_format"] = True  # safer for very large models

    # Export with both Python-level and OS-level silencing
    try:
        with redirect_stdout(open(os.devnull, "w")), redirect_stderr(open(os.devnull, "w")):
            ctx = silence_stdio_c(stdout=True, stderr=True) if silence_cpp else contextmanager(lambda: (yield))()
            with ctx:
                torch.onnx.export(model, (x0, x1), str(out_path), **kwargs)
    except Exception as e:
        # Make sure the error is visible even if we were silencing
        sys.stdout.flush(); sys.stderr.flush()
        print(f"[ONNX] EXPORT FAILED: {e}")
        raise

    # Verify file exists and is readable
    if not out_path.exists():
        raise FileNotFoundError(f"[ONNX] Expected file not found: {out_path}")

    size_mb = out_path.stat().st_size / (1024 * 1024.0)
    print(f"[ONNX] Saved -> {out_path}  ({size_mb:.1f} MB)")

    # Load with onnx to sanity-check
    try:
        mp = onnx.load(str(out_path), load_external_data=True)
        print(f"[ONNX] Model loaded OK: ir_version={mp.ir_version}, opsets={[ (o.domain,o.version) for o in mp.opset_import ]}")
    except Exception as e:
        print(f"[ONNX] WARNING: onnx.load failed: {e}")

    return str(out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export dynamic ONNX with unified weight loading")
    ap.add_argument("--onnx", default="out/matchanything_dense_dynamic.onnx")
    ap.add_argument("--ckpt", default="/home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt")
    ap.add_argument("--H", type=int, default=768)
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--no_dinov2_backfill", action="store_true", help="Disable DINOv2 timm backfill")
    ap.add_argument("--verbose_export", action="store_true", help="ONNX exporter verbose graph print")
    ap.add_argument("--no_silence_cpp", action="store_true", help="Do not silence C++ stdout/stderr during export")
    args = ap.parse_args()

    export_onnx(
        onnx_path=args.onnx,
        ckpt=args.ckpt,
        H=args.H,
        W=args.W,
        use_dinov2_backfill=not args.no_dinov2_backfill,
        verbose_export=args.verbose_export,
        silence_cpp=not args.no_silence_cpp,
    )
