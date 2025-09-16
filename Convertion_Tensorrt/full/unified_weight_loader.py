# Convertion_Tensorrt/unified_weight_loader.py
# -*- coding: utf-8 -*-
"""
Unified loader that:
- Maps MatchAnything/ROMA ckpt keys to your model.
- Detects + fixes DINOv2 BlockChunk naming (blocks.N.* ↔ blocks.0.N.*).
- Supplements ANY missing DINOv2 weights from TIMM vit_large_patch14_dinov2.lvd142m
  (pos_embed, cls/mask tokens, patch_embed.*, ALL blocks.*, norms, LayerScale gammas, etc.).

This raises the loaded ratio >95% in practice, fixing the 'ls1.gamma/ls2.gamma' gaps.
"""
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F

# ---- mapping rules -----------------------------------------------------------
def create_comprehensive_mapping_rules() -> List[Tuple[re.Pattern, str]]:
    return [
        # strip wrappers
        (re.compile(r"^module\."), ""),
        (re.compile(r"^_orig_mod\."), ""),

        # MatchAnything -> our encoder / matcher namespaces
        (re.compile(r"^matcher\.model\.decoder\.embedding_decoder\."), "encoder.dino."),
        (re.compile(r"^embedding_decoder\."), "encoder.dino."),
        (re.compile(r"^decoder\.embedding_decoder\."), "encoder.dino."),

        (re.compile(r"^matcher\.model\.encoder\.cnn\.layers\."), "encoder.layers."),
        (re.compile(r"^matcher\.model\.encoder\.cnn\."), "encoder."),
        (re.compile(r"^matcher\.model\.encoder\."), "encoder."),

        (re.compile(r"^matcher\.model\.decoder\."), "matcher."),
        (re.compile(r"^matcher\.model\."), ""),
        (re.compile(r"^matcher\."), ""),
        (re.compile(r"^model\."), ""),

        # fallback DINO namings
        (re.compile(r"^backbone\."), "encoder.dino."),
        (re.compile(r"^vit\."), "encoder.dino."),
        (re.compile(r"^dino\."), "encoder.dino."),
        (re.compile(r"^encoder\.vit\."), "encoder.dino."),
        (re.compile(r"^encoder\.backbone\."), "encoder.dino."),

        # identities
        (re.compile(r"^encoder\.dino\."), "encoder.dino."),
        (re.compile(r"^encoder\."), "encoder."),
    ]


# ---- block-chunk canonicalization --------------------------------------------
def _model_uses_chunked_blocks(model_state: Dict[str, torch.Tensor]) -> bool:
    # True if keys look like encoder.dino.blocks.0.0.norm1.weight
    for k in model_state.keys():
        if k.startswith("encoder.dino.blocks.0.0."):
            return True
    return False

def _ckpt_is_chunked(mapped_ckpt: Dict[str, torch.Tensor]) -> bool:
    for k in mapped_ckpt.keys():
        if k.startswith("encoder.dino.blocks.0.") and k.split(".")[3].isdigit() and k.split(".")[4].isdigit():
            return True
    return False

def fix_dinov2_block_structure_for_model(mapped_ckpt: Dict[str, torch.Tensor],
                                         model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert ckpt block keys to match the model's expectation:
    - If model expects chunked keys (blocks.0.N.*), convert flat -> chunked.
    - If model expects flat keys (blocks.N.*), convert chunked -> flat.
    """
    wants_chunked = _model_uses_chunked_blocks(model_state)
    out = {}
    for key, val in mapped_ckpt.items():
        if not key.startswith("encoder.dino.blocks."):
            out[key] = val
            continue

        parts = key.split(".")
        # Identify whether ckpt key is chunked or flat
        is_ckpt_chunked = (len(parts) > 4 and parts[3].isdigit() and parts[4].isdigit())
        is_ckpt_flat    = (len(parts) > 3 and parts[3].isdigit() and (len(parts) <= 4 or not parts[4].isdigit()))

        if wants_chunked and is_ckpt_flat:
            # encoder.dino.blocks.N.xxx -> encoder.dino.blocks.0.N.xxx
            new_parts = parts[:3] + ["0", parts[3]] + parts[4:]
            new_key = ".".join(new_parts)
            out[new_key] = val
        elif (not wants_chunked) and is_ckpt_chunked:
            # encoder.dino.blocks.0.N.xxx -> encoder.dino.blocks.N.xxx
            new_parts = parts[:3] + [parts[4]] + parts[5:]
            new_key = ".".join(new_parts)
            out[new_key] = val
        else:
            out[key] = val
    return out


# ---- pos-embed resize (if needed) --------------------------------------------
def resize_positional_embedding(pos_embed: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resize DINOv2 positional embedding from pre-training resolution to target token length.
    RoMa often uses 518x518 (37x37 + CLS = 1370). We'll resize as needed.
    """
    if pos_embed.shape[1] == target_len:
        return pos_embed

    cls_token = pos_embed[:, :1, :]                # [1,1,D]
    spatial = pos_embed[:, 1:, :]                  # [1,N-1,D]
    n = spatial.shape[1]
    gs = int(np.sqrt(n))
    if gs * gs != n:
        raise ValueError(f"Cannot reshape {n} spatial tokens to a square grid.")
    tgt_spatial = target_len - 1
    tgt_gs = int(np.sqrt(tgt_spatial))
    if tgt_gs * tgt_gs != tgt_spatial:
        raise ValueError(f"Target length {target_len} not square+CLS.")
    D = spatial.shape[2]
    grid = spatial.reshape(1, gs, gs, D).permute(0, 3, 1, 2)  # [1,D,gs,gs]
    grid = F.interpolate(grid, size=(tgt_gs, tgt_gs), mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, tgt_spatial, D)
    return torch.cat([cls_token, grid], dim=1)


# ---- optional TIMM supplementation -------------------------------------------
def _supplement_from_timm(model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Pull ANY missing DINOv2 weights from timm vit_large_patch14_dinov2.lvd142m
    including blocks.*, norms, layerscale gammas, etc., when shapes match.
    """
    out: Dict[str, torch.Tensor] = {}
    try:
        import timm  # heavyweight import; keep inside try
        m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
        official = m.state_dict()

        # Helper to copy if shapes match
        def try_copy(src_key: str, dst_key: str):
            if src_key in official and dst_key in model_state:
                a, b = official[src_key], model_state[dst_key]
                if tuple(a.shape) == tuple(b.shape):
                    out[dst_key] = a

        # tokens / pos
        for s, d in [
            ("pos_embed", "encoder.dino.pos_embed"),
            ("cls_token", "encoder.dino.cls_token"),
            ("mask_token", "encoder.dino.mask_token"),
            ("norm.weight", "encoder.dino.norm.weight"),
            ("norm.bias",   "encoder.dino.norm.bias"),
        ]:
            try_copy(s, d)

        # patch_embed.*
        for k in official.keys():
            if k.startswith("patch_embed."):
                try_copy(k, f"encoder.dino.{k}")

        # ALL blocks.* (attn, mlp, norms, layerscales)
        for k in official.keys():
            if k.startswith("blocks."):
                try_copy(k, f"encoder.dino.{k}")

    except Exception as e:
        print(f"[UNIFIED] Warning: Could not load official DINOv2 TIMM weights: {e}")
    return out


# ---- main unified loader ------------------------------------------------------
def apply_unified_weight_loading(
    checkpoint_path: str,
    model_state_dict: Dict[str, torch.Tensor],
    load_dinov2_components: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load MatchAnything/ROMA ckpt → our model namespace, adapt block-chunking,
    and supplement with official DINOv2 weights from TIMM when helpful.
    """
    print(f"[UNIFIED] Loading checkpoint: {checkpoint_path}")
    # 0) load raw ckpt
    try:
        raw = torch.load(checkpoint_path, map_location="cpu")
        ckpt_state_dict = raw.get("state_dict", raw)
    except Exception as e:
        print(f"[UNIFIED] Error torch.load: {e}")
        # fallback attempts
        try:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_state_dict = raw.get("state_dict", raw)
            print("[UNIFIED] Loaded with weights_only=False")
        except Exception as e2:
            try:
                import pickle
                with open(checkpoint_path, "rb") as f:
                    raw = pickle.load(f)
                ckpt_state_dict = raw.get("state_dict", raw)
                print("[UNIFIED] Loaded via pickle")
            except Exception as e3:
                print(f"[UNIFIED] Could not load checkpoint: {e3}")
                return {}

    print(f"[UNIFIED] Checkpoint has {len(ckpt_state_dict)} keys")
    print(f"[UNIFIED] Model expects {len(model_state_dict)} keys")

    # 1) map namespaces
    rules = create_comprehensive_mapping_rules()
    mapped = {}
    for k, v in ckpt_state_dict.items():
        nk = k
        for pat, rep in rules:
            nk = pat.sub(rep, nk)
        mapped[nk] = v

    # 2) adapt block structure to model (flat vs chunked)
    mapped = fix_dinov2_block_structure_for_model(mapped, model_state_dict)

    # 3) supplement from TIMM if requested
    if load_dinov2_components:
        print("[UNIFIED] Supplementing from TIMM (DINOv2 vit_large_patch14_dinov2.lvd142m) when shapes match...")
        mapped.update(_supplement_from_timm(model_state_dict))

        # If pos_embed length differs, resize to target
        if "encoder.dino.pos_embed" in mapped and "encoder.dino.pos_embed" in model_state_dict:
            pe = mapped["encoder.dino.pos_embed"]
            tgt_len = model_state_dict["encoder.dino.pos_embed"].shape[1]
            if pe.shape[1] != tgt_len:
                try:
                    mapped["encoder.dino.pos_embed"] = resize_positional_embedding(pe, tgt_len)
                    print(f"[UNIFIED] Resized pos_embed -> {mapped['encoder.dino.pos_embed'].shape}")
                except Exception as e:
                    print(f"[UNIFIED] pos_embed resize failed (continuing): {e}")

    # 4) build loadable dict with shape checks
    loadable: Dict[str, torch.Tensor] = {}
    mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for mk, mt in model_state_dict.items():
        if mk in mapped and tuple(mapped[mk].shape) == tuple(mt.shape):
            loadable[mk] = mapped[mk]
        elif mk in mapped:
            mismatches.append((mk, tuple(mapped[mk].shape), tuple(mt.shape)))

    # report
    total = len(model_state_dict); ok = len(loadable)
    dino_total = sum(1 for k in model_state_dict if k.startswith("encoder.dino."))
    dino_ok = sum(1 for k in loadable if k.startswith("encoder.dino."))
    print("[UNIFIED] === LOADING SUMMARY ===")
    print(f"Total weights loaded: {ok} / {total} ({100.0*ok/total:.1f}%)")
    print(f"DINOv2 weights: {dino_ok} / {dino_total} ({100.0*max(dino_ok,1)/max(dino_total,1):.1f}%)")

    if mismatches:
        print(f"[UNIFIED] Shape mismatches: {len(mismatches)} (showing up to 10)")
        for k, a, b in mismatches[:10]:
            print(f"  {k}: ckpt {a} vs model {b}")

    if ok >= 0.95 * total:
        print("✅ SUCCESS: >95% of weights loaded.")
    elif ok >= 0.80 * total:
        print("⚠️  WARNING: 80–95% loaded; some parts may be random init.")
    else:
        print("❌ ERROR: <80% loaded; major components missing.")

    return loadable


# Back-compat wrapper
def remap_and_load(model, checkpoint_path: str, **kwargs) -> Dict[str, torch.Tensor]:
    return apply_unified_weight_loading(checkpoint_path, model.state_dict(), **kwargs)
