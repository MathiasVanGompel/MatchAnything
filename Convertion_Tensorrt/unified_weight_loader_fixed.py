#!/usr/bin/env python3
"""
Fixed unified weight loading system that handles BlockChunk architecture.
"""

import re
import torch
import timm
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F


def create_comprehensive_mapping_rules() -> List[Tuple[re.Pattern, str]]:
    """Create comprehensive mapping rules for MatchAnything checkpoint structure."""
    return [
        (re.compile(r"^module\."), ""),
        (re.compile(r"^_orig_mod\."), ""),
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
        (re.compile(r"^backbone\."), "encoder.dino."),
        (re.compile(r"^vit\."), "encoder.dino."),
        (re.compile(r"^dino\."), "encoder.dino."),
        (re.compile(r"^encoder\.vit\."), "encoder.dino."),
        (re.compile(r"^encoder\.backbone\."), "encoder.dino."),
        (re.compile(r"^encoder\.dino\."), "encoder.dino."),
        (re.compile(r"^encoder\."), "encoder."),
    ]


def fix_dinov2_block_structure(
    weights_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Fix DINOv2 transformer block naming structure for BlockChunk architecture."""
    fixed = {}
    for key, value in weights_dict.items():
        if key.startswith("encoder.dino.blocks."):
            parts = key.split(".")
            if len(parts) >= 4 and parts[3].isdigit():
                block_num = parts[3]
                if len(parts) >= 5 and parts[4].isdigit():
                    fixed[key] = value
                else:
                    new_key = ".".join(parts[:3] + ["0", block_num] + parts[4:])
                    fixed[new_key] = value
                    print(f"[UNIFIED] Fixed BlockChunk structure: {key} -> {new_key}")
            else:
                fixed[key] = value
        else:
            fixed[key] = value
    return fixed


def resize_positional_embedding(
    pos_embed: torch.Tensor, target_size: int
) -> torch.Tensor:
    if pos_embed.shape[1] == target_size:
        print(f"[UNIFIED] Positional embedding already correct size: {pos_embed.shape}")
        return pos_embed
    print(
        f"[UNIFIED] Resizing positional embedding: {pos_embed.shape} -> [1, {target_size}, {pos_embed.shape[2]}]"
    )
    cls_token = pos_embed[:, 0:1, :]
    spatial_tokens = pos_embed[:, 1:, :]
    num_spatial = spatial_tokens.shape[1]
    original_size = int(np.sqrt(num_spatial))
    target_spatial = target_size - 1
    target_grid = int(np.sqrt(target_spatial))
    spatial_2d = spatial_tokens.reshape(1, original_size, original_size, -1).permute(
        0, 3, 1, 2
    )
    resized = F.interpolate(
        spatial_2d,
        size=(target_grid, target_grid),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.permute(0, 2, 3, 1).reshape(1, target_spatial, -1)
    return torch.cat([cls_token, resized], dim=1)


def load_dinov2_components_func(
    model_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    dinov2_weights = {}
    try:
        print("[UNIFIED] Loading official DINOv2 weights for missing components...")
        dinov2_model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=True
        )
        official = dinov2_model.state_dict()
        if "encoder.dino.pos_embed" in model_state_dict and "pos_embed" in official:
            pos = official["pos_embed"]
            tgt = model_state_dict["encoder.dino.pos_embed"]
            if pos.shape != tgt.shape:
                pos = resize_positional_embedding(pos, tgt.shape[1])
            dinov2_weights["encoder.dino.pos_embed"] = pos
            print(f"[UNIFIED] Added pos_embed: {pos.shape}")
        if "encoder.dino.cls_token" in model_state_dict:
            dinov2_weights["encoder.dino.cls_token"] = official.get(
                "cls_token",
                torch.zeros_like(model_state_dict["encoder.dino.cls_token"]),
            )
            print(
                f"[UNIFIED] Added cls_token: {dinov2_weights['encoder.dino.cls_token'].shape}"
            )
        if "encoder.dino.mask_token" in model_state_dict:
            dinov2_weights["encoder.dino.mask_token"] = official.get(
                "mask_token",
                torch.zeros_like(model_state_dict["encoder.dino.mask_token"]),
            )
            print(
                f"[UNIFIED] Added mask_token: {dinov2_weights['encoder.dino.mask_token'].shape}"
            )
        if any(
            k.startswith("encoder.dino.patch_embed.") for k in model_state_dict.keys()
        ):
            for k, v in official.items():
                if k.startswith("patch_embed."):
                    nk = f"encoder.dino.{k}"
                    if nk in model_state_dict:
                        dinov2_weights[nk] = v
                        print(f"[UNIFIED] Added {nk}: {v.shape}")
    except Exception as e:
        print(f"[UNIFIED] Warning: Could not load official DINOv2 weights: {e}")
    return dinov2_weights


def apply_unified_weight_loading(
    checkpoint_path: str,
    model_state_dict: Dict[str, torch.Tensor],
    load_dinov2_components: bool = True,
) -> Dict[str, torch.Tensor]:
    print(f"[UNIFIED] Loading checkpoint: {checkpoint_path}")
    try:
        raw = torch.load(checkpoint_path, map_location="cpu")
        ckpt = raw.get("state_dict", raw)
    except Exception as e:
        print(f"[UNIFIED] Error loading checkpoint: {e}")
        return {}
    print(f"[UNIFIED] Checkpoint has {len(ckpt)} keys")
    rules = create_comprehensive_mapping_rules()
    mapped = {}
    for k, v in ckpt.items():
        nk = k
        for pat, rep in rules:
            nk = pat.sub(rep, nk)
        mapped[nk] = v
    print(f"[UNIFIED] After mapping: {len(mapped)} keys")
    mapped = fix_dinov2_block_structure(mapped)
    if load_dinov2_components:
        mapped.update(load_dinov2_components_func(model_state_dict))
    loadable: Dict[str, torch.Tensor] = {}
    shape_mismatches = []
    for mk, mt in model_state_dict.items():
        if mk in mapped:
            ck = mapped[mk]
            if ck.shape == mt.shape:
                loadable[mk] = ck
            else:
                shape_mismatches.append((mk, ck.shape, mt.shape))
    if shape_mismatches:
        print("[UNIFIED] Shape mismatches:")
        for mk, ck_s, mt_s in shape_mismatches[:5]:
            print(f"  {mk}: {ck_s} vs {mt_s}")
    remaining_model = set(model_state_dict.keys()) - set(loadable.keys())
    remaining_ckpt = set(mapped.keys()) - set(loadable.keys())
    for mk in remaining_model:
        suffix = ".".join(mk.split(".")[-2:])
        for ck in remaining_ckpt:
            if ck.endswith(suffix) and mapped[ck].shape == model_state_dict[mk].shape:
                loadable[mk] = mapped[ck]
                break
    print("[UNIFIED] === LOADING SUMMARY ===")
    print(
        f"Total weights loaded: {len(loadable)} / {len(model_state_dict)} ({100.0*len(loadable)/max(len(model_state_dict),1):.1f}%)"
    )
    dino_loaded = sum(1 for k in loadable if k.startswith("encoder.dino."))
    dino_total = sum(1 for k in model_state_dict if k.startswith("encoder.dino."))
    print(
        f"DINOv2 weights: {dino_loaded} / {dino_total} ({100.0*dino_loaded/max(dino_total,1):.1f}%)"
    )
    return loadable


def remap_and_load(model, checkpoint_path: str, **kwargs) -> Dict[str, torch.Tensor]:
    return apply_unified_weight_loading(checkpoint_path, model.state_dict(), **kwargs)


if __name__ == "__main__":
    print("Fixed Unified Weight Loader for MatchAnything TensorRT Conversion")
