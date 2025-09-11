#!/usr/bin/env python3
"""
Fixed weight adapter that correctly maps DINOv2 block structure.
"""

import re
import torch
from typing import Dict


def fix_dino_block_structure(
    ckpt_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Fix the DINOv2 block structure to match model expectations."""

    fixed_dict = {}

    for key, value in ckpt_state_dict.items():
        new_key = key

        # Fix DINOv2 block structure: blocks.X. -> blocks.X.0.
        if (
            key.startswith("encoder.dino.blocks.")
            and ".norm" in key
            or ".attn" in key
            or ".mlp" in key
        ):
            # Parse the key: encoder.dino.blocks.0.norm1.weight -> encoder.dino.blocks.0.0.norm1.weight
            parts = key.split(".")
            if len(parts) >= 5 and parts[3].isdigit():  # blocks.0.norm1...
                # Insert an extra '0' after the block number
                parts.insert(4, "0")
                new_key = ".".join(parts)
                print(f"Fixed block structure: {key} -> {new_key}")

        fixed_dict[new_key] = value

    return fixed_dict


def apply_fixed_mapping(model, ckpt_path: str):
    """Apply the fixed mapping with correct block structure."""

    print(f"[FIXED] Loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = raw.get("state_dict", raw)

    print(f"[FIXED] Original checkpoint: {len(ckpt_state_dict)} keys")

    # Fix the DINOv2 block structure
    fixed_dict = fix_dino_block_structure(ckpt_state_dict)

    print(f"[FIXED] After block structure fix: {len(fixed_dict)} keys")

    # Apply standard mapping rules
    rules = [
        (re.compile(r"^module\."), ""),
        (re.compile(r"^matcher\.model\.encoder\.cnn\.layers\."), "encoder.layers."),
        (re.compile(r"^matcher\.model\.encoder\."), "encoder."),
        (re.compile(r"^matcher\.model\.decoder\."), "matcher."),
        (re.compile(r"^matcher\.model\."), ""),
        (re.compile(r"^matcher\."), ""),
    ]

    mapped_dict = {}
    for key, value in fixed_dict.items():
        mapped_key = key
        for pattern, replacement in rules:
            mapped_key = pattern.sub(replacement, mapped_key)
        mapped_dict[mapped_key] = value

    # Load into model
    model_state_dict = model.state_dict()
    loadable = {}

    for model_key, model_tensor in model_state_dict.items():
        if model_key in mapped_dict:
            ckpt_tensor = mapped_dict[model_key]
            if ckpt_tensor.shape == model_tensor.shape:
                loadable[model_key] = ckpt_tensor

    # Load the weights
    missing, unexpected = model.load_state_dict(loadable, strict=False)

    # Count DINOv2 weights specifically
    dino_loaded = len([k for k in loadable.keys() if k.startswith("encoder.dino.")])
    dino_total = len(
        [k for k in model_state_dict.keys() if k.startswith("encoder.dino.")]
    )

    print("[FIXED] Results:")
    print(f"  Total loaded: {len(loadable)}/{len(model_state_dict)}")
    print(f"  DINOv2 loaded: {dino_loaded}/{dino_total}")
    print(f"  Missing: {len(missing)}")

    return loadable


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from accurate_matchanything_trt import AccurateMatchAnythingTRT

    model = AccurateMatchAnythingTRT(model_name="matchanything_roma")
    apply_fixed_mapping(
        model,
        "../imcui/third_party/MatchAnything/weights/matchanything_roma_adapted_dino.ckpt",
    )
