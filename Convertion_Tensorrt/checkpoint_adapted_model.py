#!/usr/bin/env python3
"""
Model adapted to match the actual MatchAnything checkpoint structure.
Based on the papers and actual checkpoint analysis.
"""

import torch
from typing import Dict
import sys

sys.path.append(".")


def create_checkpoint_compatible_mapping():
    """Create mapping rules based on actual checkpoint structure from papers."""
    return [
        # The key insight: embedding_decoder IS the DINOv2 backbone
        (r"^matcher\.model\.decoder\.embedding_decoder\.", "encoder.dino."),
        # CNN encoder mapping
        (r"^matcher\.model\.encoder\.cnn\.layers\.", "encoder.layers."),
        (r"^matcher\.model\.encoder\.", "encoder."),
        # Matcher components
        (r"^matcher\.model\.decoder\.", "matcher."),
        (r"^matcher\.model\.", ""),
        (r"^matcher\.", ""),
        (r"^model\.", ""),
    ]


def load_with_checkpoint_structure(model, ckpt_path: str):
    """Load weights by adapting to the checkpoint's actual structure."""
    print(f"[ADAPTED] Loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = raw.get("state_dict", raw)
    print(f"[ADAPTED] Checkpoint: {len(ckpt_state_dict)} keys")

    embedding_decoder_keys = [
        k for k in ckpt_state_dict.keys() if "embedding_decoder" in k
    ]
    print(
        f"[ADAPTED] Found {len(embedding_decoder_keys)} embedding_decoder keys (this is DINOv2!)"
    )

    import re

    mapping_rules = create_checkpoint_compatible_mapping()
    mapped_dict = {}
    for ckpt_key, ckpt_value in ckpt_state_dict.items():
        mapped_key = ckpt_key
        for pattern, replacement in mapping_rules:
            mapped_key = re.sub(pattern, replacement, mapped_key)
        mapped_dict[mapped_key] = ckpt_value

    print(f"[ADAPTED] After mapping: {len(mapped_dict)} keys")

    dino_available = [k for k in mapped_dict.keys() if k.startswith("encoder.dino.")]
    print(f"[ADAPTED] DINOv2 keys after mapping: {len(dino_available)}")
    if dino_available:
        print("[ADAPTED] Sample mapped DINOv2 keys:")
        for key in sorted(dino_available)[:10]:
            print(f"  {key}: {mapped_dict[key].shape}")

    model_state_dict = model.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    shape_mismatches = []
    for model_key, model_tensor in model_state_dict.items():
        if model_key in mapped_dict:
            ckpt_tensor = mapped_dict[model_key]
            if ckpt_tensor.shape == model_tensor.shape:
                loadable[model_key] = ckpt_tensor
            else:
                shape_mismatches.append(
                    (model_key, ckpt_tensor.shape, model_tensor.shape)
                )

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    dino_loaded = len([k for k in loadable.keys() if k.startswith("encoder.dino.")])
    dino_total = len(
        [k for k in model_state_dict.keys() if k.startswith("encoder.dino.")]
    )

    print("[ADAPTED] Results:")
    print(f"  Total loaded: {len(loadable)}/{len(model_state_dict)}")
    print(f"  DINOv2 loaded: {dino_loaded}/{dino_total}")
    print(f"  Missing: {len(missing)}")
    print(f"  Shape mismatches: {len(shape_mismatches)}")

    if shape_mismatches:
        print("[ADAPTED] Shape mismatches (first 10):")
        for key, ckpt_shape, model_shape in shape_mismatches[:10]:
            print(f"  {key}: checkpoint {ckpt_shape} vs model {model_shape}")

    if dino_loaded < dino_total * 0.8:
        print("[ADAPTED] ⚠️  Many DINOv2 weights missing. Checking block structure...")
        missing_dino = [
            k
            for k in model_state_dict.keys()
            if k.startswith("encoder.dino.") and k not in loadable
        ]
        available_dino = [
            k for k in mapped_dict.keys() if k.startswith("encoder.dino.")
        ]
        print(f"[ADAPTED] Missing DINOv2 keys (first 5): {missing_dino[:5]}")
        print(f"[ADAPTED] Available DINOv2 keys (first 5): {available_dino[:5]}")
        model_blocks = [k for k in missing_dino if "blocks." in k]
        ckpt_blocks = [k for k in available_dino if "blocks." in k]
        if model_blocks and ckpt_blocks:
            print("[ADAPTED] Block structure analysis:")
            print(f"  Model expects: {model_blocks[0] if model_blocks else 'N/A'}")
            print(f"  Checkpoint has: {ckpt_blocks[0] if ckpt_blocks else 'N/A'}")

    return loadable


if __name__ == "__main__":
    from accurate_matchanything_trt import AccurateMatchAnythingTRT

    model = AccurateMatchAnythingTRT(model_name="matchanything_roma")
    load_with_checkpoint_structure(
        model, "../imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt"
    )
