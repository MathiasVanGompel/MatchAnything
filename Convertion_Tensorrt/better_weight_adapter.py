#!/usr/bin/env python3
"""
Better weight adapter that handles the specific MatchAnything checkpoint structure.
"""

import re
import torch


def create_better_mapping_rules():
    """Enhanced mapping rules for MatchAnything checkpoint."""
    return [
        # Remove wrappers
        (re.compile(r"^module\."), ""),
        (re.compile(r"^_orig_mod\."), ""),
        # MatchAnything structure - the key insight is that we need to map the adapted DINOv2 weights
        (re.compile(r"^encoder\.dino\."), "encoder.dino."),  # Keep DINOv2 weights as-is
        # CNN encoder mapping
        (re.compile(r"^matcher\.model\.encoder\.cnn\.layers\."), "encoder.layers."),
        (re.compile(r"^matcher\.model\.encoder\."), "encoder."),
        # Matcher/decoder mapping
        (re.compile(r"^matcher\.model\.decoder\."), "matcher."),
        (re.compile(r"^matcher\.model\."), ""),
        (re.compile(r"^matcher\."), ""),
        (re.compile(r"^model\."), ""),
    ]


def better_remap_and_load(model, ckpt_path: str):
    """Better weight loading with enhanced diagnostics."""
    print(f"[BETTER] Loading checkpoint: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = raw.get("state_dict", raw)
    model_state_dict = model.state_dict()

    print(f"[BETTER] Checkpoint: {len(ckpt_state_dict)} keys")
    print(f"[BETTER] Model: {len(model_state_dict)} keys")

    # Analyze checkpoint prefixes
    ckpt_prefixes = {}
    for key in ckpt_state_dict.keys():
        prefix = ".".join(key.split(".")[:2])
        ckpt_prefixes[prefix] = ckpt_prefixes.get(prefix, 0) + 1

    print("[BETTER] Checkpoint key prefixes:")
    for prefix, count in sorted(
        ckpt_prefixes.items(), key=lambda x: x[1], reverse=True
    )[:15]:
        print(f"  {prefix}: {count}")

    # Apply mapping
    rules = create_better_mapping_rules()
    mapped = {}

    for ckpt_key, ckpt_value in ckpt_state_dict.items():
        mapped_key = ckpt_key
        for pattern, replacement in rules:
            mapped_key = pattern.sub(replacement, mapped_key)
        mapped[mapped_key] = ckpt_value

    # Direct matches
    loadable = {}
    shape_mismatches = []

    for model_key, model_tensor in model_state_dict.items():
        if model_key in mapped:
            ckpt_tensor = mapped[model_key]
            if ckpt_tensor.shape == model_tensor.shape:
                loadable[model_key] = ckpt_tensor
            else:
                shape_mismatches.append(
                    (model_key, ckpt_tensor.shape, model_tensor.shape)
                )

    print(f"[BETTER] Direct matches: {len(loadable)}")

    # Show what DINOv2 weights we have
    dino_available = [k for k in mapped.keys() if k.startswith("encoder.dino.")]
    dino_needed = [k for k in model_state_dict.keys() if k.startswith("encoder.dino.")]
    dino_matched = [k for k in loadable.keys() if k.startswith("encoder.dino.")]

    print(
        f"[BETTER] DINOv2 weights - Available: {len(dino_available)}, Needed: {len(dino_needed)}, Matched: {len(dino_matched)}"
    )

    if len(dino_matched) < len(dino_needed):
        print("[BETTER] Missing DINOv2 weights (first 10):")
        missing_dino = [k for k in dino_needed if k not in dino_matched]
        for key in missing_dino[:10]:
            print(f"  - {key}")

        print("[BETTER] Available DINOv2 weights (first 10):")
        for key in dino_available[:10]:
            print(f"  + {key}")

    # Load the weights
    missing, unexpected = model.load_state_dict(loadable, strict=False)

    print("[BETTER] Final result:")
    print(f"  Loaded: {len(loadable)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")
    print(f"  Shape mismatches: {len(shape_mismatches)}")

    if shape_mismatches:
        print("[BETTER] Shape mismatches (first 5):")
        for key, ckpt_shape, model_shape in shape_mismatches[:5]:
            print(f"  {key}: {ckpt_shape} vs {model_shape}")

    return loadable


if __name__ == "__main__":
    # Test the better adapter
    import sys

    sys.path.append(".")
    from accurate_matchanything_trt import AccurateMatchAnythingTRT

    model = AccurateMatchAnythingTRT(model_name="matchanything_roma")
    better_remap_and_load(
        model,
        "../imcui/third_party/MatchAnything/weights/matchanything_roma_adapted_dino.ckpt",
    )
