#!/usr/bin/env python3
"""
Create a hybrid checkpoint that combines:
1. DINOv2 essential components (patch_embed, pos_embed, cls_token)
2. Transformer blocks from your checkpoint
"""

import torch
import timm


def create_hybrid_checkpoint():
    """Create hybrid checkpoint with DINOv2 essentials + your transformer blocks."""

    print("Creating hybrid MatchAnything checkpoint...")

    # 1. Load your checkpoint (has transformer blocks)
    print("[1/4] Loading your checkpoint...")
    your_ckpt = torch.load(
        "../imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt",
        map_location="cpu",
    )
    your_weights = your_ckpt.get("state_dict", your_ckpt)

    print(f"Your checkpoint: {len(your_weights)} keys")

    # 2. Load DINOv2 for essential components
    print("[2/4] Loading DINOv2 for essential components...")
    dinov2_model = timm.create_model("vit_large_patch14_dinov2", pretrained=True)
    dinov2_weights = dinov2_model.state_dict()

    print(f"DINOv2 weights: {len(dinov2_weights)} keys")

    # 3. Create hybrid weights
    print("[3/4] Creating hybrid weights...")
    hybrid_weights = {}

    # Add your original MatchAnything weights (CNN encoder, matcher, etc.)
    for key, value in your_weights.items():
        hybrid_weights[key] = value

    # Add essential DINOv2 components with adaptation
    essential_components = [
        "cls_token",
        "pos_embed",
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "mask_token",
    ]

    for component in essential_components:
        if component in dinov2_weights:
            value = dinov2_weights[component]

            if component == "pos_embed":
                # Adapt position embedding from DINOv2's size to ViT/16 size
                # DINOv2: [1, 1370, 1024] -> Target: [1, 197, 1024]
                print(f"  Adapting {component} from {value.shape}")
                cls_token = value[:, :1, :]  # [1, 1, 1024]
                spatial_tokens = value[:, 1:, :]  # [1, 1369, 1024]

                # Reshape and interpolate spatial tokens
                H = W = int(spatial_tokens.shape[1] ** 0.5)  # 37x37
                spatial_2d = spatial_tokens.reshape(1, H, W, 1024).permute(0, 3, 1, 2)

                # Interpolate to 14x14 (for 224x224 images with 16x16 patches)
                spatial_resized = (
                    torch.nn.functional.interpolate(
                        spatial_2d, size=(14, 14), mode="bilinear", align_corners=False
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(1, 196, 1024)
                )

                adapted_pos_embed = torch.cat([cls_token, spatial_resized], dim=1)
                hybrid_weights[f"encoder.dino.{component}"] = adapted_pos_embed
                print(f"  Adapted to: {adapted_pos_embed.shape}")

            elif component == "patch_embed.proj.weight":
                # Adapt patch embedding from 14x14 to 16x16
                print(f"  Adapting {component} from {value.shape}")
                adapted_patch = torch.nn.functional.interpolate(
                    value, size=(16, 16), mode="bilinear", align_corners=False
                )
                hybrid_weights[f"encoder.dino.{component}"] = adapted_patch
                print(f"  Adapted to: {adapted_patch.shape}")

            else:
                hybrid_weights[f"encoder.dino.{component}"] = value
                print(f"  Added {component}: {value.shape}")

    # Add transformer blocks from your checkpoint with structure fix
    print("  Processing transformer blocks from your checkpoint...")
    embedding_decoder_keys = [
        k for k in your_weights.keys() if "embedding_decoder.blocks." in k
    ]

    for key in embedding_decoder_keys:
        # Map: matcher.model.decoder.embedding_decoder.blocks.X.* -> encoder.dino.blocks.X.0.*
        new_key = key.replace(
            "matcher.model.decoder.embedding_decoder.", "encoder.dino."
        )

        # Fix block structure: blocks.X.* -> blocks.X.0.*
        if ".blocks." in new_key:
            parts = new_key.split(".")
            # Find the block number and insert '0' after it
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    parts.insert(i + 2, "0")  # Insert '0' after the block number
                    break
            new_key = ".".join(parts)

        hybrid_weights[new_key] = your_weights[key]

    print(f"  Added {len(embedding_decoder_keys)} transformer blocks")

    # 4. Save hybrid checkpoint
    print("[4/4] Saving hybrid checkpoint...")
    output_path = (
        "../imcui/third_party/MatchAnything/weights/matchanything_roma_hybrid.ckpt"
    )
    torch.save({"state_dict": hybrid_weights}, output_path)

    print(f"✅ Hybrid checkpoint created: {output_path}")
    print(f"Total keys: {len(hybrid_weights)}")

    # Analyze what we have
    dino_keys = [k for k in hybrid_weights.keys() if k.startswith("encoder.dino.")]
    print(f"DINOv2 keys: {len(dino_keys)}")

    essential_found = [
        k
        for k in dino_keys
        if any(comp in k for comp in ["cls_token", "pos_embed", "patch_embed"])
    ]
    blocks_found = [k for k in dino_keys if "blocks." in k]

    print(f"Essential components: {len(essential_found)}")
    print(f"Transformer blocks: {len(blocks_found)}")

    return output_path


def test_hybrid_checkpoint(hybrid_path):
    """Test the hybrid checkpoint with the model."""
    print(f"\nTesting hybrid checkpoint: {hybrid_path}")

    import sys

    sys.path.append(".")
    from accurate_matchanything_trt import AccurateMatchAnythingTRT

    model = AccurateMatchAnythingTRT(model_name="matchanything_roma")

    # Load hybrid checkpoint
    hybrid_ckpt = torch.load(hybrid_path, map_location="cpu")
    hybrid_weights = hybrid_ckpt.get("state_dict", hybrid_ckpt)

    # Try loading
    missing, unexpected = model.load_state_dict(hybrid_weights, strict=False)

    model_total = len(model.state_dict())
    loaded_count = model_total - len(missing)

    print("Loading results:")
    print(f"  Loaded: {loaded_count}/{model_total}")
    print(f"  Missing: {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")

    # Check DINOv2 specifically
    dino_model_keys = [
        k for k in model.state_dict().keys() if k.startswith("encoder.dino.")
    ]
    dino_loaded = [k for k in dino_model_keys if k not in missing]

    print(f"  DINOv2 loaded: {len(dino_loaded)}/{len(dino_model_keys)}")

    if len(missing) > 0:
        print(f"Missing keys (first 10): {missing[:10]}")

    return loaded_count, model_total


if __name__ == "__main__":
    hybrid_path = create_hybrid_checkpoint()
    test_hybrid_checkpoint(hybrid_path)
