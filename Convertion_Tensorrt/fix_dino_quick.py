#!/usr/bin/env python3
import torch
import timm

# Download DINOv2 weights
print("Downloading DINOv2 weights...")
model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True)
dinov2_weights = model.state_dict()

# Load your existing checkpoint
ckpt_path = "../imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu")
ma_state_dict = checkpoint.get('state_dict', checkpoint)

# Combine them
combined_state_dict = {}
for key, value in ma_state_dict.items():
    combined_state_dict[key] = value

for key, value in dinov2_weights.items():
    combined_key = "encoder.dino." + key
    combined_state_dict[combined_key] = value

# Save complete checkpoint
output_path = "../imcui/third_party/MatchAnything/weights/matchanything_roma_complete.ckpt"
torch.save({'state_dict': combined_state_dict}, output_path)
print(f"✅ Complete checkpoint saved: {output_path}")
print(f"   - MatchAnything keys: {len(ma_state_dict)}")
print(f"   - DINOv2 keys: {len(dinov2_weights)}")
print(f"   - Total keys: {len(combined_state_dict)}")
