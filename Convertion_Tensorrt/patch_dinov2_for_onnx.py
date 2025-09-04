#!/usr/bin/env python3
"""
Patch the original DINOv2 implementation to be ONNX-compatible.
This fixes the interpolate issue that causes ONNX export to fail.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def patch_dinov2_interpolate_pos_encoding(dinov2_model):
    """
    Monkey-patch DINOv2's interpolate_pos_encoding to be ONNX-safe.
    Fixes the upsample_bicubic2d argument type error.
    """
    if not hasattr(dinov2_model, 'interpolate_pos_encoding'):
        return
    
    # Store original method
    original_interpolate = dinov2_model.interpolate_pos_encoding
    pos_embed = dinov2_model.pos_embed
    
    def onnx_safe_interpolate(x, w, h):
        """
        ONNX-safe version of interpolate_pos_encoding.
        Uses integer output_size instead of tensor scale_factors.
        """
        N = pos_embed.shape[1] - 1  # Exclude CLS token
        if N == w * h:
            return pos_embed
        
        # Extract class and patch tokens
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        
        # Reshape patch embeddings to 2D grid
        sqrt_N = int(math.sqrt(N))
        if sqrt_N * sqrt_N != N:
            # Fallback for non-square patches
            h_pre = int(math.sqrt(N))
            w_pre = N // h_pre
        else:
            h_pre = w_pre = sqrt_N
        
        patch_pos_embed_2d = patch_pos_embed.reshape(1, h_pre, w_pre, dim).permute(0, 3, 1, 2)
        
        # Use integer output size for ONNX compatibility
        w_int = int(w.item()) if isinstance(w, torch.Tensor) else int(w)
        h_int = int(h.item()) if isinstance(h, torch.Tensor) else int(h)
        
        # Interpolate to target size
        patch_pos_embed_resized = F.interpolate(
            patch_pos_embed_2d,
            size=(h_int, w_int),  # Use integer size, not scale_factors
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back to sequence
        patch_pos_embed_resized = patch_pos_embed_resized.permute(0, 2, 3, 1).view(1, -1, dim)
        
        # Combine class and patch embeddings
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed_resized), dim=1)
    
    # Replace the method
    dinov2_model.interpolate_pos_encoding = onnx_safe_interpolate
    print(f"[PATCH] Applied ONNX-safe interpolate_pos_encoding to DINOv2")

def patch_all_dinov2_models(model):
    """
    Recursively find and patch all DINOv2 models in the network.
    """
    patched_count = 0
    
    def recursive_patch(module, name=""):
        nonlocal patched_count
        
        # Check if this module has the problematic method
        if hasattr(module, 'interpolate_pos_encoding') and hasattr(module, 'pos_embed'):
            patch_dinov2_interpolate_pos_encoding(module)
            patched_count += 1
            print(f"[PATCH] Patched DINOv2 at: {name}")
        
        # Recursively check children
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            recursive_patch(child_module, full_name)
    
    recursive_patch(model)
    print(f"[PATCH] Total DINOv2 models patched: {patched_count}")
    return patched_count > 0