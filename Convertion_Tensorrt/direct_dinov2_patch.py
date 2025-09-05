#!/usr/bin/env python3
"""
Direct DINOv2 patching by accessing the specific module path.
Based on the error: model.encoder.dinov2_vitl14[0]
"""
import torch
import torch.nn.functional as F
import math

def patch_dinov2_direct(model):
    """
    Directly patch the DINOv2 module using the known path from error.
    Error path: model.encoder.dinov2_vitl14[0].forward_features
    """
    patched_count = 0
    
    try:
        # Try to access the encoder
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            print(f"[PATCH] Found encoder: {type(encoder)}")
            
            # Try to access dinov2_vitl14
            if hasattr(encoder, 'dinov2_vitl14'):
                dinov2_list = encoder.dinov2_vitl14
                print(f"[PATCH] Found dinov2_vitl14: {type(dinov2_list)}, length: {len(dinov2_list) if hasattr(dinov2_list, '__len__') else 'unknown'}")
                
                # Try to access the first element (index 0)
                if hasattr(dinov2_list, '__getitem__'):
                    try:
                        dinov2_model = dinov2_list[0]
                        print(f"[PATCH] Found dinov2_vitl14[0]: {type(dinov2_model)}")
                        
                        # Check if this has the interpolate method
                        if hasattr(dinov2_model, 'interpolate_pos_encoding'):
                            print(f"[PATCH] *** FOUND TARGET: dinov2_vitl14[0] has interpolate_pos_encoding! ***")
                            
                            # Apply the patch
                            patch_dinov2_interpolate_method(dinov2_model)
                            patched_count += 1
                            print(f"[PATCH] *** SUCCESS: Patched dinov2_vitl14[0] ***")
                        else:
                            print(f"[PATCH] dinov2_vitl14[0] does not have interpolate_pos_encoding")
                            
                    except (IndexError, TypeError) as e:
                        print(f"[PATCH] Could not access dinov2_vitl14[0]: {e}")
                else:
                    print(f"[PATCH] dinov2_vitl14 is not indexable")
            else:
                print(f"[PATCH] Encoder does not have dinov2_vitl14")
                
                # List all attributes of encoder for debugging
                print(f"[DEBUG] Encoder attributes:")
                for attr_name in dir(encoder):
                    if not attr_name.startswith('_'):
                        try:
                            attr = getattr(encoder, attr_name)
                            if not callable(attr):
                                print(f"  {attr_name}: {type(attr)}")
                        except:
                            pass
        else:
            print(f"[PATCH] Model does not have encoder")
            
    except Exception as e:
        print(f"[PATCH] Error in direct patching: {e}")
        import traceback
        traceback.print_exc()
    
    return patched_count > 0

def patch_dinov2_interpolate_method(dinov2_model):
    """
    Patch the interpolate_pos_encoding method of a DINOv2 model.
    """
    if not hasattr(dinov2_model, 'pos_embed'):
        print(f"[PATCH] Warning: DINOv2 model has no pos_embed")
        return
        
    pos_embed = dinov2_model.pos_embed
    
    def onnx_safe_interpolate(x, w, h):
        """ONNX-safe interpolate_pos_encoding"""
        N = pos_embed.shape[1] - 1  # Exclude CLS token
        if N == w * h:
            return pos_embed
            
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        
        # Calculate original grid size
        sqrt_N = int(math.sqrt(N))
        if sqrt_N * sqrt_N == N:
            h_orig = w_orig = sqrt_N
        else:
            h_orig = int(math.sqrt(N))
            w_orig = N // h_orig
        
        # Reshape to 2D
        patch_pos_embed_2d = patch_pos_embed.reshape(1, h_orig, w_orig, dim).permute(0, 3, 1, 2)
        
        # Convert to integers for ONNX
        w_int = int(w.item()) if torch.is_tensor(w) else int(w)
        h_int = int(h.item()) if torch.is_tensor(h) else int(h)
        
        # Clamp to valid range
        w_int = max(1, min(w_int, 1000))  # Reasonable bounds
        h_int = max(1, min(h_int, 1000))
        
        # ONNX-safe interpolation without bicubic antialias kernel
        patch_pos_embed_resized = F.interpolate(
            patch_pos_embed_2d,
            size=(h_int, w_int),
            mode='bilinear',
            align_corners=False,
            antialias=False,
        )
        
        # Reshape back to sequence
        patch_pos_embed_resized = patch_pos_embed_resized.permute(0, 2, 3, 1).view(1, -1, dim)
        
        # Combine class and patch embeddings
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed_resized), dim=1)
    
    # Replace the method
    dinov2_model.interpolate_pos_encoding = onnx_safe_interpolate
    print(f"[PATCH] Applied ONNX-safe interpolate_pos_encoding")

if __name__ == "__main__":
    print("This module provides direct DINOv2 patching functions")