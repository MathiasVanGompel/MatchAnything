#!/usr/bin/env python3
"""
Direct source patching for DINOv2 to fix ONNX export issues.
This monkey-patches the problematic function globally.
"""
import sys
import torch
import torch.nn.functional as F
import math

def apply_global_dinov2_patch():
    """
    Apply a global monkey patch to fix DINOv2 ONNX export issues.
    This replaces the problematic interpolate function everywhere.
    """
    print("[PATCH] Applying global DINOv2 ONNX fix...")
    
    # Find the DINOv2 module in sys.modules
    dinov2_module = None
    for module_name, module in sys.modules.items():
        if 'dinov2' in module_name and hasattr(module, '__file__') and 'dinov2.py' in str(module.__file__):
            dinov2_module = module
            print(f"[PATCH] Found DINOv2 module: {module_name}")
            break
    
    if dinov2_module is None:
        print("[PATCH] DINOv2 module not found in sys.modules")
        return False
    
    # Check if the module has the problematic class
    if hasattr(dinov2_module, 'DinoVisionTransformer'):
        dinov2_class = dinov2_module.DinoVisionTransformer
        
        # Create the ONNX-safe interpolate method
        def onnx_safe_interpolate_pos_encoding(self, x, w, h):
            """
            ONNX-safe version that uses integer output_size instead of tensor scale_factors.
            """
            N = self.pos_embed.shape[1] - 1
            if N == w * h:
                return self.pos_embed
            
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            
            # Calculate grid dimensions
            sqrt_N = int(math.sqrt(N))
            if sqrt_N * sqrt_N == N:
                h_pre = w_pre = sqrt_N
            else:
                # Fallback for non-square
                h_pre = int(math.sqrt(N))
                w_pre = N // h_pre
            
            # Reshape to 2D
            patch_pos_embed_2d = patch_pos_embed.reshape(1, h_pre, w_pre, dim).permute(0, 3, 1, 2)
            
            # Convert tensor dimensions to integers
            if isinstance(w, torch.Tensor):
                w_int = int(w.item())
                h_int = int(h.item()) 
            else:
                w_int = int(w)
                h_int = int(h)
            
            # Ensure valid dimensions
            w_int = max(1, w_int)
            h_int = max(1, h_int)
            
            # Use ONNX-safe interpolation with integer size, avoiding bicubic AA
            patch_pos_embed_resized = F.interpolate(
                patch_pos_embed_2d,
                size=(h_int, w_int),
                mode='bilinear',
                align_corners=False,
                antialias=False,
            )
            
            # Reshape back
            patch_pos_embed_resized = patch_pos_embed_resized.permute(0, 2, 3, 1).view(1, -1, dim)
            
            # Combine class and patch embeddings
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed_resized), dim=1)
        
        # Replace the method globally
        dinov2_class.interpolate_pos_encoding = onnx_safe_interpolate_pos_encoding
        print(f"[PATCH] Globally patched DinoVisionTransformer.interpolate_pos_encoding")
        return True
    
    print("[PATCH] DinoVisionTransformer class not found")
    return False

def patch_dinov2_before_import():
    """
    Patch DINOv2 before it's imported by the model.
    This ensures the patch is applied before any instances are created.
    """
    # Import and patch the DINOv2 module
    try:
        # Import the specific DINOv2 transformer module
        import importlib.util
        
        # Try to find and patch the DINOv2 module
        roma_path = None
        for path in sys.path:
            if 'ROMA' in path:
                roma_path = path
                break
        
        if roma_path:
            dinov2_file = os.path.join(roma_path, 'roma', 'models', 'transformer', 'dinov2.py')
            if os.path.exists(dinov2_file):
                print(f"[PATCH] Found DINOv2 file: {dinov2_file}")
                
                # Import the module
                spec = importlib.util.spec_from_file_location("dinov2_module", dinov2_file)
                dinov2_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dinov2_module)
                
                # Patch the class
                if hasattr(dinov2_module, 'DinoVisionTransformer'):
                    return apply_global_dinov2_patch()
        
        return False
        
    except Exception as e:
        print(f"[PATCH] Error in pre-import patching: {e}")
        return False

if __name__ == "__main__":
    apply_global_dinov2_patch()