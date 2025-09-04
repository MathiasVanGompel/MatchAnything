#!/usr/bin/env python3
"""
Direct source file patching for DINOv2 to fix ONNX export issues.
This modifies the actual dinov2.py file to use output_size instead of scale_factor.
"""
import os
import sys
from pathlib import Path

def find_dinov2_file():
    """Find the actual DINOv2 source file"""
    # Look in the ROMA third_party directory
    current_dir = Path(__file__).parent
    dinov2_paths = [
        current_dir / "../imcui/third_party/MatchAnything/third_party/ROMA/roma/models/transformer/dinov2.py",
        current_dir / "../imcui/third_party/MatchAnything/third_party/ROMA/romatch/models/transformer/dinov2.py",
    ]
    
    for path in dinov2_paths:
        if path.exists():
            return path
    
    return None

def patch_dinov2_interpolate_source():
    """
    Directly patch the DINOv2 source file to fix the interpolation issue.
    """
    dinov2_file = find_dinov2_file()
    if not dinov2_file:
        print("[PATCH] Could not find DINOv2 source file")
        return False
    
    print(f"[PATCH] Found DINOv2 file: {dinov2_file}")
    
    # Read the original file
    try:
        with open(dinov2_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"[PATCH] Could not read DINOv2 file: {e}")
        return False
    
    # Check if already patched
    if "# ONNX_PATCH_APPLIED" in content:
        print("[PATCH] DINOv2 file already patched")
        return True
    
    # Find the problematic interpolation code
    old_code = '''patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )'''
    
    # New ONNX-safe code
    new_code = '''# ONNX_PATCH_APPLIED - Use output_size instead of scale_factor for ONNX compatibility
        grid_size = int(math.sqrt(N))
        new_h = int(h0.item()) if torch.is_tensor(h0) else int(h0)
        new_w = int(w0.item()) if torch.is_tensor(w0) else int(w0)
        new_h = max(1, new_h)  # Ensure positive
        new_w = max(1, new_w)  # Ensure positive
        
        patch_pos_embed_2d = patch_pos_embed.reshape(1, grid_size, grid_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed_2d,
            size=(new_h, new_w),  # Use explicit size instead of scale_factor
            mode="bicubic",
            align_corners=False
        )'''
    
    # Apply the patch
    if old_code in content:
        patched_content = content.replace(old_code, new_code)
        
        # Write back the patched file
        try:
            with open(dinov2_file, 'w') as f:
                f.write(patched_content)
            print("[PATCH] Successfully patched DINOv2 interpolate_pos_encoding")
            return True
        except Exception as e:
            print(f"[PATCH] Could not write patched file: {e}")
            return False
    else:
        print("[PATCH] Could not find the target code to patch")
        print("[PATCH] The DINOv2 file might have a different structure")
        return False

def restore_dinov2_original():
    """Restore the original DINOv2 file (undo patch)"""
    dinov2_file = find_dinov2_file()
    if not dinov2_file:
        print("[RESTORE] Could not find DINOv2 source file")
        return False
    
    try:
        with open(dinov2_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"[RESTORE] Could not read DINOv2 file: {e}")
        return False
    
    if "# ONNX_PATCH_APPLIED" not in content:
        print("[RESTORE] DINOv2 file is not patched")
        return True
    
    # Restore original code
    patched_code = '''# ONNX_PATCH_APPLIED - Use output_size instead of scale_factor for ONNX compatibility
        grid_size = int(math.sqrt(N))
        new_h = int(h0.item()) if torch.is_tensor(h0) else int(h0)
        new_w = int(w0.item()) if torch.is_tensor(w0) else int(w0)
        new_h = max(1, new_h)  # Ensure positive
        new_w = max(1, new_w)  # Ensure positive
        
        patch_pos_embed_2d = patch_pos_embed.reshape(1, grid_size, grid_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed_2d,
            size=(new_h, new_w),  # Use explicit size instead of scale_factor
            mode="bicubic",
            align_corners=False
        )'''
    
    original_code = '''patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )'''
    
    restored_content = content.replace(patched_code, original_code)
    
    try:
        with open(dinov2_file, 'w') as f:
            f.write(restored_content)
        print("[RESTORE] Successfully restored original DINOv2 file")
        return True
    except Exception as e:
        print(f"[RESTORE] Could not write restored file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true", help="Restore original file")
    args = parser.parse_args()
    
    if args.restore:
        restore_dinov2_original()
    else:
        patch_dinov2_interpolate_source()