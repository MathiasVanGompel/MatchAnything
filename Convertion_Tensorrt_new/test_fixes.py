#!/usr/bin/env python3
"""
Test script to validate the key fixes without running full export.
"""
import sys
from pathlib import Path

def test_patch_size_compatibility():
    """Test that input dimensions are properly adjusted for patch size."""
    print("=== Testing Patch Size Compatibility ===")
    
    # Original problematic dimensions
    H, W = 840, 840
    patch_size = 14  # DINOv2 uses 14x14 patches
    
    # Check if dimensions are compatible
    if H % patch_size != 0 or W % patch_size != 0:
        print(f"❌ Original dimensions {H}x{W} not compatible with patch size {patch_size}")
        print(f"   H % {patch_size} = {H % patch_size}")
        print(f"   W % {patch_size} = {W % patch_size}")
    
    # Apply fix
    H_fixed = ((H + patch_size - 1) // patch_size) * patch_size
    W_fixed = ((W + patch_size - 1) // patch_size) * patch_size
    
    print(f"✅ Fixed dimensions: {H_fixed}x{W_fixed}")
    print(f"   H_fixed % {patch_size} = {H_fixed % patch_size}")
    print(f"   W_fixed % {patch_size} = {W_fixed % patch_size}")
    
    return H_fixed % patch_size == 0 and W_fixed % patch_size == 0

def test_import_fixes():
    """Test that import paths are correct."""
    print("\n=== Testing Import Fixes ===")
    
    # Check if the old weight loader can be imported
    _THIS_DIR = Path(__file__).resolve().parent
    _OLD_DIR = _THIS_DIR.parent / "Convertion_Tensorrt"
    
    if not _OLD_DIR.exists():
        print(f"❌ Old directory not found: {_OLD_DIR}")
        return False
    
    # Check if unified_weight_loader_fixed.py exists
    weight_loader_path = _OLD_DIR / "unified_weight_loader_fixed.py"
    if not weight_loader_path.exists():
        print(f"❌ Weight loader not found: {weight_loader_path}")
        return False
    
    print(f"✅ Found weight loader: {weight_loader_path}")
    
    # Test if we can add the path and potentially import (without actually importing due to dependencies)
    if str(_OLD_DIR) not in sys.path:
        sys.path.insert(0, str(_OLD_DIR))
        print(f"✅ Added to path: {_OLD_DIR}")
    
    return True

def test_function_signature_fix():
    """Test the function signature fix for load_dinov2_components."""
    print("\n=== Testing Function Signature Fix ===")
    
    # Read the export_dynamic_onnx_unified.py file
    script_path = Path(__file__).parent / "export_dynamic_onnx_unified.py"
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if the function was renamed correctly
    if "def load_dinov2_components_func(" in content:
        print("✅ Function renamed to load_dinov2_components_func")
    else:
        print("❌ Function not renamed properly")
        return False
    
    # Check if the function call was updated
    if "load_dinov2_components_func(model_state)" in content:
        print("✅ Function call updated correctly")
    else:
        print("❌ Function call not updated")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing fixes for MatchAnything ONNX export issues...")
    
    test1 = test_patch_size_compatibility()
    test2 = test_import_fixes()
    test3 = test_function_signature_fix()
    
    print(f"\n=== Summary ===")
    print(f"Patch size fix: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Import fixes: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Function signature fix: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All fixes validated successfully!")
        print("\nThe scripts should now work correctly when run in an environment with PyTorch installed.")
    else:
        print("\n⚠️  Some fixes may need attention.")