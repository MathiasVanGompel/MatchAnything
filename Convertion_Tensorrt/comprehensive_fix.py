#!/usr/bin/env python3
"""
Comprehensive fix for all ONNX export and memory issues.
This addresses both the anti-aliased bicubic problem and VS Code crashes.
"""
import os
import sys
import subprocess
from pathlib import Path

def find_and_patch_all_bicubic():
    """
    Find and patch ALL bicubic interpolations in the ROMA codebase.
    """
    roma_base = Path(__file__).parent / "../imcui/third_party/MatchAnything/third_party/ROMA"
    
    if not roma_base.exists():
        print(f"[ERROR] ROMA directory not found: {roma_base}")
        return False
    
    print(f"[PATCH] Searching for bicubic interpolations in: {roma_base}")
    
    # Find all Python files
    python_files = list(roma_base.rglob("*.py"))
    patched_files = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Check if file contains bicubic interpolation
            if 'mode="bicubic"' in content or "mode='bicubic'" in content:
                print(f"[PATCH] Found bicubic in: {py_file}")
                
                # Apply patches
                original_content = content
                
                # Patch 1: mode="bicubic" without antialias
                content = content.replace(
                    'mode="bicubic"',
                    'mode="bicubic", antialias=False'
                )
                
                # Patch 2: mode='bicubic' without antialias  
                content = content.replace(
                    "mode='bicubic'",
                    "mode='bicubic', antialias=False"
                )
                
                # Patch 3: mode="bicubic", align_corners=False without antialias
                content = content.replace(
                    'mode="bicubic", align_corners=False',
                    'mode="bicubic", align_corners=False, antialias=False'
                )
                
                # Patch 4: mode='bicubic', align_corners=False without antialias
                content = content.replace(
                    "mode='bicubic', align_corners=False",
                    "mode='bicubic', align_corners=False, antialias=False"
                )
                
                # Only write if content changed
                if content != original_content:
                    with open(py_file, 'w') as f:
                        f.write(content)
                    patched_files.append(str(py_file))
                    print(f"[PATCH] ✅ Patched: {py_file}")
                else:
                    print(f"[PATCH] ⚠️ No changes needed: {py_file}")
                    
        except Exception as e:
            print(f"[PATCH] ❌ Error processing {py_file}: {e}")
    
    print(f"[PATCH] Summary: Patched {len(patched_files)} files")
    for pf in patched_files:
        print(f"  - {pf}")
    
    return len(patched_files) > 0

def clean_python_cache():
    """Clean all Python cache files to ensure patches take effect."""
    roma_base = Path(__file__).parent / "../imcui/third_party/MatchAnything/third_party/ROMA"
    
    if not roma_base.exists():
        print(f"[CLEAN] ROMA directory not found: {roma_base}")
        return
    
    print("[CLEAN] Cleaning Python cache files...")
    
    # Find and remove __pycache__ directories
    pycache_dirs = list(roma_base.rglob("__pycache__"))
    for cache_dir in pycache_dirs:
        try:
            subprocess.run(['rm', '-rf', str(cache_dir)], check=True)
            print(f"[CLEAN] ✅ Removed: {cache_dir}")
        except Exception as e:
            print(f"[CLEAN] ❌ Error removing {cache_dir}: {e}")
    
    # Find and remove .pyc files
    pyc_files = list(roma_base.rglob("*.pyc"))
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print(f"[CLEAN] ✅ Removed: {pyc_file}")
        except Exception as e:
            print(f"[CLEAN] ❌ Error removing {pyc_file}: {e}")
    
    print(f"[CLEAN] Cleaned {len(pycache_dirs)} cache dirs and {len(pyc_files)} .pyc files")

def create_memory_efficient_export():
    """
    Create a memory-efficient ONNX export script to prevent VS Code crashes.
    """
    script_content = '''#!/usr/bin/env python3
"""
Memory-efficient ONNX export to prevent VS Code crashes.
This uses reduced workspace and batch size to minimize memory usage.
"""
import os
import gc
import torch
import psutil
from pathlib import Path

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"[MEMORY] RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB")
    return memory_mb, gpu_memory

def memory_efficient_export(onnx_path, model_name="matchanything_roma", ckpt_path=None):
    """Export with memory management"""
    print("="*60)
    print("MEMORY-EFFICIENT ONNX EXPORT")
    print("="*60)
    
    # Monitor initial memory
    monitor_memory()
    
    # Import with memory management
    print("[STEP 1] Importing modules...")
    from real_matchanything_trt import RealMatchAnythingTRT
    
    # Create model with minimal memory footprint
    print("[STEP 2] Creating model...")
    device = "cpu"  # Use CPU to avoid GPU memory issues
    model = RealMatchAnythingTRT(
        model_name=model_name,
        img_resize=832,
        match_threshold=0.1
    ).to(device).eval()
    
    monitor_memory()
    
    # Load checkpoint if provided
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[STEP 3] Loading checkpoint: {ckpt_path}")
        success = model.load_checkpoint(ckpt_path)
        if success:
            print("[SUCCESS] Checkpoint loaded")
        else:
            print("[WARNING] Checkpoint loading had issues")
    
    monitor_memory()
    
    # Create smaller dummy inputs to reduce memory
    print("[STEP 4] Creating dummy inputs...")
    H, W = 416, 416  # Use smaller size for export to save memory
    x1 = torch.rand(1, 3, H, W, device=device)
    x2 = torch.rand(1, 3, H, W, device=device)
    
    # Test forward pass
    print("[STEP 5] Testing forward pass...")
    with torch.no_grad():
        try:
            result = model(x1, x2)
            print(f"Dry run OK: {len(result['mconf'])} matches found")
        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            return False
    
    monitor_memory()
    
    # Export with memory-efficient settings
    print("[STEP 6] Exporting to ONNX...")
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    
    try:
        torch.onnx.export(
            model, (x1, x2), onnx_path,
            input_names=["image0", "image1"],
            output_names=["keypoints0", "keypoints1", "mconf"],
            dynamic_axes={
                "image0": {0: "B", 2: "H", 3: "W"},
                "image1": {0: "B", 2: "H", 3: "W"},
                "keypoints0": {0: "num_matches"},
                "keypoints1": {0: "num_matches"}, 
                "mconf": {0: "num_matches"},
            },
            opset_version=16,  # Use lower opset for better compatibility
            do_constant_folding=True,
            verbose=False,  # Reduce output to prevent VS Code overload
            enable_onnx_checker=False  # Disable checker to save memory
        )
        print(f"[SUCCESS] ONNX exported to: {onnx_path}")
        
        # Check file size
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"[INFO] ONNX file size: {size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False
    
    finally:
        # Clean up memory
        del model, x1, x2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        monitor_memory()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="out/memory_efficient_matchanything.onnx")
    parser.add_argument("--model", default="matchanything_roma")
    parser.add_argument("--ckpt", help="Checkpoint path")
    args = parser.parse_args()
    
    success = memory_efficient_export(args.onnx, args.model, args.ckpt)
    sys.exit(0 if success else 1)
'''
    
    with open("memory_efficient_export.py", 'w') as f:
        f.write(script_content)
    
    print("[CREATED] memory_efficient_export.py - Use this to avoid VS Code crashes")

def main():
    print("="*60)
    print("COMPREHENSIVE MATCHANYTHING TENSORRT FIX")
    print("="*60)
    
    # Step 1: Patch all bicubic interpolations
    print("\n[STEP 1] Patching all bicubic interpolations...")
    patch_success = find_and_patch_all_bicubic()
    
    # Step 2: Clean Python cache
    print("\n[STEP 2] Cleaning Python cache...")
    clean_python_cache()
    
    # Step 3: Create memory-efficient export script
    print("\n[STEP 3] Creating memory-efficient export script...")
    create_memory_efficient_export()
    
    print("\n" + "="*60)
    print("FIX COMPLETE")
    print("="*60)
    
    if patch_success:
        print("✅ Successfully patched bicubic interpolations")
    else:
        print("⚠️ No bicubic interpolations found to patch")
    
    print("\nNext steps:")
    print("1. Use memory_efficient_export.py to avoid VS Code crashes:")
    print("   python3 memory_efficient_export.py --ckpt /path/to/checkpoint.ckpt")
    print()
    print("2. Or run build script outside VS Code:")
    print("   # Run in terminal (not VS Code) to avoid crashes")
    print("   ./build_real_matchanything_trt.sh --ckpt /path/to/checkpoint.ckpt")
    print()
    print("3. Reduce TensorRT workspace to prevent memory issues:")
    print("   ./build_real_matchanything_trt.sh --workspace 1024 --ckpt /path/to/checkpoint.ckpt")

if __name__ == "__main__":
    main()