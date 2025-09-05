#!/usr/bin/env python3
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
