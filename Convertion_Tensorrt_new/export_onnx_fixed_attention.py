#!/usr/bin/env python3
"""
Export ONNX for MatchAnything with ONNX-compatible attention fix.
"""

import os
import re
import inspect
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import onnx

try:
    import timm
except Exception:
    timm = None

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_THIS_DIR))

from onnx_compatible_attention import ONNXCompatibleAttention
from accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT


def patch_model_for_onnx(model):
    """
    Patch the model to replace MemEffAttention with ONNX-compatible version
    """
    print("[PATCH] Patching model for ONNX compatibility...")
    
    def replace_attention_recursive(module):
        for name, child in module.named_children():
            if child.__class__.__name__ == 'MemEffAttention':
                print(f"[PATCH] Replacing {name} (MemEffAttention) with ONNXCompatibleAttention")
                
                # Create replacement with same parameters
                new_attention = ONNXCompatibleAttention(
                    dim=child.qkv.in_features,
                    num_heads=child.num_heads,
                    qkv_bias=child.qkv.bias is not None,
                    proj_bias=child.proj.bias is not None,
                    attn_drop=child.attn_drop.p,
                    proj_drop=child.proj_drop.p,
                )
                
                # Copy weights
                new_attention.qkv.weight.data.copy_(child.qkv.weight.data)
                if child.qkv.bias is not None:
                    new_attention.qkv.bias.data.copy_(child.qkv.bias.data)
                new_attention.proj.weight.data.copy_(child.proj.weight.data)
                if child.proj.bias is not None:
                    new_attention.proj.bias.data.copy_(child.proj.bias.data)
                
                # Replace the module
                setattr(module, name, new_attention)
            else:
                replace_attention_recursive(child)
    
    replace_attention_recursive(model)
    print("[PATCH] ✅ Model patching completed")
    return model


def export_onnx(
    onnx_path: str,
    ckpt: str = "",
    H: int = 288,
    W: int = 288,
    load_dinov2_components: bool = True,
    verbose_export: bool = False,
    silence_cpp: bool = True,
):
    # Disable verbose output
    os.environ["PYTORCH_ONNX_VERBOSE"] = "0"
    os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"

    device = "cpu"
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()

    # CRITICAL: Patch the model for ONNX compatibility
    model = patch_model_for_onnx(model)

    # Load weights if checkpoint provided
    if ckpt and os.path.exists(ckpt):
        # Add your weight loading logic here
        print(f"[LOAD] Loading checkpoint: {ckpt}")
        # ... weight loading code ...
    else:
        print("[LOAD] No checkpoint provided, using random weights")

    # Adjust input dimensions to be compatible with patch size 16
    patch_size = 16
    H = ((H + patch_size - 1) // patch_size) * patch_size
    W = ((W + patch_size - 1) // patch_size) * patch_size
    print(f"[LOAD] Adjusted input dimensions to {H}x{W} (compatible with patch size {patch_size})")
    
    # Create dummy inputs
    x0 = torch.rand(1, 3, H, W, device=device)
    x1 = torch.rand(1, 3, H, W, device=device)

    # Test forward pass
    print("[TEST] Running dry run...")
    with torch.inference_mode():
        out = model(x0, x1)
        print(f"[TEST] ✅ Dry run successful: {list(out.keys())}")

    # Prepare output path
    out_path = Path(onnx_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Define dynamic axes
    dynamic_axes = {
        "image0": {"0": "B", "2": "H", "3": "W"},
        "image1": {"0": "B", "2": "H", "3": "W"},
        "warp_c": {"0": "B", "1": "Ha", "2": "Wa"},
        "cert_c": {"0": "B", "1": "Ha", "2": "Wa"},
        "valid_mask": {"0": "B", "1": "Ha", "2": "Wa"},
        "coarse_stride": {"0": "one"},
    }

    # Export to ONNX
    print("[ONNX] Starting ONNX export...")
    try:
        torch.onnx.export(
            model,
            (x0, x1),
            str(out_path),
            input_names=["image0", "image1"],
            output_names=["warp_c", "cert_c", "valid_mask", "coarse_stride"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
        
        # Verify file creation
        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024.0)
            print(f"[ONNX] ✅ Export successful: {out_path} ({size_mb:.1f} MB)")
            return str(out_path)
        else:
            raise FileNotFoundError(f"ONNX file not created: {out_path}")
            
    except Exception as e:
        print(f"[ONNX] ❌ Export failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export ONNX with fixed attention")
    parser.add_argument("--onnx", default="output/matchanything_fixed.onnx")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--H", type=int, default=288)
    parser.add_argument("--W", type=int, default=288)
    parser.add_argument("--verbose_export", action="store_true")
    args = parser.parse_args()

    export_onnx(
        onnx_path=args.onnx,
        ckpt=args.ckpt,
        H=args.H,
        W=args.W,
        verbose_export=args.verbose_export,
    )