# -*- coding: utf-8 -*-
"""
Preprocessing for MatchAnything (TRT encoder) — Python 3.8 compatible.
Ensures both images end up with IDENTICAL HxW (default: square OPT size, multiple of 14).
"""

import math
from typing import Tuple
import numpy as np
from PIL import Image
import cv2

MULT = 14  # ViT-L/14

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_trt_input(img_rgb_uint8, target_hw, out_dtype=np.float16):
    H, W = target_hw
    inter = cv2.INTER_AREA if (img_rgb_uint8.shape[0] > H or img_rgb_uint8.shape[1] > W) else cv2.INTER_CUBIC
    img = cv2.resize(img_rgb_uint8, (W, H), interpolation=inter)
    x = img.astype(np.float32) / 255.0                        # [H,W,3] in [0,1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD                    # normalize
    x = np.transpose(x, (2, 0, 1))                            # [3,H,W]
    x = x[None, ...]                                          # [1,3,H,W]
    # ensure C-contiguous + correct dtype for TRT
    x = np.ascontiguousarray(x, dtype=out_dtype)
    return x

def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_chw_fp32(img_rgb_uint8: np.ndarray) -> np.ndarray:
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x

def _scale_to_fit(h: int, w: int, target_h: int, target_w: int) -> Tuple[int, int]:
    """Keep aspect ratio, fit inside target."""
    scale = min(target_h / float(h), target_w / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return new_h, new_w

def letterbox_to_exact(img_rgb: np.ndarray,
                       target_hw: Tuple[int, int],
                       multiple: int = MULT) -> np.ndarray:
    """
    Letterbox to EXACT target size (pad to target, not just to multiple).
    Target must be multiples of 'multiple' (14 for ViT-L/14).
    """
    th, tw = target_hw
    assert th % multiple == 0 and tw % multiple == 0, "Target must be multiple of 14."

    h, w = img_rgb.shape[:2]
    nh, nw = _scale_to_fit(h, w, th, tw)

    # resize (PIL for quality)
    im = Image.fromarray(img_rgb).resize((nw, nh), Image.BICUBIC)
    out = np.zeros((th, tw, 3), dtype=np.uint8)

    # center pad
    top = (th - nh) // 2
    left = (tw - nw) // 2
    out[top:top + nh, left:left + nw] = np.array(im, dtype=np.uint8)
    return out

def prepare_pair_same_hw(path0, path1, target_hw):
    img0_bgr = cv2.imread(path0, cv2.IMREAD_COLOR)
    img1_bgr = cv2.imread(path1, cv2.IMREAD_COLOR)
    assert img0_bgr is not None and img1_bgr is not None, "Failed to read images"
    img0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    x0 = _to_trt_input(img0, target_hw, out_dtype=np.float16) # engine is FP16
    x1 = _to_trt_input(img1, target_hw, out_dtype=np.float16)
    # quick sanity debug:
    # print("x0 flags:", x0.flags['C_CONTIGUOUS'], x0.dtype, x0.shape)
    return x0, x1
