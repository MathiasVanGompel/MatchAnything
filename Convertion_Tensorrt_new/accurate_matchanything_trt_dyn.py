#!/usr/bin/env python3
"""
Accurate MatchAnything (ROMA-style) for ONNX/TensorRT (dynamic HxW).
- Pads inside the graph to the next multiple of encoder.patch (e.g., 14).
- Returns dense fields: warp_c [B,Ha,Wa,2], cert_c [B,Ha,Wa]
- Also returns: valid_mask [B,Ha,Wa] (1 real, 0 padded), coarse_stride [1] (float)
- No in-graph thresholding; host filters/top-K.
"""

from typing import Dict, Tuple
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- allow importing modules from your old folder (Convertion_Tensorrt) ---
_THIS_DIR = Path(__file__).resolve().parent
_OLD_DIR = _THIS_DIR.parent / "Convertion_Tensorrt"
if _OLD_DIR.exists() and str(_OLD_DIR) not in sys.path:
    sys.path.insert(0, str(_OLD_DIR))
# -------------------------------------------------------------------------

from encoders_trt_full import CNNandDinov2TRT
from gp_trt import GPMatchEncoderTRT


def _pad_to_multiple(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, int, int]:
    """Right/bottom pad to make H,W multiples of mult. Return (padded, pad_h, pad_w)."""
    B, C, H, W = x.shape
    pad_h = (mult - (H % mult)) % mult
    pad_w = (mult - (W % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    # pad format: (left, right, top, bottom)
    return F.pad(x, (0, pad_w, 0, pad_h)), pad_h, pad_w


class AccurateMatchAnythingTRT(nn.Module):
    def __init__(self, amp: bool = False):
        super().__init__()
        self.encoder = CNNandDinov2TRT(amp=amp)  # has .patch (your enc asserts 16 for DINOv2)
        self.matcher = GPMatchEncoderTRT(beta=10.0)  # ROMA-style GP matcher

    def forward(
        self, image0: torch.Tensor, image1: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Inputs: image0, image1 -> [B,3,H,W] RGB float in [0,1], any H,W
        Outputs:
            warp_c        [B,Ha,Wa,2]  coarse coords (x,y) in img1
            cert_c        [B,Ha,Wa]
            valid_mask    [B,Ha,Wa]    1 for real image area, 0 for padded
            coarse_stride [1]  (float) encoder patch/stride (e.g., 14.)
        """
        B, C, H, W = image0.shape
        patch = int(getattr(self.encoder, "patch", 16))

        # RGB->Gray inside graph (ONNX-friendly)
        img0_gray = (
            0.299 * image0[:, 0:1] + 0.587 * image0[:, 1:2] + 0.114 * image0[:, 2:3]
        )
        img1_gray = (
            0.299 * image1[:, 0:1] + 0.587 * image1[:, 1:2] + 0.114 * image1[:, 2:3]
        )

        # Pad to multiple-of-patch
        img0p, _, _ = _pad_to_multiple(img0_gray, patch)
        img1p, _, _ = _pad_to_multiple(img1_gray, patch)

        # Coarse features (stride=patch)
        feat0 = self.encoder(img0p)["coarse"]  # [B,Cc,Ha,Wa]
        feat1 = self.encoder(img1p)["coarse"]  # [B,Cc,Ha,Wa]

        # Dense matching
        warp_c, cert_c = self.matcher(feat0, feat1)  # [B,Ha,Wa,2], [B,Ha,Wa]
        _, Ha, Wa, _ = warp_c.shape

        # Valid coarse mask for original extent (ignore padded coarse cells)
        yy = torch.arange(Ha, device=warp_c.device).view(1, Ha, 1).float()
        xx = torch.arange(Wa, device=warp_c.device).view(1, 1, Wa).float()
        valid_y = (yy * patch) < float(H)
        valid_x = (xx * patch) < float(W)
        valid_mask = (valid_y & valid_x).to(cert_c.dtype).expand(B, Ha, Wa)

        # Stride as tensor (export-friendly)
        coarse_stride = torch.tensor(
            [float(patch)], device=warp_c.device, dtype=warp_c.dtype
        )

        return {
            "warp_c": warp_c,
            "cert_c": cert_c,
            "valid_mask": valid_mask,
            "coarse_stride": coarse_stride,
        }
