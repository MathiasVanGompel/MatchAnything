#!/usr/bin/env python3
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders_trt_full import CNNandDinov2TRT
from gp_trt import GPMatchEncoderTRT

class RoMaTRTCoreFull(nn.Module):
    """
    Outputs:
      warp: (B, 4, H, W)  = (x0_norm, y0_norm, x1_norm, y1_norm) in [-1,1]
      cert: (B, 1, H, W)  = certainty map in [0,1]
    """
    def __init__(self, amp: bool = False, beta: float = 10.0):
        super().__init__()
        self.encoder = CNNandDinov2TRT(amp=amp)         # {"coarse": [B,1024,H/14,W/14]}
        self.gp      = GPMatchEncoderTRT(beta=beta)     # (warp_c[B,Ha,Wa,2], cert_c[B,Ha,Wa])

    @staticmethod
    def _norm_from_pix(x_pix: torch.Tensor, size: int) -> torch.Tensor:
        return (x_pix * (2.0 / max(size - 1, 1.0))) - 1.0

    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = image0.shape
        f0 = self.encoder(image0)["coarse"]
        f1 = self.encoder(image1)["coarse"]
        _, _, Ha, Wa = f0.shape

        warp_c, cert_c = self.gp(f0, f1)  # [B,Ha,Wa,2], [B,Ha,Wa]

        # source grid (A) in coarse pixels
        with torch.no_grad():
            ys = torch.arange(Ha, device=f0.device, dtype=f0.dtype)
            xs = torch.arange(Wa, device=f0.device, dtype=f0.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            src_c = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B,Ha,Wa,2]

        # upsample to full-res
        tgt_full  = F.interpolate(warp_c.permute(0,3,1,2), size=(H,W), mode="bilinear", align_corners=False)
        src_full  = F.interpolate(src_c.permute(0,3,1,2),  size=(H,W), mode="bilinear", align_corners=False)
        cert_full = F.interpolate(cert_c.unsqueeze(1),     size=(H,W), mode="bilinear", align_corners=False)

        sx = float(W) / float(max(Wa, 1)); sy = float(H) / float(max(Ha, 1))
        x0_pix = src_full[:, 0] * sx; y0_pix = src_full[:, 1] * sy
        x1_pix = tgt_full[:, 0] * sx; y1_pix = tgt_full[:, 1] * sy

        warp = torch.stack([
            self._norm_from_pix(x0_pix, W),
            self._norm_from_pix(y0_pix, H),
            self._norm_from_pix(x1_pix, W),
            self._norm_from_pix(y1_pix, H),
        ], dim=1)  # [B,4,H,W]
        cert = torch.clamp(cert_full, 0.0, 1.0)  # [B,1,H,W]
        return warp, cert