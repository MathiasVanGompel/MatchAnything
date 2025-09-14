#!/usr/bin/env python3
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPMatchEncoderTRT(nn.Module):
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = float(beta)

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int) -> torch.Tensor:
        # ONNX-friendly L2 normalization (avoid linalg_vector_norm)
        eps = 1e-6
        denom = torch.clamp(torch.sum(x * x, dim=dim, keepdim=True), min=eps * eps).sqrt()
        return x / denom

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f0: [B,C,Ha,Wa], f1: [B,C,Hb,Wb]
        returns:
          warp_c: [B,Ha,Wa,2] in coarse pixel units of image1
          cert_c: [B,Ha,Wa]   in [0,1]
        """
        B, C, Ha, Wa = f0.shape
        _, _, Hb, Wb = f1.shape

        # [B,Na,C] and [B,Nb,C]
        a = f0.view(B, C, Ha * Wa).transpose(1, 2)
        b = f1.view(B, C, Hb * Wb).transpose(1, 2)
        a = self._l2norm(a, dim=2)
        b = self._l2norm(b, dim=2)

        # similarity + softmax over Nb (destination pixels)
        sim  = torch.bmm(a, b.transpose(1, 2)) * self.beta  # [B,Na,Nb]
        attn = F.softmax(sim, dim=2)

        # Build coords without meshgrid (ONNX-friendly)
        # xs: [Hb*Wb], ys: [Hb*Wb], coords: [Nb,2]
        ys = torch.arange(Hb, device=f0.device, dtype=f0.dtype).view(Hb, 1).expand(Hb, Wb).reshape(-1)
        xs = torch.arange(Wb, device=f0.device, dtype=f0.dtype).view(1, Wb).expand(Hb, Wb).reshape(-1)
        coords = torch.stack([xs, ys], dim=1)  # [Nb,2], float to match dtype

        # Weighted average of coords
        tgt = torch.bmm(attn, coords.unsqueeze(0).expand(B, -1, -1))  # [B,Na,2]
        # channels-first: [B, 2, Hc, Wc] to align with export & runner
        warp_c = tgt.view(B, Ha, Wa, 2).permute(0, 3, 1, 2).contiguous()
        # add channel dim: [B, 1, Hc, Wc]
        cert_c = attn.max(dim=2).values.view(B, 1, Ha, Wa)
        return warp_c, cert_c
