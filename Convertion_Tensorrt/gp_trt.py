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
        eps = 1e-6
        # torch.norm exports to ONNX as linalg_vector_norm, which is unsupported
        # on TensorRT. Compute the L2 norm manually using basic ops that export
        # cleanly: sqrt(sum(x * x)) instead of torch.norm.
        norm = torch.sqrt(torch.sum(x * x, dim=dim, keepdim=True))
        return x / torch.clamp(norm, min=eps)

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f0: [B,C,Ha,Wa], f1: [B,C,Hb,Wb]
        returns:
          warp_c: [B,Ha,Wa,2] in coarse pixel units of image1
          cert_c: [B,Ha,Wa]   in [0,1]
        """
        B, C, Ha, Wa = f0.shape
        _, _, Hb, Wb  = f1.shape

        a = f0.view(B, C, Ha * Wa).transpose(1, 2)   # [B, Na, C]
        b = f1.view(B, C, Hb * Wb).transpose(1, 2)   # [B, Nb, C]
        a = self._l2norm(a, dim=2)
        b = self._l2norm(b, dim=2)

        sim  = torch.bmm(a, b.transpose(1, 2)) * self.beta   # [B, Na, Nb]
        attn = F.softmax(sim, dim=2)

        ys = torch.arange(Hb, device=f0.device, dtype=f0.dtype).unsqueeze(1)
        xs = torch.arange(Wb, device=f0.device, dtype=f0.dtype).unsqueeze(0)
        yy = ys.expand(Hb, Wb)
        xx = xs.expand(Hb, Wb)
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [Nb,2]

        tgt = torch.bmm(attn, coords.unsqueeze(0).expand(B, -1, -1))  # [B,Na,2]
        warp_c = tgt.view(B, Ha, Wa, 2)

        cert_c = attn.max(dim=2).values.view(B, Ha, Wa)
        return warp_c, cert_c
