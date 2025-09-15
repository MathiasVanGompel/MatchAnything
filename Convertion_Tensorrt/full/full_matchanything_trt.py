#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoders_trt_full import DINOv2EncoderTRT
from gp_trt import GPMatchEncoderTRT

class FullMatchAnythingTRT(nn.Module):
    """
    End-to-end: (image0, image1) -> (warp_c, cert_c) all inside one graph.
    - image*: [B,3,H,W] in [0,1], H/W multiples of 14
    - warp_c: [B,2,H/14,W/14]  (x,y) on coarse grid of image1
    - cert_c: [B,1,H/14,W/14]
    """
    def __init__(self, input_hw: Tuple[int, int] = (518, 518), amp: bool = False, beta: float = 14.285714285714286):
        super().__init__()
        self.encoder = DINOv2EncoderTRT(amp=amp, input_hw=input_hw)
        self.matcher = GPMatchEncoderTRT(beta=beta)

    @torch.no_grad()
    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        f0 = self.encoder(image0)["coarse"]  # [B,1024,Hc,Wc]
        f1 = self.encoder(image1)["coarse"]
        warp_c, cert_c = self.matcher(f0, f1)
        return warp_c, cert_c
