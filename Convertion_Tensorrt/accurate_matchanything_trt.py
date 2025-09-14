# Convertion_Tensorrt/accurate_matchanything_trt.py
from typing import Tuple
import torch
import torch.nn as nn

from encoders_trt_full import CNNandDinov2TRT
from gp_trt import GPHeadTRT  # your TRT-friendly GP head that outputs (warp_c, cert_c)

class AccurateMatchAnythingTRT(nn.Module):
    """
    Minimal TRT-friendly wrapper:
      - ROMA DINOv2 encoder (coarse features at 1/14).
      - Lightweight GP head to predict coarse warp + certainty.
    """
    def __init__(self, amp: bool = False):
        super().__init__()
        self.encoder = CNNandDinov2TRT(amp=amp)
        self.gp_head = GPHeadTRT()  # expects two coarse feature maps -> (2,Hc,Wc) & (1,Hc,Wc)
        self.amp = amp

    @torch.no_grad()
    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # image0, image1: [1,3,H,W] (dynamic sizes; each padded to ×14 inside encoder)
        feat0 = self.encoder(image0)["coarse"]   # [1,1024,Hc0,Wc0]
        feat1 = self.encoder(image1)["coarse"]   # [1,1024,Hc1,Wc1]

        # GP head is written to accept different Hc/Wc per image (no forced common resize)
        warp_c, cert_c = self.gp_head(feat0, feat1)  # warp [1,2,Hc,Wc], cert [1,1,Hc,Wc]
        return warp_c, cert_c


def export_accurate_matchanything_onnx(model: nn.Module, onnx_path: str, H: int = 840, W: int = 840) -> str:
    model.eval()
    device = next(model.parameters()).device
    dummy0 = torch.randn(1, 3, H, W, device=device)
    dummy1 = torch.randn(1, 3, H, W, device=device)

    torch.onnx.export(
        model,
        (dummy0, dummy1),
        onnx_path,
        input_names=["image0", "image1"],
        output_names=["warp_c", "cert_c"],
        dynamic_axes={
            "image0": {2: "H0", 3: "W0"},
            "image1": {2: "H1", 3: "W1"},
            "warp_c": {2: "Hc", 3: "Wc"},
            "cert_c": {2: "Hc", 3: "Wc"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return onnx_path
