import torch
import torch.nn as nn
from typing import Tuple
from encoders_trt_full import DINOv2EncoderTRT

class AccurateMatchAnythingTRT(nn.Module):
    """
    Minimal export wrapper that mirrors the "accurate" MatchAnything_roma encoder behavior.
    It returns the coarse ViT-L/14 features for both input images (stride 14),
    ready for a downstream matcher (e.g., RoMa's decoder).
    """
    def __init__(self, input_hw: Tuple[int, int] = (518, 518), amp: bool = False):
        super().__init__()
        self.encoder = DINOv2EncoderTRT(amp=amp, input_hw=input_hw)

    @torch.no_grad()
    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        """
        image*: [B, 1|3, H, W] in [0,1], H/W multiples of 14 (e.g., 518).
        Returns 2 tensors: f0, f1 with shape [B, 1024, H/14, W/14].
        """
        f0 = self.encoder(image0)["coarse"]
        f1 = self.encoder(image1)["coarse"]
        return f0, f1


def export_accurate_matchanything_onnx(
    model: AccurateMatchAnythingTRT,
    onnx_path: str,
    H: int,
    W: int,
    opset: int = 17,
):
    model.eval()
    device = next(model.parameters()).device
    dummy0 = torch.rand(1, 3, H, W, device=device, dtype=torch.float32)
    dummy1 = torch.rand(1, 3, H, W, device=device, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy0, dummy1),
        onnx_path,
        opset_version=opset,
        input_names=["image0", "image1"],
        output_names=["f0", "f1"],
        dynamic_axes={
            "image0": {0: "B", 2: "H", 3: "W"},
            "image1": {0: "B", 2: "H", 3: "W"},
            "f0": {0: "B"},
            "f1": {0: "B"},
        },
    )
    return onnx_path
