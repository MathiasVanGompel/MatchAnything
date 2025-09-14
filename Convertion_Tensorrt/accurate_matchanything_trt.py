#!/usr/bin/env python3
"""Minimal MatchAnything model for TensorRT export.
Returns dense warp and certainty maps matching RoMa's interface."""
import torch
import torch.nn as nn

from encoders_trt_full import CNNandDinov2TRT
from gp_trt import GPMatchEncoderTRT

class AccurateMatchAnythingTRT(nn.Module):
    def __init__(self, amp: bool = False):
        super().__init__()
        self.encoder = CNNandDinov2TRT(amp=amp)
        self.matcher = GPMatchEncoderTRT(beta=10.0)

    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        feat0 = self.encoder(image0)["coarse"]
        feat1 = self.encoder(image1)["coarse"]
        warp_c, cert_c = self.matcher(feat0, feat1)
        return warp_c, cert_c

def export_accurate_matchanything_onnx(model: nn.Module, onnx_path: str, H: int = 840, W: int = 840):
    """Export the given model to ONNX format."""
    device = "cpu"
    model = model.to(device).eval()
    x0 = torch.rand(1, 3, H, W, device=device)
    x1 = torch.rand(1, 3, H, W, device=device)

    dynamic_axes = {
        "image0": {0: "B", 2: "H", 3: "W"},
        "image1": {0: "B", 2: "H", 3: "W"},
        "warp_c": {0: "B", 2: "Hc", 3: "Wc"},
        "cert_c": {0: "B", 2: "Hc", 3: "Wc"},
    }

    import onnx
    try:
        from onnx import external_data_utils
    except ImportError:
        from onnx import external_data_helper as external_data_utils

    torch.onnx.export(
        model,
        (x0, x1),
        onnx_path,
        input_names=["image0", "image1"],
        output_names=["warp_c", "cert_c"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    model_proto = onnx.load(onnx_path, load_external_data=True)
    external_data_utils.convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location=onnx_path + ".data",
        size_threshold=0,
    )
    onnx.save_model(model_proto, onnx_path, save_as_external_data=True)

    from pathlib import Path
    for shard in Path(onnx_path).parent.glob("onnx__*"):
        shard.unlink(missing_ok=True)

    return onnx_path

