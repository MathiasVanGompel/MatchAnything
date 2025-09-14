# Convertion_Tensorrt/encoders_trt_full.py
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Use the ROMA DINOv2 implementation that the Space ships with
# https://huggingface.co/spaces/LittleFrog/MatchAnything/tree/main/imcui/third_party/MatchAnything/third_party/ROMA
from imcui.third_party.MatchAnything.third_party.ROMA.roma.models.transformer.dinov2 import vit_large as roma_vit_large

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _to_3ch(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W] on (cuda|cpu)
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x

def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = x.new_tensor(IMAGENET_STD ).view(1, 3, 1, 1)
    return (x - mean) / std

def _pad_to_multiple(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    # Pad H,W up to next multiple of `mult` using ONNX-friendly ops
    _, _, H, W = x.shape
    pad_h = (mult - (H % mult)) % mult
    pad_w = (mult - (W % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    # F.pad uses (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (0, pad_w, 0, pad_h)

class CNNandDinov2TRT(nn.Module):
    """
    ROMA DINOv2 ViT-L/14 backbone that returns a 'coarse' feature map
    with shape [B, 1024, H/14, W/14], matching ROMA's expectations.
    """
    def __init__(self, amp: bool = False, coarse_patch_size: int = 14):
        super().__init__()
        self.amp = bool(amp)
        self.amp_dtype = torch.float16 if amp else torch.float32
        self.patch = int(coarse_patch_size)

        # Instantiate ROMA's DINOv2 ViT-L/14; ROMA handles pos-embed interpolation internally.
        # These kwargs mirror the ROMA encoders.py instantiation.
        self.dino = roma_vit_large(
            patch_size=self.patch,
            img_size=518,        # matches ROMA config; not baked as a hard limit thanks to interpolate_pos_encoding
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        ).eval()

        # The ROMA DINOv2 weights are loaded by the higher-level model;
        # this module expects state_dict to be populated before export/infer.

        # No trainable params here for export
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B,3,H,W] in float32/float16, device = model device
        x = _to_3ch(x)
        x = _imagenet_norm(x)

        if self.amp:
            x = x.to(self.amp_dtype)

        # ROMA PatchEmbed asserts H,W multiples of patch; do an ONNX-friendly pad if needed
        x, _ = _pad_to_multiple(x, self.patch)

        # ROMA DINOv2 forward_features returns dict with 'x_norm_patchtokens' [B, N, 1024]
        out = self.dino.forward_features(x)
        tokens = out["x_norm_patchtokens"]  # [B, N, 1024]

        B, N, C = tokens.shape
        # Recover (Hc, Wc) from token count with known patch size
        # N = Hc * Wc
        # We infer Hc, Wc from the padded input spatial size
        _, _, Hp, Wp = x.shape
        Hc = Hp // self.patch
        Wc = Wp // self.patch

        # Safety: avoid implicit Python ints during trace
        tokens = tokens.permute(0, 2, 1).contiguous()           # [B,1024,N]
        feats16 = tokens.view(B, C, Hc, Wc)                     # [B,1024,H/14,W/14]

        return {"coarse": feats16}
