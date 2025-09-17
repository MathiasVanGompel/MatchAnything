# Convertion_Tensorrt/encoders_trt_full.py
# -*- coding: utf-8 -*-
import importlib
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import os, sys

# Make top-level package importable no matter where we run from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _first_present_key(d: dict, keys) -> Optional[torch.Tensor]:
    """Return the first value in dict d that is not None among keys."""
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return None

# --- robust import helper -----------------------------------------------------
def _import_vit_large():
    """
    Try several likely module paths for RoMa's DINOv2 vit_large builder.
    Works with your repo layout and common forks.
    """
    candidates = [
        # submodule path (MatchAnything vendored RoMa)
        "imcui.third_party.MatchAnything.third_party.ROMA.roma.models.transformer.dinov2",
        # some forks rename 'roma' -> 'romatch'
        "imcui.third_party.MatchAnything.third_party.ROMA.romatch.models.transformer.dinov2",
        # direct vendor installs
        "roma.models.transformer.dinov2",
        "romatch.models.transformer.dinov2",
    ]
    last_err = None
    for m in candidates:
        try:
            return importlib.import_module(m).vit_large
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import RoMa vit_large; tried {candidates}") from last_err


# --- DINOv2-L/14 builder (RoMa-compatible) ------------------------------------
def build_dinov2_vitl14_romacfg(img_size: int = 518, block_chunks: int = 0) -> nn.Module:
    """
    Build the RoMa DINOv2 ViT-L/14 backbone.
    weights are injected by unified_weight_loader to keep memory low & names consistent.
    - img_size should be 518 for 37x37 patches + CLS = 1370 tokens in RoMa demos.
    - block_chunks controls whether blocks are grouped (affects key naming). We default to 0.
      (Some RoMa/DINOv2 builds use chunking that yields keys like blocks.0.0.*)  # ref: issues show both variants
    """
    vit_large = _import_vit_large()
    model = vit_large(
        img_size=img_size,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=block_chunks,  # 0 = flat (blocks.N.*). Chunked => (blocks.0.N.*)
    )
    model.eval()
    return model


# --- TRT-friendly encoder wrapper ---------------------------------------------
class DINOv2EncoderTRT(nn.Module):
    """
    Wrap DINOv2 so we always keep dtype-consistency and ONNX-traceable control-flow.
    - amp=False (default): FP32 throughout → safest ONNX export.
    - amp=True : casts module params + inputs to half.

    Returns {'coarse': [B, 1024, H/14, W/14]} for ViT-L/14.
    """

    def __init__(
        self,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        input_hw: Tuple[int, int] = (518, 518),
        block_chunks: int = 0,
    ):
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.input_hw = input_hw
        self.dino = build_dinov2_vitl14_romacfg(img_size=input_hw[0], block_chunks=block_chunks)
        if self.amp:
            self.dino.half()  # align params/biases with FP16 inputs

        # cache for ONNX-friendly constants
        self.patch = 14

    def _cast_input_to_encoder_dtype(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to weight dtype (prevents Half/Float mismatch in Conv2d)
        w = self.dino.patch_embed.proj.weight
        return x.to(dtype=w.dtype) if x.dtype != w.dtype else x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, 1|3, H, W] in [0,1]. Ensures 3 channels and correct dtype.
        Returns: {'coarse': [B, C, H/14, W/14]} with C=1024 for ViT-L/14.
        """
        B, C, H, W = x.shape

        # grayscale → 3ch (trace-friendly)
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # H,W must be multiples of patch size (14)
        assert H % self.patch == 0 and W % self.patch == 0, "H and W must be multiples of 14."
        Hc, Wc = H // self.patch, W // self.patch

        # enforce dtype to match encoder weights; avoid autocast in ONNX export
        x = self._cast_input_to_encoder_dtype(x)
        out = self.dino.forward_features(x)

        # DINOv2 forward_features returns a dict in RoMa; prefer normalized patch tokens.
        preferred_keys = [
            "x_norm_patchtokens",  # RoMa / DINOv2 normalized patch tokens
            "patch_tokens",
            "tokens",
            "last_hidden_state",
            "x",
        ]
        if isinstance(out, dict):
            seq = _first_present_key(out, preferred_keys)
        else:
            seq = out  # some forks return the tensor directly

        if seq is None:
            raise RuntimeError(
                "Unexpected DINOv2 forward_features output format. "
                f"Available keys: {list(out.keys()) if isinstance(out, dict) else type(out)}"
            )

        # reshape to [B,C,Hc,Wc]
        if seq.dim() == 3:
            # seq: [B, N, Ctok] (may include CLS → N == Hc*Wc + 1). Take the last Hc*Wc tokens.
            B_, N_, Ctok = seq.shape
            tokens = seq[:, -(Hc * Wc):, :]
            fmap = tokens.transpose(1, 2).contiguous().view(B_, Ctok, Hc, Wc)
        elif seq.dim() == 4:
            fmap = seq.contiguous()
        else:
            raise RuntimeError(f"Unsupported token tensor shape: {tuple(seq.shape)}")

        return {"coarse": fmap}
