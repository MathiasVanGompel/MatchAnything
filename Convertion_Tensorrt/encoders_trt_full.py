#!/usr/bin/env python3
from typing import Dict, Optional, Tuple
import math, sys, torch
import torch.nn as nn
import torch.nn.functional as F

# --- make your ROMA repo importable ---
import os
ROMA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imcui", "third_party", "MatchAnything", "third_party", "ROMA")
if ROMA_ROOT not in sys.path:
    sys.path.append(ROMA_ROOT)

from roma.models.transformer.dinov2 import vit_large  # ROMA DINOv2 ViT-L/14

__all__ = ["CNNandDinov2TRT"]

def _first_present(d: Dict[str, torch.Tensor], keys) -> Optional[torch.Tensor]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _tokens_from_dino(out) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(out, torch.Tensor):
        return out, None
    if not isinstance(out, dict):
        raise TypeError(f"DINO forward_features returned {type(out)}")
    patch = _first_present(out, ["x_norm_patchtokens", "patch_tokens", "last_hidden_state", "x_prenorm"])
    if patch is None:
        raise KeyError("Cannot find patch tokens in DINO output")
    cls = _first_present(out, ["x_norm_clstoken", "cls_token", "pooler_output"])
    if cls is not None and cls.dim() == 3 and cls.size(1) == 1:
        cls = cls[:, 0, :]
    return patch, cls

def _patch_interpolate_pos_encoding(dino_module: nn.Module, patch_size: int = 14):
    """
    Monkey-patch DINO's interpolate_pos_encoding to be ONNX-safe and to return
    PATCH-ONLY positions (no CLS). Use integer `size=(gh,gw)` instead of
    Tensor scale_factors to avoid ONNX export issues with bicubic. :contentReference[oaicite:1]{index=1}
    """
    if not hasattr(dino_module, "interpolate_pos_encoding"):
        return

    pos_embed = dino_module.pos_embed  # [1, 1+N_pre, C]
    dim = int(pos_embed.shape[-1])

    def safe_interp(x: torch.Tensor, w: torch.Tensor, h: torch.Tensor):
        """
        x : [B, N, C]  (PATCH TOKENS ONLY; no CLS)
        return: [B, N, C] patch PE resized to current grid
        """
        # Pretrain grid side from pos_embed (strip CLS):
        pe = pos_embed[:, 1:, :]  # [1, N_pre, C]
        n_pre = int(pe.shape[1])
        side_pre = int(round(math.sqrt(max(n_pre, 1))))
        pe_4d = pe.reshape(1, side_pre, side_pre, dim).permute(0, 3, 1, 2)  # [1,C,Hpre,Wpre]

        # Current token count (N) is known; for square inputs N = (H/patch)*(W/patch)
        n_now = int(x.shape[1])
        side_now = int(round(math.sqrt(max(n_now, 1))))
        gh = side_now
        gw = max(1, n_now // max(1, gh))  # works for rectangular too

        # ONNX-safe: pass integer size (gh,gw), NOT Tensor scale_factors
        pe_resized = F.interpolate(pe_4d, size=(int(gh), int(gw)),
                                   mode="bicubic", align_corners=False)
        pe_tokens = pe_resized.permute(0, 2, 3, 1).reshape(1, gh * gw, dim)  # [1,N,C]
        # Broadcast to batch
        return pe_tokens.expand(x.shape[0], -1, -1)

    dino_module.interpolate_pos_encoding = safe_interp  # type: ignore

class CNNandDinov2TRT(nn.Module):
    """
    DINOv2 ViT-L/14 (ROMA) -> 'coarse' feature map [B,1024,H/14,W/14]
    """
    def __init__(self, amp: bool = False, use_cls_token: bool = False, out_channels: int = 1024):
        super().__init__()
        self.amp = bool(amp)
        self.use_cls = bool(use_cls_token)
        self.amp_dtype = torch.float16 if amp else torch.float32
        self.patch = 14

        self.dino = vit_large()  # expects ImageNet-normalized RGB
        _patch_interpolate_pos_encoding(self.dino, patch_size=self.patch)  # <-- fixed here

        self.proj = nn.Identity()
        if out_channels != 1024:
            self.proj = nn.Conv2d(1024, out_channels, kernel_size=1, bias=False)

    @staticmethod
    def _normalize_im(x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std  = torch.as_tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        if (H % self.patch) != 0 or (W % self.patch) != 0:
            raise AssertionError(f"H,W must be multiples of {self.patch}, got {H}x{W}")

        x_in = self._normalize_im(x)
        if self.amp: x_in = x_in.to(self.amp_dtype)

        out = self.dino.forward_features(x_in)
        tokens, cls = _tokens_from_dino(out)  # tokens are PATCH ONLY

        N = tokens.size(1)
        gh, gw = H // self.patch, W // self.patch  # for 448 & p=14: 32×32 => N=1024
        if N != gh * gw:
            # fallback for odd shapes
            side = int(round(math.sqrt(max(N, 1))))
            if side * side == N:
                gh = side; gw = side
            elif (N % gw) == 0:
                gh = N // gw
            elif (N % gh) == 0:
                gw = N // gh
            else:
                raise RuntimeError(f"Cannot reshape tokens (N={N}) to grid {gh}x{gw}")

        feat = tokens.reshape(B, gh, gw, tokens.size(2)).permute(0, 3, 1, 2).contiguous()  # [B,1024,gh,gw]
        feat = self.proj(feat)
        outd = {"coarse": feat}
        if self.use_cls and cls is not None:
            outd["cls"] = cls.to(feat.dtype)
        return outd