#!/usr/bin/env python3
from typing import Dict, Optional, Tuple
import math, sys, torch
import torch.nn as nn
import torch.nn.functional as F

# --- make your ROMA repo importable ---
import os
# Try multiple possible ROMA paths
ROMA_PATHS = [
    "/home/mathias/MatchAnything-1/imcui/third_party/MatchAnything/third_party/ROMA",  # Your specific path
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "imcui", "third_party", "MatchAnything", "third_party", "ROMA"),  # Relative path
    os.environ.get("ROMA_ROOT", ""),  # Environment variable
]

ROMA_ROOT = None
for path in ROMA_PATHS:
    if path and os.path.exists(path):
        ROMA_ROOT = path
        break

if ROMA_ROOT is None:
    raise RuntimeError(f"ROMA not found in any of these paths: {ROMA_PATHS}")

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
    # Expect dict with patch tokens + optional cls token (ROMA style)
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

def _best_factor_pair(n: int) -> Tuple[int, int]:
    """
    Find factors (gh,gw) such that gh*gw = n, preferring near-square.
    Guarantees an exact product; avoids using runtime tensors or floats.
    """
    s = int(math.sqrt(max(n, 1)))
    for d in range(s, 0, -1):
        if n % d == 0:
            return d, n // d
    return n, 1  # fallback (shouldn't happen)

def _patch_interpolate_pos_encoding(dino_module: nn.Module, patch_size: int = 14):
    """
    Monkey-patch DINO's interpolate_pos_encoding to be ONNX-safe and to return
    CLS+PATCH positions (same length as x). Use integer size=(gh,gw) instead of
    Tensor scale_factors to avoid ONNX export issues with bicubic. :contentReference[oaicite:2]{index=2}
    """
    if not hasattr(dino_module, "interpolate_pos_encoding"):
        return

    pos_embed = dino_module.pos_embed  # [1, 1+N_pre, C]
    dim = int(pos_embed.shape[-1])

    def safe_interp(x: torch.Tensor, w: torch.Tensor, h: torch.Tensor):
        """
        x : [B, 1+N_now, C]  (CLS + PATCH tokens)
        returns: [B, 1+N_now, C] positional embeddings
        """
        B, L, C = x.shape
        if L <= 1:
            # degenerate (no patches) -> just return CLS pos
            return pos_embed[:, :1, :].expand(B, -1, -1)

        # Strip CLS from pretrained PE and reshape to 2D grid
        pe = pos_embed[:, 1:, :]        # [1, N_pre, C]
        n_pre = int(pe.shape[1])
        side_pre = int(round(math.sqrt(max(n_pre, 1))))
        pe_4d = pe.reshape(1, side_pre, side_pre, dim).permute(0, 3, 1, 2)  # [1, C, Hpre, Wpre]

        # Current patch token count (exclude CLS)
        N_now = L - 1
        gh, gw = _best_factor_pair(N_now)

        # ONNX-safe: specify integer output size and avoid bicubic/AA kernel
        pe_resized = F.interpolate(
            pe_4d, size=(int(gh), int(gw)), mode="bilinear",
            align_corners=False, antialias=False
        )
        pe_tokens = pe_resized.permute(0, 2, 3, 1).reshape(1, gh * gw, dim)  # [1, N_now, C]

        # Concatenate CLS positional token in front and broadcast over batch
        cls_tok = pos_embed[:, :1, :]  # [1,1,C]
        full = torch.cat([cls_tok, pe_tokens], dim=1)  # [1, 1+N_now, C]
        return full.expand(B, -1, -1)

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
        self.patch = 16

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
        tokens, cls = _tokens_from_dino(out)  # PATCH tokens (normalized)

        # Expected coarse grid from the current spatial size
        gh, gw = H // self.patch, W // self.patch
        N = tokens.size(1)
        if N != gh * gw:
            # Fallback if DINO decided a different tokenization internally; trust N
            gh, gw = _best_factor_pair(N)

        feat = tokens.reshape(B, gh, gw, tokens.size(2)).permute(0, 3, 1, 2).contiguous()  # [B,1024,gh,gw]
        feat = self.proj(feat)
        outd = {"coarse": feat}
        if self.use_cls and cls is not None:
            outd["cls"] = cls.to(feat.dtype)
        return outd