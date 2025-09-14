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

        self.dino = vit_large(patch_size=self.patch)  # expects ImageNet-normalized RGB

        with torch.no_grad():
            N0 = int(self.dino.pos_embed.shape[1])
            gh0 = int((N0) ** 0.5)
            while N0 % gh0 != 0:
                gh0 -= 1
            gw0 = N0 // gh0
            self.register_buffer("_pe_gh0", torch.tensor(gh0, dtype=torch.int64), persistent=False)
            self.register_buffer("_pe_gw0", torch.tensor(gw0, dtype=torch.int64), persistent=False)

        self.proj = nn.Identity()
        if out_channels != 1024:
            self.proj = nn.Conv2d(1024, out_channels, kernel_size=1, bias=False)

    def _resize_pos_embed_dynamic(self, pe: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        """
        pe:  [1, N0, C]  (learned PE from pretrain)
        x_in: [B, 3, H, W]  (current input)
        Returns PE resized to current patch grid: [1, gh*gw, C]
        """
        H = x_in.shape[-2]
        W = x_in.shape[-1]
        gh = H // self.patch
        gw = W // self.patch

        C = pe.shape[-1]
        gh0 = int(self._pe_gh0.item())
        gw0 = int(self._pe_gw0.item())
        pe_4d = pe.transpose(1, 2).reshape(1, C, gh0, gw0)
        pe_4d = F.interpolate(pe_4d, size=(gh, gw), mode="bilinear", align_corners=False)
        return pe_4d.flatten(2).transpose(1, 2)

    def forward(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x_in.shape

        if C == 1:
            x_in = x_in.repeat(1, 3, 1, 1)

        mean = x_in.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x_in.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x_in - mean) / std
        if self.amp:
            x = x.to(self.amp_dtype)

        if hasattr(self.dino, "pos_embed"):
            self.dino.pos_embed = self._resize_pos_embed_dynamic(self.dino.pos_embed, x_in)

        out = self.dino.forward_features(x)
        tokens, cls = _tokens_from_dino(out)

        gh, gw = H // self.patch, W // self.patch
        N = tokens.size(1)
        if N != gh * gw:
            gh, gw = _best_factor_pair(N)

        feat = tokens.reshape(B, gh, gw, tokens.size(2)).permute(0, 3, 1, 2).contiguous()
        feat = self.proj(feat)
        outd = {"coarse": feat}
        if self.use_cls and cls is not None:
            outd["cls"] = cls.to(feat.dtype)
        return outd
