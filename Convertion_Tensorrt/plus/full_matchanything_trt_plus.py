# full_matchanything_trt_plus.py
# Wrapper that reuses your DINOv2 encoder + GP head,
# and adds a RoMa-style decoder and optional refinement.
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# import your existing encoder and GP head
try:
    from encoders_trt_full import DINOv2EncoderTRT
except Exception as e:
    raise ImportError(
        "DINOv2EncoderTRT not found. Keep this file next to your existing "
        "code so the import works. Original error: %r" % (e,)
    )

try:
    from gp_trt import GPMatchEncoderTRT  # your current GP head
except Exception as e:
    raise ImportError(
        "GPMatchEncoderTRT not found. Keep this file next to your existing code. "
        "Original error: %r" % (e,)
    )

from decoder_refine_trt import MatchDecoderTRT, RefineCNNTRT

def _as_tensor_feat(x):
    """
    Robustly unwrap encoder outputs:
      - If tensor: return it
      - If dict: prefer common keys; otherwise first tensor value
      - If (list/tuple): first tensor element or recurse for dicts
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        # common names seen in matching backbones
        for k in ("feat", "features", "feats", "x", "out", "backbone", "c"):
            v = x.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        for v in x.values():
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                return v[0]
    if isinstance(x, (list, tuple)) and len(x):
        # pick first tensor-like thing
        for v in x:
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, dict):
                t = _as_tensor_feat(v)
                if isinstance(t, torch.Tensor):
                    return t
    raise TypeError(f"Unsupported encoder output type: {type(x)}")

class FullMatchAnythingTRTPlus(nn.Module):
    def __init__(
        self,
        use_head: str = "gp",                     # 'gp' | 'decoder' | 'decoder_refine'
        gp_beta: float = 1.0/0.07,               # keep parity with your original GP head
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder_dim: int = 1024,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        decoder_head_dim: int = 64,
        refine_iters: int = 1,
        upsample_res: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.use_head = use_head
        self.upsample_res = upsample_res

        # ---- robust encoder construction (no assumption about arg names) ----
        self.encoder = self._build_encoder(encoder_kwargs)

        # ---- heads ----
        self.gp = GPMatchEncoderTRT(beta=gp_beta)
        self.decoder: Optional[MatchDecoderTRT] = None
        self.refiner: Optional[RefineCNNTRT] = None

        if use_head in ("decoder", "decoder_refine"):
            self.decoder = MatchDecoderTRT(
                dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, head_dim=decoder_head_dim
            )
        if use_head == "decoder_refine":
            self.refiner = RefineCNNTRT(in_ch=3, hidden=64, iters=refine_iters)

    def _build_encoder(self, encoder_kwargs: Optional[Dict[str, Any]]) -> nn.Module:
        """Try several constructor shapes to match your local class signature."""
        kw = dict(encoder_kwargs or {})
        # 1) try exactly what was provided
        try:
            return DINOv2EncoderTRT(**kw)
        except TypeError:
            pass
        # 2) common pattern in your repo: (input_hw, amp)
        try:
            ihw = kw.get("input_hw", (560, 560))
            amp = kw.get("amp", False)
            return DINOv2EncoderTRT(input_hw=ihw, amp=amp)
        except TypeError:
            pass
        # 3) bare constructor (no kwargs)
        try:
            return DINOv2EncoderTRT()
        except TypeError as e:
            raise TypeError(
                "Could not construct DINOv2EncoderTRT with any of the expected signatures. "
                "Tried (**kw), (input_hw, amp) and (). Original error: %r" % (e,)
            )

    @torch.no_grad()
    def forward(self, imgA: torch.Tensor, imgB: torch.Tensor):
        # encode
        encA, encB = self.encoder(imgA), self.encoder(imgB)  # may be dicts
        fA = _as_tensor_feat(encA)
        fB = _as_tensor_feat(encB)

        B, C, Ha, Wa = fA.shape
        Hb, Wb = fB.shape[-2:]

        if self.use_head == "gp" or self.decoder is None:
            warp, cert = self.gp(fA, fB)
        else:
            warp, cert = self.decoder(fA, fB, Ha, Wa, Hb, Wb)
            if self.use_head == "decoder_refine" and self.refiner is not None:
                warp, cert = self.refiner(warp, cert)

        if self.upsample_res is not None:
            warp = F.interpolate(warp, size=self.upsample_res, mode="bilinear", align_corners=True)
            cert = F.interpolate(cert, size=self.upsample_res, mode="bilinear", align_corners=True)
        return warp, cert
