import importlib
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import sys, os

# make top-level package importable no matter where we run from
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
        # your submodule path
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
def build_dinov2_vitl14_romacfg(img_size: int = 518) -> nn.Module:
    vit_large = _import_vit_large()
    model = vit_large(
        img_size=img_size,      # MUST be 518 to match checkpoint (pos_embed length 1370)
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
    )
    model.eval()
    state = torch.hub.load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        map_location="cpu",
        check_hash=False,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


# --- TRT-friendly encoder wrapper ---------------------------------------------
class DINOv2EncoderTRT(nn.Module):
    """
    Wrap DINOv2 so we always keep dtype-consistency and ONNX-traceable control-flow.
    - amp=False (default): FP32 throughout → safest ONNX export.
    - amp=True : casts model+inputs to half.
    Returns {'coarse': [B, 1024, H/14, W/14]} for L/14.
    """
    def __init__(
        self,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        input_hw: Tuple[int, int] = (518, 518),
    ):
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.input_hw = input_hw
        self.dino = build_dinov2_vitl14_romacfg()
        if self.amp:
            self.dino.half()  # align params/biases with FP16 inputs

        # cache for ONNX-friendly constants
        self.patch = 14

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, 1|3, H, W] in [0,1]. Ensures 3 channels and correct dtype.
        Returns: {'coarse': [B, C, H/14, W/14]} with C=1024 for ViT-L/14.
        """
        B, C, H, W = x.shape

        # grayscale → 3ch (avoid Python bool branches that annoy ONNX)
        rep = (C == 1)
        if rep:
            x = x.repeat(1, 3, 1, 1)

        # H,W must be multiples of patch size (14)
        assert H % self.patch == 0 and W % self.patch == 0, "H and W must be multiples of 14."
        Hc, Wc = H // self.patch, W // self.patch

        if self.amp:
            x = x.to(dtype=self.amp_dtype)
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                out = self.dino.forward_features(x)
        else:
            x = x.to(dtype=torch.float32)
            out = self.dino.forward_features(x)

        # DINOv2 forward_features returns a dict for RoMa; prefer normalized patch tokens.
        preferred_keys = [
            "x_norm_patchtokens",  # RoMa / DINOv2 normalized patch tokens
            "patch_tokens",
            "tokens",
            "last_hidden_state",   # HF/transformers style
            "x",                   # some forks
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

        # ---- Normalize to [B, C, Hc, Wc] without Python shape-conditions ----------
        if seq.dim() == 3:
            # seq: [B, N, Ctok]  (may include CLS → N == Hc*Wc + 1)
            # take the last Hc*Wc tokens; drops CLS if present, no branches.
            B_, N_, Ctok = seq.shape
            tokens = seq[:, -(Hc * Wc):, :]
            fmap = tokens.transpose(1, 2).contiguous().view(B_, Ctok, Hc, Wc)

        elif seq.dim() == 4:
            # already [B, C, Hc, Wc]
            fmap = seq.contiguous()
        else:
            raise RuntimeError(f"Unsupported token tensor shape: {tuple(seq.shape)}")

        return {"coarse": fmap}
