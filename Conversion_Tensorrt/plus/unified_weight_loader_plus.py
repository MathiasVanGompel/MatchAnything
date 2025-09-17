# Extension of the unified weight loader that supports decoder and refiner weights.

from typing import Dict
import re
import torch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from full.unified_weight_loader import build_mapped_state_dict as base_build
except Exception:
    base_build = None

_ROMA_TO_THIS = [
    (re.compile(r"^matcher\.model\.decoder\."), "decoder."),         # Map decoder weights to MatchDecoderTRT.
    (re.compile(r"^matcher\.model\.encoder\.cnn\."), "refiner."),    # Map fine CNN weights to RefineCNNTRT.
    (re.compile(r"^roma\.decoder\."), "decoder."),
    (re.compile(r"^roma\.refine\."), "refiner."),
]

def _apply_extra_maps(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        mapped = k
        for rgx, rep in _ROMA_TO_THIS:
            if rgx.search(mapped):
                mapped = rgx.sub(rep, mapped)
                break
        out[mapped] = v
    return out

def build_mapped_state_dict(ckpt: Dict[str, torch.Tensor], model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped = _apply_extra_maps(ckpt)
    if base_build is not None:
        return base_build(mapped, model_state_dict)
    loadable = {}
    for k, t in model_state_dict.items():
        if k in mapped and tuple(mapped[k].shape) == tuple(t.shape):
            loadable[k] = mapped[k]
    return loadable
