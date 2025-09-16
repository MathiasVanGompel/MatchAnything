#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, torch

# --- BEGIN: RoMa path bootstrap ---
THIS = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(THIS, "..", ".."))
ROMA_CANDIDATES = [
    os.path.join(REPO, "imcui", "third_party", "MatchAnything", "third_party", "ROMA"),
    os.path.join(REPO, "imcui", "third_party", "MatchAnything", "third_party", "RoMa"),
    os.path.join(REPO, "imcui", "third_party", "ROMA"),
    os.path.join(REPO, "third_party", "ROMA"),
]
for p in ROMA_CANDIDATES:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
# --- END: RoMa path bootstrap ---

from typing import Dict
from full_matchanything_trt import FullMatchAnythingTRT
from unified_weight_loader import apply_unified_weight_loading

def parse_args():
    ap = argparse.ArgumentParser("Full MatchAnything (encoder+head) → ONNX, with ROMA ckpt remap")
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--onnx", default="Convertion_Tensorrt/out/matchanything_full_518.onnx")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--precision", default="fp16", choices=["fp32","fp16"])
    ap.add_argument("--H", type=int, default=518)
    ap.add_argument("--W", type=int, default=518)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--beta", type=float, default=14.285714285714286)  # 1/0.07
    return ap.parse_args()

def _filter_to_model_keys(loadable: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]):
    out = {}
    for k, v in loadable.items():
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
            out[k] = v
    return out

def main():
    a = parse_args()
    amp = (a.precision == "fp16")
    device = torch.device(a.device)

    model = FullMatchAnythingTRT(input_hw=(a.H, a.W), amp=amp, beta=a.beta).to(device).eval()

    print(f"[LOAD] Remapping ROMA checkpoint: {a.ckpt}")
    model_state = model.state_dict()
    loadable = apply_unified_weight_loading(a.ckpt, model_state, load_dinov2_components=True)
    filtered = _filter_to_model_keys(loadable, model_state)
    print(f"[LOAD] Will load {len(filtered)} tensors into model")
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)

    if amp: model = model.half()
    else:   model = model.float()

    dtype = torch.float16 if amp else torch.float32
    d0 = torch.rand(1,3,a.H,a.W, device=device, dtype=dtype)
    d1 = torch.rand(1,3,a.H,a.W, device=device, dtype=dtype)

    os.makedirs(os.path.dirname(a.onnx), exist_ok=True)
    print("[ONNX] exporting to:", a.onnx)
    torch.onnx.export(
        model, (d0,d1), a.onnx,
        opset_version=a.opset,
        input_names=["image0","image1"],
        output_names=["warp","cert"],
        dynamic_axes={
            "image0": {0:"B",2:"H",3:"W"},
            "image1": {0:"B",2:"H",3:"W"},
            "warp":   {0:"B",2:"Hc",3:"Wc"},
            "cert":   {0:"B",2:"Hc",3:"Wc"},
        },
        do_constant_folding=True,
    )
    print("[OK] ONNX saved:", a.onnx)

if __name__ == "__main__":
    main()
