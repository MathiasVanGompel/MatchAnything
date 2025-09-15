#!/usr/bin/env python3
import argparse, os, torch
from full_matchanything_trt import FullMatchAnythingTRT

def parse_args():
    ap = argparse.ArgumentParser("Full MatchAnything (encoder+head) → ONNX")
    ap.add_argument("--onnx", default="Convertion_Tensorrt/out/matchanything_full.onnx")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--precision", default="fp16", choices=["fp32","fp16"])
    ap.add_argument("--H", type=int, default=784)
    ap.add_argument("--W", type=int, default=784)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--beta", type=float, default=14.285714285714286)  # 1/0.07
    return ap.parse_args()

def main():
    a = parse_args()
    amp = (a.precision == "fp16")
    device = torch.device(a.device)

    model = FullMatchAnythingTRT(input_hw=(a.H, a.W), amp=amp, beta=a.beta).to(device).eval()
    d0 = torch.rand(1,3,a.H,a.W, device=device, dtype=torch.float16 if amp else torch.float32)
    d1 = torch.rand(1,3,a.H,a.W, device=device, dtype=torch.float16 if amp else torch.float32)

    os.makedirs(os.path.dirname(a.onnx), exist_ok=True)
    print("[ONNX] exporting to:", a.onnx)

    export_kwargs = dict(
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

    # Try external-data arg; fall back if your torch doesn’t support it.
    try:
        torch.onnx.export(model, (d0,d1), a.onnx, use_external_data_format=True, **export_kwargs)
    except TypeError:
        # Newer PyTorch uses external data automatically for >2GB models, so this is fine. :contentReference[oaicite:1]{index=1}
        torch.onnx.export(model, (d0,d1), a.onnx, **export_kwargs)

    print("[OK] ONNX saved:", a.onnx)

if __name__ == "__main__":
    main()
