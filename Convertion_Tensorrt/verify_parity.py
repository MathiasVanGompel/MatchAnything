# Convertion_Tensorrt/verify_parity.py
import argparse, os, sys
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# allow running as "python Convertion_Tensorrt/verify_parity.py"
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from accurate_matchanything_trt import AccurateMatchAnythingTRT

@torch.no_grad()
def run_pytorch(model, img0, img1):
    return model(img0, img1)

def load_image(path, device):
    im = Image.open(path).convert("RGB")
    t = TF.to_tensor(im).unsqueeze(0)  # [1,3,H,W], float32
    return t.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs=2, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = AccurateMatchAnythingTRT(amp=False).to(device).eval()

    img0 = load_image(args.images[0], device)
    img1 = load_image(args.images[1], device)

    w_pt, c_pt = run_pytorch(model, img0, img1)
    print("warp_c:", tuple(w_pt.shape), "cert_c:", tuple(c_pt.shape))

if __name__ == "__main__":
    main()
