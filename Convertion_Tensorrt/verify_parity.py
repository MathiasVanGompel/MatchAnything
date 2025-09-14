import argparse
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
from preprocess import preprocess_rgb, unpad_like

@torch.no_grad()
def run_pytorch(model, img0, img1):
    img0, pads0 = preprocess_rgb(img0)
    img1, pads1 = preprocess_rgb(img1)
    w0, c0 = model(img0, img1)
    return unpad_like(w0, pads0), unpad_like(c0, pads0)

def run_onnx(sess, img0, img1):
    img0, pads0 = preprocess_rgb(img0)
    img1, pads1 = preprocess_rgb(img1)
    outs = sess.run(None, {"image0": img0.cpu().numpy(), "image1": img1.cpu().numpy()})
    warp, cert = [torch.from_numpy(x) for x in outs]
    return unpad_like(warp, pads0), unpad_like(cert, pads0)

def mae(a, b):
    return (a - b).abs().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, default="")
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--images", nargs=2, required=True)
    args = ap.parse_args()

    def load_img(p):
        import cv2
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(im).permute(2,0,1).unsqueeze(0)
        return t

    img0 = load_img(args.images[0])
    img1 = load_img(args.images[1])

    from accurate_matchanything_trt import AccurateMatchAnythingTRT
    model = AccurateMatchAnythingTRT().eval().cuda() if torch.cuda.is_available() else AccurateMatchAnythingTRT().eval()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    w_pt, c_pt = run_pytorch(model, img0.clone(), img1.clone())
    w_ox, c_ox = run_onnx(sess, img0.clone(), img1.clone())

    print("MAE warp:", mae(w_pt, w_ox))
    print("MAE cert:", mae(c_pt, c_ox))

    H, W = w_pt.shape[-2:]
    ys = torch.randint(0, H, (512,))
    xs = torch.randint(0, W, (512,))
    flow_pt = w_pt[0, :, ys, xs].T
    flow_ox = w_ox[0, :, ys, xs].T
    err = (flow_pt - flow_ox).norm(dim=1)
    print("Pct within 1px:", (err < 1.0).float().mean().item() * 100.0)

if __name__ == "__main__":
    main()

