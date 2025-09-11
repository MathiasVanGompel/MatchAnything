# check_onnx_outputs.py
import argparse
import onnxruntime as ort
import numpy as np
import cv2


def load_rgb(path, size, norm):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    x = img.astype(np.float32) / 255.0
    if norm == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
    elif norm == "neg_one_one":
        x = x * 2.0 - 1.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--img0", required=True)
    ap.add_argument("--img1", required=True)
    ap.add_argument("--size", type=int, default=832)
    ap.add_argument(
        "--norm", choices=["none", "imagenet", "neg_one_one"], default="none"
    )
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.name for i in sess.get_inputs()}
    outputs = [o.name for o in sess.get_outputs()]
    print("[INFO] Inputs:", inputs)
    print("[INFO] Outputs:", outputs)

    name0 = "image0" if "image0" in inputs else list(inputs.keys())[0]
    name1 = "image1" if "image1" in inputs else list(inputs.keys())[1]

    x0 = load_rgb(args.img0, args.size, args.norm)
    x1 = load_rgb(args.img1, args.size, args.norm)

    out = sess.run(output_names=outputs, input_feed={name0: x0, name1: x1})

    def pick(name, idx):
        if name in outputs:
            return out[outputs.index(name)]
        return out[idx]

    k0 = pick("keypoints0", 0)
    k1 = pick("keypoints1", 1)
    mc = pick("mconf", 2).reshape(-1)

    def stats(a, name):
        finite = np.isfinite(a).sum()
        print(
            f"{name}: shape={a.shape}, finite={finite}/{a.size}, min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}, mean={np.nanmean(a):.6g}"
        )

    stats(k0, "keypoints0")
    stats(k1, "keypoints1")
    stats(mc, "mconf")


if __name__ == "__main__":
    main()
