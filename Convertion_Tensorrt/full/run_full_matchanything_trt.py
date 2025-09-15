#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, cv2, pycuda.autoinit  # noqa
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import prepare_pair_same_hw
from trt_engine import TRTEngine

def dense_to_keypoints(warp_c: np.ndarray, cert_c: np.ndarray, scale: int = 14):
    warp = warp_c[0]          # [2,Hc,Wc]
    cert = cert_c[0,0]        # [Hc,Wc]
    Hc, Wc = cert.shape
    ys, xs = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing="ij")
    mkpts0 = np.stack([xs, ys], -1).reshape(-1,2) * scale
    mkpts1 = warp.transpose(1,2,0).reshape(-1,2) * scale
    mconf  = cert.reshape(-1)
    return mkpts0.astype(np.float32), mkpts1.astype(np.float32), mconf.astype(np.float32)

def extract_matches(k0, k1, conf, thr):
    m = conf >= float(thr)
    if not np.any(m): return np.empty((0,4), np.float32)
    a, b = k0[m], k1[m]
    return np.column_stack([a[:,0],a[:,1], b[:,0],b[:,1]]).astype(np.float32)

def draw_matches(img0_path, img1_path, matches, out_path, src_size):
    import cv2, numpy as np

    optH, optW = src_size  # the network input (e.g., 518, 518)

    img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    h0, w0 = img0.shape[:2]; h1, w1 = img1.shape[:2]

    # scale (x,y) from opt-space -> original image space
    sx0, sy0 = w0 / float(optW), h0 / float(optH)
    sx1, sy1 = w1 / float(optW), h1 / float(optH)

    m = matches.copy()
    # scale left endpoints
    m[:, 0] *= sx0; m[:, 1] *= sy0
    # scale right endpoints
    m[:, 2] *= sx1; m[:, 3] *= sy1

    # concat canvas
    canvas = np.zeros((max(h0,h1), w0+w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1

    # random colors for lines
    rng = np.random.default_rng(0)
    for (x0,y0,x1,y1) in m:
        color = tuple(int(c) for c in rng.integers(0,255, size=3))
        p0 = (int(x0), int(y0))
        p1 = (int(x1)+w0, int(y1))
        cv2.circle(canvas, p0, 2, color, -1)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print("[OK] wrote", out_path)


def main():
    ap = argparse.ArgumentParser("MatchAnything (full TensorRT) inference")
    ap.add_argument("--engine", required=True, help="full .plan (image0,image1 -> warp,cert)")
    ap.add_argument("--image0", required=True)
    ap.add_argument("--image1", required=True)
    ap.add_argument("--opt", type=int, default=784, help="square OPT used at build (multiple of 14)")
    ap.add_argument("--conf", type=float, default=0.30, help="certainty threshold (try 0.3~0.5)")
    ap.add_argument("--outdir", default="Convertion_Tensorrt/out/full_results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # 1) preprocess to exact (opt,opt)
    x0, x1 = prepare_pair_same_hw(args.image0, args.image1, (args.opt,args.opt))
    # 2) TRT
    trt_net = TRTEngine(args.engine)
    outs = trt_net.infer({"image0": x0, "image1": x1})
    warp_c, cert_c = outs["warp"], outs["cert"]
    print("warp:", warp_c.shape, "cert:", cert_c.shape)

    # 3) matches + viz
    k0,k1,conf = dense_to_keypoints(warp_c, cert_c)
    matches = extract_matches(k0,k1,conf,args.conf)
    np.savez(os.path.join(args.outdir,"matches.npz"), matches=matches, keypoints0=k0, keypoints1=k1, mconf=conf)
    print("[OK] saved matches:", os.path.join(args.outdir,"matches.npz"), "N=", len(matches))
    draw_matches(
    args.image0, args.image1, matches,
    os.path.join(args.outdir,"matches_visualization.jpg"),
    src_size=(args.opt, args.opt)
)

if __name__ == "__main__":
    main()
