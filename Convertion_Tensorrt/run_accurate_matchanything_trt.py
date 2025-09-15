#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accurate MatchAnything — HYBRID pipeline:
- TensorRT 10 encoder (image0,image1 -> f0,f1)
- PyTorch RoMa coarse head (f0,f1 -> warp_c, cert_c)
- Matches + visualization

Requires: tensorrt>=10, pycuda, torch, numpy, pillow, opencv-python
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import cv2

import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

from preprocess import prepare_pair_same_hw
from roma_head import coarse_from_features

# ---- simple viz --------------------------------------------------------------

def dense_to_keypoints(warp_c: np.ndarray, cert_c: np.ndarray, scale: int = 14):
    warp = warp_c[0]  # [2,Hc,Wc]
    cert = cert_c[0, 0]  # [Hc,Wc]
    Hc, Wc = cert.shape
    ys, xs = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing="ij")
    mkpts0 = np.stack([xs, ys], axis=-1).reshape(-1, 2) * scale
    mkpts1 = warp.transpose(1, 2, 0).reshape(-1, 2) * scale
    mconf = cert.reshape(-1)
    return mkpts0.astype(np.float32), mkpts1.astype(np.float32), mconf.astype(np.float32)

def extract_matches_from_results(k0: np.ndarray, k1: np.ndarray, conf: np.ndarray, thr: float = 0.1):
    mask = conf >= float(thr)
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32)
    a = k0[mask]
    b = k1[mask]
    return np.column_stack([a[:, 0], a[:, 1], b[:, 0], b[:, 1]]).astype(np.float32)

def draw_matches_visualization(img0_path: str, img1_path: str, matches: np.ndarray, output_path: str, max_matches: int = 1000):
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    if img0 is None or img1 is None:
        print("Warning: could not load images for visualization.")
        return
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    m = matches if len(matches) <= max_matches else matches[np.random.choice(len(matches), max_matches, replace=False)]
    rng = np.random.default_rng(123)
    for x0, y0, x1, y1 in m:
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)) + w0, int(round(y1)))
        cv2.circle(canvas, p0, 2, color, -1)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    print(f"[OK] visualization: {output_path}")

# ---- TRT 10 encoder wrapper --------------------------------------------------

class TRTEngineEncoder:
    """
    Minimal TensorRT v10 runtime wrapper for an engine with:
      inputs:  image0, image1   (NCHW float)
      outputs: f0, f1           (NCHW float)

    Uses new TensorRT 10 Python API (named tensors + set_input_shape / execute_async_v3).
    See NVIDIA docs on dynamic shapes and runtime API. :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Introspect I/O tensors
        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

        print(f"Engine loaded: {engine_path}")
        print("Inputs: ", self.inputs)
        print("Outputs:", self.outputs)

    def infer_encoder(self, img0: np.ndarray, img1: np.ndarray) -> Dict[str, np.ndarray]:
        """
        img0, img1: numpy [1,3,H,W] float32, CONTIGUOUS, same HxW.
        Returns numpy dict: {'f0': [1,1024,Hc,Wc], 'f1': ...}
        """
        assert img0.flags['C_CONTIGUOUS'] and img1.flags['C_CONTIGUOUS'], "Inputs must be contiguous."
        assert img0.shape == img1.shape, "Both inputs must have identical shapes (named dims H/W are equal in engine)."

        # Set shapes
        self.context.set_input_shape("image0", tuple(img0.shape))
        self.context.set_input_shape("image1", tuple(img1.shape))

        # Create stream
        stream = cuda.Stream()

        # Allocate device inputs
        d0 = cuda.mem_alloc(img0.nbytes)
        d1 = cuda.mem_alloc(img1.nbytes)

        # Set input tensor addresses
        self.context.set_tensor_address("image0", int(d0))
        self.context.set_tensor_address("image1", int(d1))

        # Copy H2D
        cuda.memcpy_htod_async(d0, img0, stream)
        cuda.memcpy_htod_async(d1, img1, stream)

        # Allocate outputs AFTER shapes are set
        d_out = {}
        h_out = {}
        for name in self.outputs:
            shp = tuple(self.context.get_tensor_shape(name))  # fully-resolved now
            dtype = np.float32  # encoder exported as FP32/FP16 compute, but outputs stored as float (engine usually returns fp32 here)
            h_out[name] = np.empty(shp, dtype=dtype, order="C")
            d_out[name] = cuda.mem_alloc(h_out[name].nbytes)
            self.context.set_tensor_address(name, int(d_out[name]))

        # Run
        ok = self.context.execute_async_v3(stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        # D2H
        for name in self.outputs:
            cuda.memcpy_dtoh_async(h_out[name], d_out[name], stream)
        stream.synchronize()

        return h_out

# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Accurate MatchAnything — HYBRID")
    ap.add_argument("--engine", required=True, help="TensorRT .plan for encoder (2 inputs -> 2 outputs)")
    ap.add_argument("--image0", required=True)
    ap.add_argument("--image1", required=True)
    ap.add_argument("--opt", type=int, default=784, help="Square OPT size used at build (multiple of 14).")
    ap.add_argument("--conf", type=float, default=0.1, help="confidence threshold")
    ap.add_argument("--outdir", default="Convertion_Tensorrt/out/accurate_results")
    args = ap.parse_args()

    print("="*60)
    print("ACCURATE MATCHANYTHING — HYBRID INFERENCE")
    print("="*60)
    print(f"Engine: {args.engine}")

    trt_engine = TRTEngineEncoder(args.engine)

    # 1) Preprocess to EXACT same HxW = (opt,opt)  — fixes named-dimension equality
    x0, x1 = prepare_pair_same_hw(args.image0, args.image1, (args.opt, args.opt))
    print(f"Prepared shapes: image0={x0.shape}, image1={x1.shape}")

    # 2) TRT encoder
    outs = trt_engine.infer_encoder(x0, x1)
    f0 = outs["f0"]
    f1 = outs["f1"]
    print(f"Encoder outputs: f0={f0.shape}, f1={f1.shape}")

    # 3) RoMa coarse head (PyTorch)
    warp_c, cert_c = coarse_from_features(f0, f1, temperature=0.07)
    print(f"Head outputs: warp_c={warp_c.shape}, cert_c={cert_c.shape}")

    # 4) keypoints/matches + viz
    k0, k1, conf = dense_to_keypoints(warp_c, cert_c)
    matches = extract_matches_from_results(k0, k1, conf, args.conf)

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "matches.npz"),
             matches=matches, keypoints0=k0, keypoints1=k1, mconf=conf)
    print(f"[OK] saved matches: {os.path.join(args.outdir, 'matches.npz')}  (N={len(matches)})")

    viz_path = os.path.join(args.outdir, "matches_visualization.jpg")
    draw_matches_visualization(args.image0, args.image1, matches, viz_path, max_matches=1000)

    print("\nDone.")

if __name__ == "__main__":
    main()
