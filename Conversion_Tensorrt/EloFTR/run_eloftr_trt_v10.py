#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TensorRT 10.x runtime driver for the grayscale E-LoFTR head."""

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np
import tensorrt as trt

# Use PyCUDA because it ships with the MatchAnything tooling environment.
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ---------------------------------------------------------------------------
# TensorRT-specific helpers
# ---------------------------------------------------------------------------
class OutputAllocator(trt.IOutputAllocator):
    """Simple TensorRT output allocator."""

    # One allocator instance can handle multiple outputs. TensorRT calls:
    #   - notify_shape(name, shape): tells you the final resolved shape.
    #   - reallocate_output(name, memory, size, alignment): asks you to return
    #     a device pointer with at least ``size`` bytes.
    def __init__(self):
        super().__init__()
        self.device_buffers = {}  # name -> pycuda.driver.DeviceAllocation
        self.shapes = {}          # name -> tuple(int)

    def notify_shape(self, tensor_name, shape):
        # shape is a sequence of ints (no -1 at this point)
        self.shapes[tensor_name] = tuple(int(x) for x in shape)

    def reallocate_output(self, tensor_name, memory, size, alignment):
        import pycuda.driver as cuda
        # Free old buffer if TRT asks for a new size
        if tensor_name in self.device_buffers:
            # If the size didn't change, you could return the same pointer.
            # To keep it simple, always reallocate:
            self.device_buffers[tensor_name].free()
        self.device_buffers[tensor_name] = cuda.mem_alloc(int(size))
        # Return the device pointer *as an integer*
        return int(self.device_buffers[tensor_name])

    # Optional: async variant (TRT may call this instead of reallocate_output)
    def reallocate_output_async(self, tensor_name, memory, size, alignment, stream_handle):
        return self.reallocate_output(tensor_name, memory, size, alignment)

def np_from_trt_dtype(dt: trt.DataType) -> np.dtype:
    """Map a TensorRT dtype to the corresponding ``numpy`` dtype."""
    return np.dtype(trt.nptype(dt))

def vol(shape):
    v = 1
    for s in shape:
        v *= int(s)
    return v

def preprocess_gray(path: Optional[str], H: int, W: int) -> np.ndarray:
    if path is None:
        img = np.zeros((H, W), dtype=np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img[None, None, :, :]  # NCHW
    return np.ascontiguousarray(img)

def draw_matches_gray(imgL, imgR, mkpts0, mkpts1, mconf, out_path):
    """Very simple side-by-side line visualization."""
    H, W = imgL.shape
    color = (0, 255, 0)
    imgL3 = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    imgR3 = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
    canvas = np.concatenate([imgL3, imgR3], axis=1)
    # scale points because model ran on possibly resized images
    # here we assume we passed (H,W) to preprocessing, so points are in that space.
    N = min(200, mkpts0.shape[0])  # don't draw too many
    if mkpts0.ndim == 2 and mkpts0.shape[1] == 2 and mkpts1.ndim == 2:
        for i in range(N):
            p0 = (int(round(mkpts0[i, 0])), int(round(mkpts0[i, 1])))
            p1 = (int(round(mkpts1[i, 0] + W)), int(round(mkpts1[i, 1])))
            cv2.circle(canvas, p0, 2, color, -1)
            cv2.circle(canvas, p1, 2, color, -1)
            cv2.line(canvas, p0, p1, color, 1)
    cv2.imwrite(out_path, canvas)

def load_engine(engine_path: str):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")
    return engine

def print_io(engine):
    """Print the TensorRT engine bindings and tensors."""

    # TensorRT v10+ tensor-centric API
    if hasattr(engine, "num_io_tensors"):
        n = engine.num_io_tensors
        print("=== Engine tensors (v10+) ===")
        for i in range(n):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)
            print(f"[{i}] {name:15s}  {str(mode):6s}  {dtype}  shape={tuple(shape)}")
    else:
        # Legacy fallback (TRT <= 8.x)
        nb = engine.num_bindings
        print("=== Engine bindings (legacy) ===")
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            kind = "INPUT" if is_input else "OUTPUT"
            print(f"[{i}] {name:15s}  {kind:6s}  {dtype}  shape={tuple(shape)}")

def pick_io_names(engine):
    """Infer canonical I/O tensor names to match the exporter convention."""

    # Prefer standard names: image0, image1, mkpts0, mkpts1, mconf. Otherwise
    # choose by roles (INPUT/OUTPUT) and simple shape heuristics.
    names = {"image0": None, "image1": None, "mkpts0": None, "mkpts1": None, "mconf": None}
    if hasattr(engine, "num_io_tensors"):
        # v10+
        inputs, outputs = [], []
        for i in range(engine.num_io_tensors):
            nm = engine.get_tensor_name(i)
            if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                inputs.append(nm)
            else:
                outputs.append(nm)
        # Try known names first
        for k in names:
            if k in inputs + outputs:
                names[k] = k
        # Fill missing inputs
        in_missing = [k for k in ["image0", "image1"] if names[k] is None]
        if in_missing and len(inputs) >= len(in_missing):
            for k, nm in zip(in_missing, sorted(inputs)):
                names[k] = nm
        # Fill missing outputs
        out_missing = [k for k in ["mkpts0", "mkpts1", "mconf"] if names[k] is None]
        if out_missing:
            # use simple heuristic: two outputs with last dim=2 -> mkpts*, one 1D -> mconf
            # (If shapes are dynamic (-1), we'll resolve after setting input shape.)
            # For now, just sort names to keep deterministic order.
            for k, nm in zip(out_missing, sorted(outputs)):
                names[k] = nm
    else:
        # Legacy API (won't be used for you, but keep for completeness)
        inputs, outputs = [], []
        for i in range(engine.num_bindings):
            nm = engine.get_binding_name(i)
            (inputs if engine.binding_is_input(i) else outputs).append(nm)
        for k in names:
            if k in inputs + outputs:
                names[k] = k
        for k, nm in zip([k for k in ["image0","image1"] if names[k] is None], sorted(inputs)):
            names[k] = nm
        for k, nm in zip([k for k in ["mkpts0","mkpts1","mconf"] if names[k] is None], sorted(outputs)):
            names[k] = nm
    return names

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Entry point for running inference with the TensorRT engine."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--left", default=None, help="Left image (grayscale ok)")
    ap.add_argument("--right", default=None, help="Right image")
    ap.add_argument("--H", type=int, default=320)
    ap.add_argument("--W", type=int, default=480)
    ap.add_argument("--viz", default=None, help="Optional output viz path")
    args = ap.parse_args()

    if not os.path.exists(args.engine):
        print(f"Engine not found: {args.engine}", file=sys.stderr)
        sys.exit(1)

    engine = load_engine(args.engine)
    print_io(engine)

    # Use new API if available (your case)
    is_v10 = hasattr(engine, "num_io_tensors")
    context = engine.create_execution_context()

    # Preprocess (host)
    H, W = args.H, args.W
    h_img0 = preprocess_gray(args.left,  H, W)
    h_img1 = preprocess_gray(args.right, H, W)

    # Guess tensor names
    names = pick_io_names(engine)
    image0 = names["image0"]; image1 = names["image1"]
    mkpts0 = names["mkpts0"]; mkpts1 = names["mkpts1"]; mconf = names["mconf"]

    if is_v10:
        # 1) Set input shapes
        context.set_input_shape(image0, h_img0.shape)
        context.set_input_shape(image1, h_img1.shape)

        # 2) Upload inputs
        d_img0 = cuda.mem_alloc(h_img0.nbytes); cuda.memcpy_htod(d_img0, h_img0)
        d_img1 = cuda.mem_alloc(h_img1.nbytes); cuda.memcpy_htod(d_img1, h_img1)

        # 3) Install allocator for dynamic-size outputs
        allocator = OutputAllocator()
        for out_name in [mkpts0, mkpts1, mconf]:
            context.set_output_allocator(out_name, allocator)  # TRT owns output device ptrs via allocator

        # 4) Set input addresses and run
        context.set_tensor_address(image0, int(d_img0))
        context.set_tensor_address(image1, int(d_img1))
        stream = cuda.Stream()
        ok = context.execute_async_v3(int(stream.handle))
        if not ok:
            raise RuntimeError("execute_async_v3() returned false")
        stream.synchronize()

        # 5) Pull outputs back to host from allocator
        def pull(name: str) -> np.ndarray:
            if name not in allocator.shapes or name not in allocator.device_buffers:
                raise RuntimeError(f"Allocator did not provide buffer/shape for {name}")
            shp = allocator.shapes[name]
            dt = np_from_trt_dtype(engine.get_tensor_dtype(name))
            host = np.empty(shp, dtype=dt)
            cuda.memcpy_dtoh(host, allocator.device_buffers[name])
            return host

        h_mkpts0 = pull(mkpts0)
        h_mkpts1 = pull(mkpts1)
        h_mconf  = pull(mconf)


    else:
        # ---- Legacy fallback (TRT <= 8.x) ----
        # Keep for completeness; most likely you won't hit this path.
        b_image0 = engine.get_binding_index(image0)
        b_image1 = engine.get_binding_index(image1)
        b_mkpts0 = engine.get_binding_index(mkpts0)
        b_mkpts1 = engine.get_binding_index(mkpts1)
        b_mconf  = engine.get_binding_index(mconf)

        context.set_binding_shape(b_image0, h_img0.shape)
        context.set_binding_shape(b_image1, h_img1.shape)

        d_img0 = cuda.mem_alloc(h_img0.nbytes); cuda.memcpy_htod(d_img0, h_img0)
        d_img1 = cuda.mem_alloc(h_img1.nbytes); cuda.memcpy_htod(d_img1, h_img1)

        shp_mkpts0 = tuple(context.get_binding_shape(b_mkpts0))
        shp_mkpts1 = tuple(context.get_binding_shape(b_mkpts1))
        shp_mconf  = tuple(context.get_binding_shape(b_mconf))

        dtype_mkpts0 = np.float32; dtype_mkpts1 = np.float32; dtype_mconf = np.float32
        nbytes_mkpts0 = vol(shp_mkpts0) * 4
        nbytes_mkpts1 = vol(shp_mkpts1) * 4
        nbytes_mconf  = vol(shp_mconf)  * 4

        d_mkpts0 = cuda.mem_alloc(nbytes_mkpts0)
        d_mkpts1 = cuda.mem_alloc(nbytes_mkpts1)
        d_mconf  = cuda.mem_alloc(nbytes_mconf)

        bindings = [None]*engine.num_bindings
        bindings[b_image0] = int(d_img0)
        bindings[b_image1] = int(d_img1)
        bindings[b_mkpts0] = int(d_mkpts0)
        bindings[b_mkpts1] = int(d_mkpts1)
        bindings[b_mconf]  = int(d_mconf)

        ok = context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("execute_v2() returned false")

        h_mkpts0 = np.empty(shp_mkpts0, dtype=dtype_mkpts0); cuda.memcpy_dtoh(h_mkpts0, d_mkpts0)
        h_mkpts1 = np.empty(shp_mkpts1, dtype=dtype_mkpts1); cuda.memcpy_dtoh(h_mkpts1, d_mkpts1)
        h_mconf  = np.empty(shp_mconf,  dtype=dtype_mconf);  cuda.memcpy_dtoh(h_mconf,  d_mconf)

    # Print shapes and a few rows
    print("\nOutput shapes:")
    print(" mkpts0:", h_mkpts0.shape)
    print(" mkpts1:", h_mkpts1.shape)
    print(" mconf :", h_mconf.shape)

    rows = min(5, h_mkpts0.shape[0] if h_mkpts0.ndim > 0 else 0)
    if rows > 0:
        print("\nFirst matches:")
        for i in range(rows):
            x0, y0 = h_mkpts0[i, 0], h_mkpts0[i, 1]
            x1, y1 = h_mkpts1[i, 0], h_mkpts1[i, 1]
            c = h_mconf[i] if h_mconf.ndim == 1 else h_mconf[i, 0]
            print(f"{i:02d} p0=({x0:.1f},{y0:.1f})  p1=({x1:.1f},{y1:.1f})  conf={c:.3f}")
    else:
        print("No matches returned.")

    # Optional visualization (only if you provided real images)
    if args.viz and args.left and args.right:
        # Load original resized grayscale for drawing
        imL = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
        imR = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
        imL = cv2.resize(imL, (W, H), interpolation=cv2.INTER_AREA)
        imR = cv2.resize(imR, (W, H), interpolation=cv2.INTER_AREA)
        draw_matches_gray(imL, imR, h_mkpts0, h_mkpts1, h_mconf, args.viz)
        print(f"Saved viz to {args.viz}")

if __name__ == "__main__":
    main()
