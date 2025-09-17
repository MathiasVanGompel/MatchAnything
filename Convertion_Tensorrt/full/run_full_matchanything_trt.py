#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, cv2, numpy as np
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# TensorRT 10.x runtime wrapper that relies on name-based tensor I/O.
class TRTEngine:
    def __init__(self, plan_path):
        import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit  # noqa
        self.trt, self.cuda = trt, cuda
        logger = trt.Logger(trt.Logger.ERROR)
        with open(plan_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.is_input = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.io_names}
        self.device_buffers = {n: None for n in self.io_names}
        self.host_buffers   = {n: None for n in self.io_names}
        self.nbytes         = {n: 0    for n in self.io_names}

    def _elt_size(self, dtype): return np.dtype(self.trt.nptype(dtype)).itemsize

    def _ensure_alloc(self, name, shape, dtype):
        n_elem = int(np.prod(shape)) if len(shape) else 1
        need = n_elem * self._elt_size(dtype)
        if self.nbytes[name] != need:
            if self.device_buffers[name] is not None:
                self.cuda.mem_free(self.device_buffers[name])
            self.device_buffers[name] = self.cuda.mem_alloc(need)
            self.host_buffers[name] = np.empty(shape, dtype=self.trt.nptype(dtype))
            self.nbytes[name] = need

    def infer(self, feeds: dict):
        trt, cuda = self.trt, self.cuda

        # Set input shapes from the provided feeds.
        for name in self.io_names:
            if self.is_input[name] and name in feeds:
                arr = np.asarray(feeds[name])
                self.context.set_input_shape(name, tuple(arr.shape))

        # Allocate buffers and bind them to the execution context.
        for name in self.io_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            self._ensure_alloc(name, shape, dtype)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        # Copy host tensors to device memory after matching the engine dtype.
        for name, arr in feeds.items():
            exp_dtype = self.engine.get_tensor_dtype(name)
            np_dtype = np.dtype(trt.nptype(exp_dtype))
            if arr.dtype != np_dtype:
                arr = arr.astype(np_dtype, copy=False)
            arr = np.ascontiguousarray(arr)
            if arr.nbytes > self.nbytes[name]:
                raise RuntimeError(f"Host bytes {arr.nbytes} > device buffer {self.nbytes[name]} for '{name}' "
                                   f"(engine expects dtype {np_dtype}).")
            cuda.memcpy_htod_async(self.device_buffers[name], arr, self.stream)

        # Execute the TensorRT engine.
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy device outputs back to host memory.
        outs = {}
        for name in self.io_names:
            if not self.is_input[name]:
                host = self.host_buffers[name]
                self.cuda.memcpy_dtoh_async(host, self.device_buffers[name], self.stream)
                outs[name] = host
        self.stream.synchronize()

        # Cast to float32 so OpenCV can consume the outputs without additional conversions.
        for k in outs:
            if outs[k].dtype == np.float16:
                outs[k] = outs[k].astype(np.float32)
        return outs

# Pre- and post-processing utilities.
def load_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def preprocess_pair(path0: str, path1: str, size: int):
    im0, im1 = load_rgb(path0), load_rgb(path1)
    H0, W0 = im0.shape[:2]; H1, W1 = im1.shape[:2]
    im0r = cv2.resize(im0, (size, size), interpolation=cv2.INTER_LINEAR)
    im1r = cv2.resize(im1, (size, size), interpolation=cv2.INTER_LINEAR)
    def to_chw(img):
        x = img.astype(np.float32)/255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2,0,1))[None, ...]
        return np.ascontiguousarray(x, dtype=np.float32)  # The TensorRT wrapper will cast if needed.
    return (to_chw(im0r), to_chw(im1r)), (im0, im1), (H0, W0, H1, W1)

def upsample_warp_cert(warp, cert, H, W):
    warp = warp.astype(np.float32, copy=False)
    cert = cert.astype(np.float32, copy=False)
    _, _, gh, gw = warp.shape
    wx = cv2.resize(warp[0,0], (W, H), interpolation=cv2.INTER_LINEAR)
    wy = cv2.resize(warp[0,1], (W, H), interpolation=cv2.INTER_LINEAR)
    c  = cv2.resize(cert[0,0], (W, H), interpolation=cv2.INTER_LINEAR)
    patch_x = float(W)/float(gw); patch_y = float(H)/float(gh)
    tgt_x = (wx + 0.5) * patch_x
    tgt_y = (wy + 0.5) * patch_y
    jj, ii = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    src_x = jj + 0.5; src_y = ii + 0.5
    return src_x, src_y, tgt_x, tgt_y, c

def select_matches(src_x, src_y, tgt_x, tgt_y, cert, topk=5000, conf=0.25, W=518, H=518, sample=0.0):
    thr = max(float(conf), float(sample))
    inb = (tgt_x >= 0) & (tgt_x <= (W-1)) & (tgt_y >= 0) & (tgt_y <= (H-1))
    m = inb & (cert >= thr)
    if not np.any(m):
        z = np.zeros((0,2), np.float32); return z, z.copy(), np.zeros((0,), np.float32)
    c = cert[m]; idx_all = np.flatnonzero(m); k = min(topk, c.size)
    top = idx_all[np.argpartition(-c, k-1)[:k]]
    yk, xk = np.unravel_index(top, cert.shape)
    pts0 = np.stack([src_x[yk, xk], src_y[yk, xk]], 1).astype(np.float32)
    pts1 = np.stack([tgt_x[yk, xk], tgt_y[yk, xk]], 1).astype(np.float32)
    scores = cert[yk, xk].astype(np.float32)
    return pts0, pts1, scores

def mutual_filter(pts01, pts10, tol=4.0):
    p0a, p1a, sA = pts01; p1b, p0b, sB = pts10
    if len(p0a)==0 or len(p1b)==0: return pts01
    def nn(a, b):
        if len(b)==0: return (np.full((len(a),), -1, int), np.full((len(a),), 1e9, np.float32))
        d2 = ((a[:,None,:]-b[None,:,:])**2).sum(-1)
        return d2.argmin(1), d2.min(1)
    idx_b, d2_ab = nn(p1a, p1b); idx_a, d2_ba = nn(p0b, p0a)
    keep = [i for i,(j,da) in enumerate(zip(idx_b, d2_ab)) if j>=0 and idx_a[j]==i and (da**0.5)<tol]
    if not keep: return (p0a[:0], p1a[:0], sA[:0])
    keep = np.asarray(keep, np.int32)
    return (p0a[keep], p1a[keep], sA[keep])

def ransac_filter(pts0, pts1, model="F", rth=3.0, prob=0.999, maxIters=5000):
    if len(pts0) < 8: return pts0, pts1, np.zeros((len(pts0),), bool)
    if model.upper() == "H":
        _, inl = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransacReprojThreshold=rth,
                                    maxIters=maxIters, confidence=prob)
    else:
        _, inl = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC,
                                        ransacReprojThreshold=rth, confidence=prob, maxIters=maxIters)
    inl = inl.ravel().astype(bool) if inl is not None else np.zeros((len(pts0),), bool)
    return pts0[inl], pts1[inl], inl

def draw_matches(im0, im1, p0, p1, mode="lines", line_thickness=3, dot_radius=4, margin=16):
    H0, W0 = im0.shape[:2]; H1, W1 = im1.shape[:2]
    canvas = np.zeros((max(H0,H1)+2*margin, W0+W1+3*margin, 3), np.uint8)
    canvas[margin:margin+H0, margin:margin+W0] = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    canvas[margin:margin+H1, 2*margin+W0:2*margin+W0+W1] = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
    p1s = p1 + np.array([W0+margin, 0], np.float32)[None,:]
    # Draw each match on the concatenated canvas.
    color = (0,255,0)
    if mode == "dots":
        for a, b in zip(p0, p1s):
            a = (int(round(a[0]))+margin, int(round(a[1]))+margin)
            b = (int(round(b[0]))+margin, int(round(b[1]))+margin)
            cv2.circle(canvas, a, dot_radius, color, -1)
            cv2.circle(canvas, b, dot_radius, color, -1)
    else:
        for a, b in zip(p0, p1s):
            a = (int(round(a[0]))+margin, int(round(a[1]))+margin)
            b = (int(round(b[0]))+margin, int(round(b[1]))+margin)
            cv2.circle(canvas, a, dot_radius, color, -1)
            cv2.circle(canvas, b, dot_radius, color, -1)
            cv2.line(canvas, a, b, color, line_thickness, cv2.LINE_AA)
    return canvas

# Entry point for running inference with the TensorRT engine.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--image0", required=True)
    ap.add_argument("--image1", required=True)
    ap.add_argument("--opt", type=int, default=518)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sample", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--ransac", action="store_true")
    ap.add_argument("--model", choices=["F","H"], default="H")
    ap.add_argument("--rth", type=float, default=2.0)
    ap.add_argument("--viz", choices=["cv2"], default="cv2")
    ap.add_argument("--draw", choices=["lines","dots"], default="lines")
    ap.add_argument("--outdir", default="Convertion_Tensorrt/out/full_results")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    net = TRTEngine(args.engine)

    (x0, x1), (im0, im1), (H0, W0, H1, W1) = preprocess_pair(args.image0, args.image1, args.opt)
    outs01 = net.infer({"image0": x0, "image1": x1})
    src_x, src_y, tgt_x, tgt_y, c = upsample_warp_cert(outs01["warp"], outs01["cert"], args.opt, args.opt)
    p0_opt, p1_opt, s01 = select_matches(src_x, src_y, tgt_x, tgt_y, c,
                                         topk=args.topk, conf=args.conf, W=args.opt, H=args.opt, sample=args.sample)
    print(f"[LOG] raw01 matches: {len(p0_opt)} (thr={max(args.conf,args.sample):.3f})")

    if args.mutual:
        (x1b, x0b), *_ = preprocess_pair(args.image1, args.image0, args.opt)
        outs10 = net.infer({"image0": x1b, "image1": x0b})
        sx, sy, tx, ty, cc = upsample_warp_cert(outs10["warp"], outs10["cert"], args.opt, args.opt)
        p1b_opt, p0b_opt, s10 = select_matches(sx, sy, tx, ty, cc,
                                               topk=args.topk, conf=args.conf, W=args.opt, H=args.opt, sample=args.sample)
        p0_opt, p1_opt, s01 = mutual_filter((p0_opt, p1_opt, s01), (p1b_opt, p0b_opt, s10), tol=4.0)
        print(f"[LOG] mutual-filtered matches: {len(p0_opt)}")

    # Rescale matches back to the original image resolution.
    pts0 = p0_opt * np.array([W0/args.opt, H0/args.opt], np.float32)[None,:]
    pts1 = p1_opt * np.array([W1/args.opt, H1/args.opt], np.float32)[None,:]

    base = f"{Path(args.image0).stem}__{Path(args.image1).stem}"
    np.savez_compressed(Path(args.outdir)/f"matches_raw_{base}.npz", kpts0=pts0, kpts1=pts1, scores=s01)

    vis_pairs = (pts0, pts1)
    if args.ransac:
        in0, in1, inl = ransac_filter(pts0, pts1, model=args.model, rth=args.rth)
        print(f"[LOG] RANSAC inliers: {len(in0)} / {len(pts0)}  (model={args.model}, rth={args.rth})")
        np.savez_compressed(Path(args.outdir)/f"matches_ransac_{base}.npz", kpts0=in0, kpts1=in1, inliers=inl)
        vis_pairs = (in0, in1)

    # Warn when no matches survive and report certainty statistics.
    if len(vis_pairs[0]) == 0:
        print("[WARN] No matches to draw. Try lowering --conf (e.g., 0.2).")

    if args.viz == "cv2":
        vis = draw_matches(im0, im1, *vis_pairs, mode=args.draw, line_thickness=3, dot_radius=4, margin=16)
        out_path = Path(args.outdir)/f"matches_{'ransac_' if args.ransac else 'raw_'}{base}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()
