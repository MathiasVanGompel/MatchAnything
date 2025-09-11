#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ------------------------- image utils -------------------------
def load_rgb_tensor(path, size=(832, 832), norm="none"):
    """
    Read BGR image, resize to `size`, convert to RGB, return:
      - x:  NCHW float32 torch tensor on CPU (normalized per `norm`)
      - img_bgr_orig: original BGR (for viz)
      - img_bgr_resz: resized BGR (for viz in network space)
    norm in {"none","imagenet","neg_one_one"}
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    if size is not None:
        img_bgr_resz = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LANCZOS4)
    else:
        img_bgr_resz = img_bgr

    img_rgb = cv2.cvtColor(img_bgr_resz, cv2.COLOR_BGR2RGB).astype(np.float32)

    # scale/normalize
    if norm == "imagenet":
        # [0,1], then (x-mean)/std per channel
        x = img_rgb / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
        x = (x - mean) / std
    elif norm == "neg_one_one":
        # [-1, 1]
        x = (img_rgb / 127.5) - 1.0
    else:
        # [0,1]
        x = img_rgb / 255.0

    # HWC -> NCHW
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous().float()
    return x, img_bgr, img_bgr_resz


def to_numpy(t):  # detach-safe
    return t.detach().cpu().numpy()


# ------------------------- TensorRT runner (TRT-10) -------------------------
class TrtRunner:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()

        # Name-based I/O (per your engine)
        self.in0 = "image0"
        self.in1 = "image1"
        self.out_k0 = "keypoints0"
        self.out_k1 = "keypoints1"
        self.out_mc = "mconf"

    @torch.inference_mode()
    def __call__(self, x0: torch.Tensor, x1: torch.Tensor):
        assert x0.is_cuda and x1.is_cuda, "Inputs must be on CUDA (use .cuda())"
        B, C, H, W = x0.shape

        # Bind dynamic shapes
        self.ctx.set_input_shape(self.in0, (B, C, H, W))
        self.ctx.set_input_shape(self.in1, (B, C, H, W))

        # Zero-copy inputs
        self.ctx.set_tensor_address(self.in0, int(x0.data_ptr()))
        self.ctx.set_tensor_address(self.in1, int(x1.data_ptr()))

        # Allocate worst-case outputs (grid ~ (H/16)*(W/16))
        grid = (H // 16) * (W // 16)
        k0_buf = torch.empty((grid, 2), dtype=torch.float32, device="cuda")
        k1_buf = torch.empty((grid, 2), dtype=torch.float32, device="cuda")
        mc_buf = torch.empty((grid, 1), dtype=torch.float32, device="cuda")

        self.ctx.set_tensor_address(self.out_k0, int(k0_buf.data_ptr()))
        self.ctx.set_tensor_address(self.out_k1, int(k1_buf.data_ptr()))
        self.ctx.set_tensor_address(self.out_mc, int(mc_buf.data_ptr()))

        # Execute (default stream 0)
        self.ctx.execute_async_v3(stream_handle=0)

        # Query actual produced N
        n = self.ctx.get_tensor_shape(self.out_k0)[0]
        if n == 0:
            return (
                np.empty((0, 2), np.float32),
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
            )
        k0 = to_numpy(k0_buf[:n])
        k1 = to_numpy(k1_buf[:n])
        mc = to_numpy(mc_buf[:n, 0])
        return k0, k1, mc


# ------------------------- geometry / filters -------------------------
def ransac_filter(pts0, pts1, reproj=3.0, conf=0.999, maxIters=5000):
    if len(pts0) < 8:
        return None, np.zeros(len(pts0), dtype=bool)
    H, mask = cv2.findHomography(
        pts0,
        pts1,
        cv2.RANSAC,
        ransacReprojThreshold=reproj,
        confidence=conf,
        maxIters=maxIters,
    )
    inl = (mask.ravel() > 0) if mask is not None else np.zeros(len(pts0), dtype=bool)
    return H, inl


def pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return an (N, M) matrix of squared Euclidean distances between
    a (N,2) and b (M,2), using broadcasting.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    # (N,1,2) - (1,M,2) -> (N,M,2)  then sum over last axis -> (N,M)
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def mutual_nn(pts0: np.ndarray, pts1: np.ndarray, eps: float = 3.0) -> np.ndarray:
    """
    Mutual nearest neighbors within eps (in *resized* pixel units).
    Returns a boolean mask of length N (for pts0).
    """
    if len(pts0) == 0 or len(pts1) == 0:
        return np.zeros(len(pts0), dtype=bool)

    D = pairwise_sqdist(pts0, pts1)  # (N, M)

    # nearest neighbor of each a_i in B, and of each b_j in A
    j = np.argmin(D, axis=1).astype(np.int64)  # (N,)
    i_back = np.argmin(D, axis=0).astype(np.int64)  # (M,)

    idx = np.arange(len(pts0), dtype=np.int64)
    keep = i_back[j] == idx  # mutual check

    # distance check
    dmin = D[idx, j]  # (N,)
    keep &= dmin <= (eps * eps)

    return keep


def grid_nms_keep(pts, scores, cell=12):
    """
    One best match per (floor(x/cell), floor(y/cell)) bin.
    Keeps index with largest score per bin.
    """
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    bins = np.floor(pts / float(cell)).astype(int)
    out = {}
    for i, (bx, by) in enumerate(bins):
        key = (bx, by)
        if key not in out or scores[i] > scores[out[key]]:
            out[key] = i
    keep_idx = np.array(sorted(out.values()), dtype=int)
    keep = np.zeros(len(pts), dtype=bool)
    keep[keep_idx] = True
    return keep


# ------------------------- drawing -------------------------
def draw_matches_side_by_side(imgA, imgB, ptsA, ptsB, conf=None, max_draw=1000):
    A = imgA.copy()
    B = imgB.copy()
    if A.ndim == 2:
        A = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)
    if B.ndim == 2:
        B = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
    h = max(A.shape[0], B.shape[0])
    canvas = np.zeros((h, A.shape[1] + B.shape[1], 3), dtype=np.uint8)
    canvas[: A.shape[0], : A.shape[1]] = A
    canvas[: B.shape[0], A.shape[1] : A.shape[1] + B.shape[1]] = B

    N = ptsA.shape[0]
    order = np.arange(N) if conf is None else np.argsort(conf)[::-1]
    order = order[: min(max_draw, N)]

    offx = A.shape[1]
    for i in order:
        x0, y0 = ptsA[i]
        x1, y1 = ptsB[i]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + offx)), int(round(y1)))
        # points
        cv2.circle(canvas, p0, 2, (0, 255, 0), -1)
        cv2.circle(canvas, p1, 2, (0, 255, 0), -1)
        # line
        cv2.line(canvas, p0, p1, (0, 200, 255), 1, cv2.LINE_AA)
    return canvas


# ------------------------- inference wrapper -------------------------
def run_infer(engine_path, img0_path, img1_path, size=832, norm="none"):
    x0, img0_bgr_orig, img0_bgr_resz = load_rgb_tensor(
        img0_path, (size, size), norm=norm
    )
    x1, img1_bgr_orig, img1_bgr_resz = load_rgb_tensor(
        img1_path, (size, size), norm=norm
    )
    x0 = x0.cuda()
    x1 = x1.cuda()
    runner = TrtRunner(engine_path)
    k0, k1, mconf = runner(x0, x1)
    return k0, k1, mconf, (img0_bgr_orig, img1_bgr_orig), (img0_bgr_resz, img1_bgr_resz)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine", type=str, default="out/matchanything_final_complete.plan"
    )
    ap.add_argument("--img0", required=True)
    ap.add_argument("--img1", required=True)
    ap.add_argument("--size", type=int, default=832)
    ap.add_argument(
        "--norm", type=str, default="none", choices=["none", "imagenet", "neg_one_one"]
    )

    # confidence post-processing
    ap.add_argument(
        "--use_conf",
        action="store_true",
        help="If set, apply your own threshold to mconf; otherwise trust engine mask.",
    )
    ap.add_argument("--conf_th", type=float, default=0.05)
    ap.add_argument(
        "--topk", type=int, default=0, help="If >0, keep only top-K by mconf."
    )

    # matching cleanups
    ap.add_argument(
        "--mutual", action="store_true", help="Mutual NN filter in pixel space."
    )
    ap.add_argument(
        "--eps", type=float, default=12.0, help="Max pixel distance for mutual NN."
    )
    ap.add_argument(
        "--grid_nms", type=int, default=0, help="If >0, one best per cell (pixels)."
    )

    # other
    ap.add_argument("--no_ransac", action="store_true", help="Skip RANSAC refinement.")
    ap.add_argument(
        "--swap_xy",
        action="store_true",
        help="Swap x/y columns in both keypoint arrays.",
    )
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args()

    # Inference
    k0, k1, mconf, (img0_bgr_orig, img1_bgr_orig), (img0_bgr_resz, img1_bgr_resz) = (
        run_infer(args.engine, args.img0, args.img1, size=args.size, norm=args.norm)
    )

    print("TRT returned:", k0.shape, k1.shape, mconf.shape, "matches")
    finite = np.isfinite(mconf)
    print(
        f"mconf stats: finite={finite.sum()}/{mconf.size}, "
        f"min={np.nanmin(mconf):.6g}, max={np.nanmax(mconf):.6g}, mean={np.nanmean(mconf):.6g}"
    )

    # Optional axis swap (if your build exported (y,x) order)
    if args.swap_xy:
        k0 = k0[:, [1, 0]]
        k1 = k1[:, [1, 0]]

    pts0 = k0.copy()
    pts1 = k1.copy()
    scores = mconf.copy()

    # Confidence threshold
    if args.use_conf:
        keep = np.isfinite(scores) & (scores >= args.conf_th)
        print(
            f"Kept {int(keep.sum())} / {len(scores)} matches @ conf_th={args.conf_th}"
        )
        if keep.any():
            pts0 = pts0[keep]
            pts1 = pts1[keep]
            scores = scores[keep]
        else:
            print(
                "[WARN] Threshold removed all matches; falling back to engine matches."
            )

    # Optional top-K
    if args.topk and len(scores) > args.topk:
        order = np.argsort(np.nan_to_num(scores, nan=-1.0))[::-1][: args.topk]
        pts0 = pts0[order]
        pts1 = pts1[order]
        scores = scores[order]
        print(f"Top-K: kept {len(scores)} matches")

    # Mutual NN
    if args.mutual and len(pts0) > 0:
        keep = mutual_nn(pts0, pts1, eps=args.eps)
        print(f"Mutual keep: {int(keep.sum())}/{len(pts0)}")
        pts0 = pts0[keep]
        pts1 = pts1[keep]
        scores = scores[keep]

    # Grid NMS on target image coords (deduplicate nearby lines)
    if args.grid_nms and len(pts1) > 0:
        keep = grid_nms_keep(pts1, scores, cell=int(args.grid_nms))
        print(f"Grid-NMS({args.grid_nms}px): kept {int(keep.sum())}/{len(pts1)}")
        pts0 = pts0[keep]
        pts1 = pts1[keep]
        scores = scores[keep]

    # RANSAC homography (in resized/network space)
    inliers = None
    if not args.no_ransac and len(pts0) >= 8:
        H, inliers = ransac_filter(pts0, pts1)
        if inliers is not None and inliers.any():
            print(f"RANSAC inliers: {inliers.sum()}/{len(inliers)}")
        else:
            print("RANSAC skipped or no inliers; using current matches.")
            inliers = None
    elif args.no_ransac:
        print("RANSAC disabled.")
    else:
        print("Not enough points for RANSAC; using current matches.")

    # Choose what to draw
    if inliers is not None:
        p0_draw = pts0[inliers]
        p1_draw = pts1[inliers]
        s_draw = scores[inliers]
    else:
        p0_draw = pts0
        p1_draw = pts1
        s_draw = scores

    # Prepare mapping to original-size coords
    h0o, w0o = img0_bgr_orig.shape[:2]
    h1o, w1o = img1_bgr_orig.shape[:2]
    sx0, sy0 = w0o / float(args.size), h0o / float(args.size)
    sx1, sy1 = w1o / float(args.size), h1o / float(args.size)

    # Draw (resized space)
    vis_resized = draw_matches_side_by_side(
        img0_bgr_resz, img1_bgr_resz, p0_draw, p1_draw, conf=s_draw, max_draw=1000
    )

    # Draw (original space)
    p0_orig = np.stack([p0_draw[:, 0] * sx0, p0_draw[:, 1] * sy0], axis=1)
    p1_orig = np.stack([p1_draw[:, 0] * sx1, p1_draw[:, 1] * sy1], axis=1)
    vis_orig = draw_matches_side_by_side(
        img0_bgr_orig, img1_bgr_orig, p0_orig, p1_orig, conf=s_draw, max_draw=1000
    )

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem0 = Path(args.img0).stem
    stem1 = Path(args.img1).stem
    out_resz = out_dir / f"vis_resized_{stem0}_vs_{stem1}.png"
    out_orig = out_dir / f"vis_{stem0}_vs_{stem1}.png"
    ok1 = cv2.imwrite(str(out_resz), vis_resized)
    ok2 = cv2.imwrite(str(out_orig), vis_orig)
    print(f"[SAVE] Resized-space viz -> {out_resz} (ok={ok1})")
    print(f"[SAVE] Original-size viz -> {out_orig} (ok={ok2})")


if __name__ == "__main__":
    main()
