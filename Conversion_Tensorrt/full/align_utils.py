#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_utils.py — warp image1 so it aligns to image0.

Two modes:
  - dense:  use model's (warp, cert) output and cv2.remap to warp image1
  - homo :  use sparse matches -> RANSAC homography -> cv2.warpPerspective

Refs:
- OpenCV remap mapping semantics and tutorial.  https://docs.opencv.org/3.4/d1/da0/tutorial_remap.html
- findHomography + warpPerspective docs.       https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
"""
from __future__ import annotations
import cv2, numpy as np
from pathlib import Path
import argparse

# Common helper utilities.
def _alpha_blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (alpha * a + (1.0 - alpha) * b).clip(0, 255).astype(np.uint8)

def _checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
    H, W = a.shape[:2]
    mask = (((np.indices((H, W)).sum(axis=0) // tile) & 1) == 0)[..., None]
    out = a.copy()
    out[~mask] = b[~mask]
    return out

# Dense warp derived from the model's coarse flow field.
def dense_warp_align(
    im0_resized: np.ndarray,    # RGB uint8 image shaped [H, W, 3] at the optimization size.
    im1_resized: np.ndarray,    # RGB uint8 image shaped [H, W, 3] at the optimization size.
    warp: np.ndarray,           # Model output shaped [1, 2, gh, gw] stored as float32 or float16.
    cert: np.ndarray | None,    # Model output shaped [1, 1, gh, gw], or None when certainty is absent.
    use_cert_mask: bool = False,
    border: int = cv2.BORDER_REFLECT101,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp im1 into im0's (resized) frame using dense mapping and return:
        warped1_resized, blended_overlay
    """
    H, W = im0_resized.shape[:2]
    warp = warp.astype(np.float32, copy=False)

    # Upsample the warp to full resolution and convert to absolute pixel coordinates.
    # This mirrors the math used when extracting sparse matches.
    gh, gw = warp.shape[2], warp.shape[3]
    wx = cv2.resize(warp[0, 0], (W, H), interpolation=cv2.INTER_LINEAR)  # Map remains in grid units.
    wy = cv2.resize(warp[0, 1], (W, H), interpolation=cv2.INTER_LINEAR)
    patch_x = float(W) / float(gw)
    patch_y = float(H) / float(gh)
    map_x = (wx + 0.5) * patch_x
    map_y = (wy + 0.5) * patch_y

    # Clamp to the valid sampling range.
    map_x = np.clip(map_x, 0, W - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, H - 1).astype(np.float32)

    warped1 = cv2.remap(im1_resized, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border)  # Apply the dense warp with OpenCV remap.

    if use_cert_mask and cert is not None:
        c = cv2.resize(cert.astype(np.float32)[0, 0], (W, H), interpolation=cv2.INTER_LINEAR)
        c = np.clip(c, 0.0, 1.0)[..., None]  # Treat certainty as a soft alpha map.
        blended = (c * im0_resized + (1.0 - c) * warped1).astype(np.uint8)
    else:
        blended = _alpha_blend(im0_resized, warped1, alpha=0.5)

    return warped1, blended

# Homography-based warp computed from sparse matches.
def homography_warp_align(
    im0: np.ndarray,            # RGB uint8 image shaped [H0, W0, 3] at the original size.
    im1: np.ndarray,            # RGB uint8 image shaped [H1, W1, 3] at the original size.
    kpts0: np.ndarray,          # Array of [N, 2] (x, y) keypoints in im0.
    kpts1: np.ndarray,          # Array of [N, 2] (x, y) keypoints in im1.
    ransac_thresh: float = 2.0,
    confidence: float = 0.999,
    maxIters: int = 5000,
    overlay: str = "alpha",     # Options: "alpha" or "checker".
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate H (im1->im0) with RANSAC and warp image1 to image0.
    Returns (warped1, overlay_img, inlier_mask_bool)
    """
    if len(kpts0) < 8 or len(kpts1) < 8:
        raise ValueError("Need at least 8 correspondences for a robust homography.")

    H, mask = cv2.findHomography(kpts1, kpts0, cv2.RANSAC,
                                 ransacReprojThreshold=ransac_thresh,
                                 maxIters=maxIters, confidence=confidence)  # Estimate the homography with OpenCV RANSAC.
    if H is None:
        raise RuntimeError("cv2.findHomography failed.")

    H0, W0 = im0.shape[:2]
    warped1 = cv2.warpPerspective(im1, H, (W0, H0), flags=cv2.INTER_LINEAR)  # Warp image1 into image0's frame with OpenCV.

    if overlay == "checker":
        ov = _checkerboard(cv2.cvtColor(im0, cv2.COLOR_RGB2BGR),
                           cv2.cvtColor(warped1, cv2.COLOR_RGB2BGR),
                           tile=32)
    else:
        ov = _alpha_blend(cv2.cvtColor(im0, cv2.COLOR_RGB2BGR),
                          cv2.cvtColor(warped1, cv2.COLOR_RGB2BGR), alpha=0.5)

    inliers = mask.ravel().astype(bool)
    return warped1, ov, inliers

# Small CLI for quick local experiments.
def _load_rgb(p): 
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _resize_rgb(img, size): return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "homo"], required=True)
    ap.add_argument("--im0", required=True)
    ap.add_argument("--im1", required=True)
    ap.add_argument("--opt", type=int, default=518)
    ap.add_argument("--warp_npy", help="Path to npz with keys warp, cert (from your TRT forward)")
    ap.add_argument("--matches_npz", help="Path to npz with kpts0,kpts1 (and optional inliers)")
    ap.add_argument("--outdir", default="Conversion_Tensorrt/out/align")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    im0 = _load_rgb(args.im0); im1 = _load_rgb(args.im1)

    if args.mode == "dense":
        assert args.warp_npy, "--warp_npy is required for dense mode"
        data = np.load(args.warp_npy)
        warp = data["warp"]; cert = data["cert"] if "cert" in data else None
        im0r = _resize_rgb(im0, args.opt)
        im1r = _resize_rgb(im1, args.opt)
        warped1, overlay = dense_warp_align(im0r, im1r, warp, cert, use_cert_mask=False)
        cv2.imwrite(str(Path(args.outdir)/"dense_warped1.jpg"), cv2.cvtColor(warped1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(Path(args.outdir)/"dense_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("[OK] saved dense_warped1.jpg and dense_overlay.jpg")

    else:
        assert args.matches_npz, "--matches_npz is required for homo mode"
        data = np.load(args.matches_npz)
        k0, k1 = data["kpts0"].astype(np.float32), data["kpts1"].astype(np.float32)
        warped1, overlay, inl = homography_warp_align(im0, im1, k0, k1, ransac_thresh=2.0)
        np.savez_compressed(Path(args.outdir)/"H_inliers.npz", inliers=inl)
        cv2.imwrite(str(Path(args.outdir)/"homo_warped1.jpg"), cv2.cvtColor(warped1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(Path(args.outdir)/"homo_overlay.jpg"), overlay)
        print(f"[OK] saved homo_warped1.jpg / homo_overlay.jpg | inliers: {inl.sum()} / {len(inl)}")

if __name__ == "__main__":
    main()
