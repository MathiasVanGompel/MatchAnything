#!/usr/bin/env python3
import os, argparse
from pathlib import Path
import numpy as np
import cv2, torch, tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_rgb_tensor(path, size=None):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    if size:
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(np.transpose(img_rgb, (2,0,1))[None, ...]).contiguous().float()
    return t, img_bgr

def draw_matches_side_by_side(imgA, imgB, ptsA, ptsB, conf=None, max_draw=1000):
    A = imgA.copy(); B = imgB.copy()
    if A.ndim == 2: A = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)
    if B.ndim == 2: B = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
    H = max(A.shape[0], B.shape[0])
    canvas = np.zeros((H, A.shape[1] + B.shape[1], 3), dtype=np.uint8)
    canvas[:A.shape[0], :A.shape[1]] = A
    canvas[:B.shape[0], A.shape[1]:A.shape[1]+B.shape[1]] = B
    offx = A.shape[1]
    N = ptsA.shape[0]
    order = np.argsort(np.nan_to_num(conf, nan=-1.0))[::-1] if conf is not None else np.arange(N)
    order = order[:min(N, max_draw)]
    for i in order:
        x0, y0 = ptsA[i]; x1, y1 = ptsB[i]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + offx)), int(round(y1)))
        cv2.circle(canvas, p0, 2, (0,255,0), -1)
        cv2.circle(canvas, p1, 2, (0,255,0), -1)
        cv2.line(canvas, p0, p1, (0,200,255), 1, cv2.LINE_AA)
    return canvas

class TrtRunner:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in0 = "image0"; self.in1 = "image1"
        self.outW = "warp_c"; self.outC = "cert_c"
        self.outM = "valid_mask"; self.outS = "coarse_stride"

    @torch.inference_mode()
    def __call__(self, x0: torch.Tensor, x1: torch.Tensor):
        assert x0.is_cuda and x1.is_cuda
        B, C, H, W = x0.shape
        self.ctx.set_input_shape(self.in0, (B, C, H, W))
        self.ctx.set_input_shape(self.in1, (B, C, H, W))
        self.ctx.set_tensor_address(self.in0, int(x0.data_ptr()))
        self.ctx.set_tensor_address(self.in1, int(x1.data_ptr()))

        shapeW = self.ctx.get_tensor_shape(self.outW)  # (B,Ha,Wa,2)
        shapeC = self.ctx.get_tensor_shape(self.outC)  # (B,Ha,Wa)
        shapeM = self.ctx.get_tensor_shape(self.outM)  # (B,Ha,Wa)
        shapeS = self.ctx.get_tensor_shape(self.outS)  # (1,)

        warp_buf  = torch.empty(tuple(shapeW), dtype=torch.float32, device="cuda")
        cert_buf  = torch.empty(tuple(shapeC), dtype=torch.float32, device="cuda")
        mask_buf  = torch.empty(tuple(shapeM), dtype=torch.float32, device="cuda")
        stridebuf = torch.empty(tuple(shapeS), dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.outW, int(warp_buf.data_ptr()))
        self.ctx.set_tensor_address(self.outC, int(cert_buf.data_ptr()))
        self.ctx.set_tensor_address(self.outM, int(mask_buf.data_ptr()))
        self.ctx.set_tensor_address(self.outS, int(stridebuf.data_ptr()))

        self.ctx.execute_async_v3(stream_handle=0)
        return warp_buf, cert_buf, mask_buf, stridebuf

def ransac_filter(pts0, pts1, reproj=3.0, conf=0.999, maxIters=5000):
    if len(pts0) < 8:
        return None, np.zeros(len(pts0), dtype=bool)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransacReprojThreshold=reproj,
                                 confidence=conf, maxIters=maxIters)
    inl = (mask.ravel() > 0) if mask is not None else np.zeros(len(pts0), dtype=bool)
    return H, inl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, default="out/matchanything_dense_dynamic.plan")
    ap.add_argument("--img0", required=True)
    ap.add_argument("--img1", required=True)
    ap.add_argument("--size", type=int, default=0, help="Optional square resize. 0=keep originals.")
    ap.add_argument("--conf_th", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--no_ransac", action="store_true")
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args()

    x0, im0 = load_rgb_tensor(args.img0, size=(args.size or None))
    x1, im1 = load_rgb_tensor(args.img1, size=(args.size or None))
    x0 = x0.cuda(); x1 = x1.cuda()

    runner = TrtRunner(args.engine)
    warp_c, cert_c, valid_c, stride_t = runner(x0, x1)  # [1,Ha,Wa,2], [1,Ha,Wa], [1,Ha,Wa], [1]

    # Build coarse grid for image0
    _, Ha, Wa, _ = warp_c.shape
    yy, xx = torch.meshgrid(torch.arange(Ha, device="cuda"),
                            torch.arange(Wa, device="cuda"), indexing="ij")
    grid0_c = torch.stack([xx, yy], dim=-1).float()   # [Ha,Wa,2]

    # Flatten + host filtering using mask + conf_th (+topK)
    k0_c = grid0_c.view(-1, 2).cpu().numpy()
    k1_c = warp_c[0].view(-1, 2).detach().cpu().numpy()
    mconf = cert_c[0].reshape(-1).detach().cpu().numpy()
    mval  = valid_c[0].reshape(-1).detach().cpu().numpy()

    keep = (mval > 0.5) & np.isfinite(mconf) & (mconf >= args.conf_th)
    idx  = np.flatnonzero(keep)
    if args.topk and idx.size > args.topk:
        order = np.argsort(np.nan_to_num(mconf[idx], nan=-1.0))[::-1][:args.topk]
        idx   = idx[order]

    stride = float(stride_t[0].item())
    k0 = k0_c[idx] * stride
    k1 = k1_c[idx] * stride
    scores = mconf[idx]

    print(f"Kept {len(scores)} matches @ conf_th={args.conf_th}")
    if not args.no_ransac and len(scores) >= 8:
        Hm, inl = ransac_filter(k0, k1)
        if inl is not None and inl.any():
            print(f"RANSAC inliers: {inl.sum()}/{len(inl)}")
            k0, k1, scores = k0[inl], k1[inl], scores[inl]

    vis = draw_matches_side_by_side(im0, im1, k0, k1, conf=scores, max_draw=1000)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stem0 = Path(args.img0).stem; stem1 = Path(args.img1).stem
    out_png = out_dir / f"vis_{stem0}_vs_{stem1}.png"
    ok = cv2.imwrite(str(out_png), vis)
    print(f"[SAVE] {out_png} (ok={ok})")

if __name__ == "__main__":
    main()
