#!/usr/bin/env python3
import argparse, os, sys, numpy as np, cv2, tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize(img_rgb_f32, norm):
    if norm == "imagenet": return (img_rgb_f32 - IMNET_MEAN) / IMNET_STD
    if norm == "minus11":  return img_rgb_f32 * 2.0 - 1.0
    return img_rgb_f32

def load_image_tensor(path, H, W, to_dtype, norm="none"):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if H and W: rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = normalize(rgb, norm)
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, 0).astype(to_dtype, copy=False)

def _norm_to_pix(xn, size): return np.clip((xn + 1.0) * (size - 1) / 2.0, 0, size - 1)

def to_uint8(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).any(): return np.zeros_like(x, dtype=np.uint8)
    vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
    if (vmax - vmin) < 1e-8: return np.zeros_like(x, dtype=np.uint8)
    x = (np.clip((x - vmin) / (vmax - vmin), 0, 1) * 255.0)
    x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
    return x.astype(np.uint8)

def matches_from_warp_cert(warp, cert, topk=1000):
    _, C, H, W = warp.shape
    if C != 4: return np.empty((0,4), np.float32)
    c = cert[0,0].reshape(-1)
    idx = np.argsort(-c)[:max(1, min(topk, c.size))]
    y = idx // W; x = idx % W
    x0 = _norm_to_pix(warp[0,0,y,x], W); y0 = _norm_to_pix(warp[0,1,y,x], H)
    x1 = _norm_to_pix(warp[0,2,y,x], W); y1 = _norm_to_pix(warp[0,3,y,x], H)
    valid = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    return np.stack([x0[valid], y0[valid], x1[valid], y1[valid]], axis=1).astype(np.float32)

def draw_side_by_side(img0_path, img1_path, W, H, matches, out_png):
    left  = cv2.resize(cv2.imread(img0_path, cv2.IMREAD_COLOR), (W, H))
    right = cv2.resize(cv2.imread(img1_path, cv2.IMREAD_COLOR), (W, H))
    side = cv2.hconcat([left, right])
    if matches is not None and len(matches) > 0:
        draw = side.copy()
        for x0,y0,x1,y1 in matches:
            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1)) + W, int(round(y1)))
            cv2.circle(draw, p0, 2, (0,255,0), -1)
            cv2.circle(draw, p1, 2, (0,255,0), -1)
            cv2.line(draw, p0, p1, (0,255,0), 1, cv2.LINE_AA)
        side = draw
    cv2.imwrite(out_png, side)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--image0", required=True)
    ap.add_argument("--image1", required=True)
    ap.add_argument("--outdir", default="Convertion_Tensorrt/out/out_trt")
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--norm", choices=["imagenet","minus11","none"], default="none")
    ap.add_argument("--budget", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    with open(args.engine, "rb") as f:
        runtime = trt.Runtime(logger)
        engine  = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    io = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    mode = {n: engine.get_tensor_mode(n) for n in io}
    inputs  = [n for n in io if mode[n] == trt.TensorIOMode.INPUT]
    outputs = [n for n in io if mode[n] == trt.TensorIOMode.OUTPUT]
    if "image0" not in inputs or "image1" not in inputs:
        print("I/O:", io, file=sys.stderr)
        raise RuntimeError("Engine must expose 'image0' and 'image1'")

    def ensure_shape(name):
        shp = ctx.get_tensor_shape(name)
        if -1 in tuple(shp):
            if args.H is None or args.W is None:
                raise RuntimeError(f"Dynamic shape for {name} not set; pass --H --W.")
            ctx.set_input_shape(name, (1, 3, args.H, args.W))
            shp = ctx.get_tensor_shape(name)
        return tuple(shp)

    s0 = ensure_shape("image0"); s1 = ensure_shape("image1")
    H, W = s0[2], s0[3]

    dtype0 = trt.nptype(engine.get_tensor_dtype("image0"))
    dtype1 = trt.nptype(engine.get_tensor_dtype("image1"))
    inp0 = load_image_tensor(args.image0, H, W, to_dtype=dtype0, norm=args.norm)
    inp1 = load_image_tensor(args.image1, H, W, to_dtype=dtype1, norm=args.norm)

    stream = cuda.Stream()
    def bind_input(name, host):
        dptr = cuda.mem_alloc(host.nbytes)
        ctx.set_tensor_address(name, int(dptr))
        cuda.memcpy_htod_async(dptr, host, stream)
        return dptr
    def bind_output(name):
        shape = tuple(ctx.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host = np.empty(shape, dtype=dtype)
        dptr = cuda.mem_alloc(host.nbytes)
        ctx.set_tensor_address(name, int(dptr))
        return dptr, host

    d_i0 = bind_input("image0", inp0)
    d_i1 = bind_input("image1", inp1)
    d_out, host_out = {}, {}
    for name in outputs:
        dptr, host = bind_output(name)
        d_out[name] = dptr; host_out[name] = host

    ok = ctx.execute_async_v3(stream.handle)
    if not ok: raise RuntimeError("TensorRT execute_async_v3 failed")
    for name, dptr in d_out.items():
        cuda.memcpy_dtoh_async(host_out[name], dptr, stream)
    stream.synchronize()

    if "cert" in host_out and host_out["cert"].ndim == 4 and host_out["cert"].shape[1] == 1:
        cv2.imwrite(os.path.join(args.outdir, "cert.png"), to_uint8(host_out["cert"][0,0]))

    matches = None
    if "warp" in host_out and "cert" in host_out:
        matches = matches_from_warp_cert(host_out["warp"].astype(np.float32),
                                         host_out["cert"].astype(np.float32),
                                         topk=args.budget)

    draw_png = os.path.join(args.outdir, "matches_green.png")
    draw_side_by_side(args.image0, args.image1, W, H, matches, draw_png)
    print("Wrote:", draw_png)

if __name__ == "__main__":
    main()