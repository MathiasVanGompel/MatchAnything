#!/usr/bin/env python3
import argparse, cv2, numpy as np
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

# TensorRT 10.x helper that binds tensors by name.
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
        n = int(np.prod(shape)) if len(shape) else 1
        need = n * self._elt_size(dtype)
        if self.nbytes[name] != need:
            if self.device_buffers[name] is not None:
                self.cuda.mem_free(self.device_buffers[name])
            self.device_buffers[name] = self.cuda.mem_alloc(need)
            self.host_buffers[name] = np.empty(shape, dtype=self.trt.nptype(dtype))
            self.nbytes[name] = need

    def infer(self, feeds: dict):
        trt, cuda = self.trt, self.cuda
        # Set input shapes on the execution context.
        for n in self.io_names:
            if self.is_input[n] and n in feeds:
                arr = np.asarray(feeds[n])
                self.context.set_input_shape(n, tuple(arr.shape))
        # Allocate device buffers and bind tensor addresses.
        for n in self.io_names:
            shp = tuple(self.context.get_tensor_shape(n))
            dt  = self.engine.get_tensor_dtype(n)
            self._ensure_alloc(n, shp, dt)
            self.context.set_tensor_address(n, int(self.device_buffers[n]))
        # Copy host inputs to device memory, casting to the expected dtype.
        for n, arr in feeds.items():
            exp = self.engine.get_tensor_dtype(n)
            npdt = np.dtype(trt.nptype(exp))
            if arr.dtype != npdt: arr = arr.astype(npdt, copy=False)
            arr = np.ascontiguousarray(arr)
            cuda.memcpy_htod_async(self.device_buffers[n], arr, self.stream)
        # Execute the TensorRT engine.
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # Copy device outputs back to host memory.
        outs = {}
        for n in self.io_names:
            if not self.is_input[n]:
                host = self.host_buffers[n]
                self.cuda.memcpy_dtoh_async(host, self.device_buffers[n], self.stream)
                outs[n] = host
        self.stream.synchronize()
        # Cast outputs for OpenCV compatibility.
        for k in outs:
            if outs[k].dtype == np.float16:
                outs[k] = outs[k].astype(np.float32)
        return outs

def _load_rgb(path): 
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _pre(img, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32)/255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2,0,1))[None, ...]  # Arrange as NCHW.
    return np.ascontiguousarray(x, np.float32), img  # Also return the resized uint8 image for reference.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--image0", required=True)
    ap.add_argument("--image1", required=True)
    ap.add_argument("--opt", type=int, default=518)
    ap.add_argument("--outdir", default="Conversion_Tensorrt/out/full_results")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    net = TRTEngine(args.engine)

    im0 = _load_rgb(args.image0)
    im1 = _load_rgb(args.image1)
    x0, _ = _pre(im0, args.opt)
    x1, _ = _pre(im1, args.opt)

    outs = net.infer({"image0": x0, "image1": x1})
    # Expect keys 'warp' (1×2×gh×gw) and 'cert' (1×1×gh×gw).
    base = f"{Path(args.image0).stem}__{Path(args.image1).stem}"
    np.savez_compressed(Path(args.outdir)/f"warp_cert_{base}.npz",
                        warp=outs["warp"], cert=outs["cert"])
    print(f"[OK] wrote warp+cert to {Path(args.outdir)/f'warp_cert_{base}.npz'}")
    # Sanity check.
    print("warp:", outs["warp"].shape, outs["warp"].dtype,
          "| cert:", outs["cert"].shape, outs["cert"].dtype)

if __name__ == "__main__":
    main()
