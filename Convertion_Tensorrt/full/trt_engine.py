# Convertion_Tensorrt/full/trt_engine.py
from typing import Dict, List
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # creates a CUDA context

def _np_dtype_for_trt(dt: trt.DataType):
    if dt == trt.DataType.FLOAT: return np.float32
    if dt == trt.DataType.HALF:  return np.float16
    if dt == trt.DataType.INT32: return np.int32
    if dt == trt.DataType.BOOL:  return np.bool_
    return np.float32

class TRTEngine:
    def __init__(self, engine_path: str, log_level: int = trt.Logger.INFO):
        self.logger = trt.Logger(log_level)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        # discover I/O tensors once
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

        print(f"[TRT] Loaded {engine_path}")
        print("  inputs :", self.inputs)
        print("  outputs:", self.outputs)

    # <-- the tiny helper your runner expects
    def io_tensors(self):
        return self.inputs, self.outputs

    def infer(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        stream = cuda.Stream()
        d_in: Dict[str, cuda.DeviceAllocation] = {}
        d_out: Dict[str, cuda.DeviceAllocation] = {}
        h_out: Dict[str, np.ndarray] = {}

        try:
            # Inputs: cast + contiguous + bind + HtoD
            for name, arr in feed.items():
                exp_dt = _np_dtype_for_trt(self.engine.get_tensor_dtype(name))
                arr = np.ascontiguousarray(arr, dtype=exp_dt)
                self.ctx.set_input_shape(name, tuple(arr.shape))
                buf = cuda.mem_alloc(arr.nbytes)
                d_in[name] = buf
                self.ctx.set_tensor_address(name, int(buf))
                cuda.memcpy_htod_async(buf, arr, stream)

            # Outputs: allocate with correct shapes/dtypes + bind
            for name in self.outputs:
                shp = tuple(self.ctx.get_tensor_shape(name))
                dt  = _np_dtype_for_trt(self.engine.get_tensor_dtype(name))
                host = np.empty(shp, dtype=dt, order="C")
                dev  = cuda.mem_alloc(host.nbytes)
                h_out[name] = host
                d_out[name] = dev
                self.ctx.set_tensor_address(name, int(dev))

            # Run (TensorRT v10 I/O API)
            ok = self.ctx.execute_async_v3(stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed.")

            # D2H
            for name in self.outputs:
                cuda.memcpy_dtoh_async(h_out[name], d_out[name], stream)
            stream.synchronize()
            return h_out

        finally:
            # Let PyCUDA free on GC.
            pass