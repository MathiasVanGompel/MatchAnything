#!/usr/bin/env python3
import numpy as np, tensorrt as trt, pycuda.driver as cuda


def _np_dtype_for_trt(dt: trt.DataType):
    return {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF:  np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL:  np.bool_,
    }.get(dt, np.float32)

#!/usr/bin/env python3
import numpy as np, tensorrt as trt, pycuda.driver as cuda

def _np_dtype_for_trt(dt: trt.DataType):
    return {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF:  np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL:  np.bool_,
    }.get(dt, np.float32)

class TRTEngine:
    def __init__(self, plan_path: str, log_level=trt.Logger.INFO):
        self.logger = trt.Logger(log_level)
        with open(plan_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:  self.inputs.append(n)
            else:                                                          self.outputs.append(n)
        print(f"[TRT] Loaded {plan_path}")
        print("  inputs :", self.inputs)
        print("  outputs:", self.outputs)

    def infer(self, feed_dict):
        stream = cuda.Stream()
        d_in, d_out, h_out = {}, {}, {}

        try:
            # Inputs: cast + make contiguous + set shapes/addresses
            for name, arr in feed_dict.items():
                exp_dt = _np_dtype_for_trt(self.engine.get_tensor_dtype(name))
                arr = np.ascontiguousarray(arr, dtype=exp_dt)  # <-- fix
                self.ctx.set_input_shape(name, tuple(arr.shape))
                d_in[name] = cuda.mem_alloc(arr.nbytes)
                self.ctx.set_tensor_address(name, int(d_in[name]))
                cuda.memcpy_htod_async(d_in[name], arr, stream)

            # Outputs: allocate with correct shapes/dtypes
            for name in self.outputs:
                shp = tuple(self.ctx.get_tensor_shape(name))
                dt = _np_dtype_for_trt(self.engine.get_tensor_dtype(name))
                h_out[name] = np.empty(shp, dtype=dt, order="C")
                d_out[name] = cuda.mem_alloc(h_out[name].nbytes)
                self.ctx.set_tensor_address(name, int(d_out[name]))

            # Run
            ok = self.ctx.execute_async_v3(stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed.")

            # D2H
            for name in self.outputs:
                cuda.memcpy_dtoh_async(h_out[name], d_out[name], stream)
            stream.synchronize()
            return h_out

        finally:
            # Let PyCUDA free allocations when objects go out of scope;
            # stream is destroyed here as it leaves scope.
            pass

