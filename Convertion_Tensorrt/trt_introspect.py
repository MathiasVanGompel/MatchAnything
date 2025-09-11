# trt_introspect.py
import argparse
import tensorrt as trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    with open(args.engine, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    print(f"[INFO] TRT version: {trt.__version__}")
    n = engine.num_io_tensors
    print(f"[INFO] Tensors in engine: {n}")
    for i in range(n):
        name = engine.get_tensor_name(i)
        role = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(
            f" - {name:30s} | {str(role).split('.')[-1]:6s} | shape={shape} | dtype={dtype}"
        )


if __name__ == "__main__":
    main()
