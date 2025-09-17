from convert_eloftr_to_trt import (
    load_eloftr_model, export_eloftr_onnx, build_trt_engine_from_onnx
)

# 1) Load model + weights
model, cfg = load_eloftr_model(device="cuda")

# 2) Export ONNX (dynamic batch/H/W)
onnx_path = "/home/mathias/MatchAnything-1/eloftr_dynamic.onnx"
export_eloftr_onnx(model, onnx_path, opset=17, sample_hw=(480, 640))

# 3) Build a TensorRT engine (FP16)
engine_path = "/home/mathias/MatchAnything-1/eloftr_dynamic_fp16.plan"
build_trt_engine_from_onnx(
    onnx_path="/home/mathias/MatchAnything-1/eloftr_dynamic.onnx",
    engine_path="/home/mathias/MatchAnything-1/eloftr_fp16_lowmem.plan",
    min_hw=(240, 320),
    opt_hw=(480, 640),
    max_hw=(640, 960),
    fp16=True,
    workspace_mb=512,
    builder_optimization_level=1,
)
