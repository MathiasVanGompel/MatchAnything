#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-LoFTR → ONNX → TensorRT conversion utilities."""

import os
import pathlib
import sys
from importlib import util as importlib_util
from typing import Union

import numpy as np
import torch


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

env_override = os.environ.get("MATCHANYTHING_DIR")
candidate_roots = []
if env_override:
    candidate_roots.append(pathlib.Path(env_override))
candidate_roots.extend(
    [
        REPO_ROOT / "imcui" / "third_party" / "MatchAnything",
        REPO_ROOT / "third_party" / "MatchAnything",
    ]
)

MATCHANYTHING_ROOT = None
for root in candidate_roots:
    if root.is_dir():
        MATCHANYTHING_ROOT = root
        break

if MATCHANYTHING_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the vendorised MatchAnything checkout. Set MATCHANYTHING_DIR "
        "or place it under imcui/third_party/MatchAnything."
    )

SRC_DIR = MATCHANYTHING_ROOT / "src"
CONF_DIR = MATCHANYTHING_ROOT / "configs" / "models"
WEIGHTS_ELOFTR = MATCHANYTHING_ROOT / "weights" / "matchanything_eloftr.ckpt"

for path in (MATCHANYTHING_ROOT, SRC_DIR):
    if path.is_dir():
        resolved = str(path)
        if resolved not in sys.path:
            sys.path.append(resolved)

# -------------------------
# 1) Load E-LoFTR + weights
# -------------------------
def load_eloftr_model(
    device: str = "cuda",
    ckpt_path: Union[str, os.PathLike] = WEIGHTS_ELOFTR,
    eval_mode: bool = True,
):
    """Load the vendorised E-LoFTR checkpoint and wrap it for ONNX export."""

    from imcui.third_party.MatchAnything.src.loftr import LoFTR
    from imcui.third_party.MatchAnything.src.utils.misc import lower_config

    spec = importlib_util.spec_from_file_location("eloftr_cfg", str(CONF_DIR / "eloftr_model.py"))
    eloftr_cfg = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(eloftr_cfg)  # defines cfg

    cfg = lower_config(eloftr_cfg.cfg)
    loftr_cfg = cfg["loftr"]

    coarse = loftr_cfg.setdefault("coarse", {})
    if coarse.get("npe") is None:
        height, width = 480, 640
        coarse["npe"] = [height, width, height, width]

    model = LoFTR(config=loftr_cfg)

    ckpt_path = pathlib.Path(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)

    if eval_mode:
        model.eval()

    class LoFTRExportWrapper(torch.nn.Module):
        """Adapter that exposes a signature matching the ONNX exporter expectations."""

        def __init__(self, core: torch.nn.Module):
            super().__init__()
            self.core = core

        @torch.no_grad()
        def forward(self, image0, image1):
            data = {"image0": image0, "image1": image1}
            out = self.core(data)
            mkpts0 = out.get("mkpts0_f", out.get("mkpts0_c"))
            mkpts1 = out.get("mkpts1_f", out.get("mkpts1_c"))
            mconf = out.get("mconf", out.get("m_b_conf", None))
            if mconf is None:
                mconf = torch.empty(0, device=image0.device)
            return mkpts0, mkpts1, mconf

    wrapper = LoFTRExportWrapper(model).to(device)
    return wrapper, loftr_cfg

# --------------------------------------
# 2) Export ONNX with dynamic dimensions
# --------------------------------------
def export_eloftr_onnx(model, onnx_path, opset=17, sample_hw=(480, 640)):
    """Export the wrapped E-LoFTR module to ONNX with dynamic shapes."""
    model.eval().to("cuda")  # 'model' is the wrapper that takes (image0, image1)
    H, W = sample_hw
    image0 = torch.randn(1, 1, H, W, device="cuda")
    image1 = torch.randn(1, 1, H, W, device="cuda")

    # Preferred modern exporter:
    try:
        torch.onnx.export(
            model,
            (image0, image1),
            onnx_path,
            input_names=["image0", "image1"],
            output_names=["mkpts0", "mkpts1", "mconf"],
            opset_version=opset,
            dynamo=True,                        # <- new ONNX exporter
            dynamic_shapes={                    # only for dynamo=True
                "image0": {0: "N", 2: "H", 3: "W"},
                "image1": {0: "N", 2: "H", 3: "W"},
            },
        )
        print(f"[OK] Saved ONNX via torch.onnx.export(dynamo=True) -> {onnx_path}")
        return onnx_path
    except Exception as e:
        print("[WARN] Dynamo path failed; falling back to classic exporter:", e)

    # Fallback (classic):
    torch.onnx.export(
        model, (image0, image1), onnx_path,
        input_names=["image0", "image1"],
        output_names=["mkpts0", "mkpts1", "mconf"],
        opset_version=opset,
        dynamic_axes={
            "image0": {0: "N", 2: "H", 3: "W"},
            "image1": {0: "N", 2: "H", 3: "W"},
            "mkpts0": {0: "M"},
            "mkpts1": {0: "M"},
            "mconf":  {0: "M"},
        },
        do_constant_folding=True,
    )
    print(f"[OK] Saved ONNX via classic exporter -> {onnx_path}")
    return onnx_path


# -------------------------------------------------
# 3) Build a TensorRT engine from an ONNX file (FP16)
# -------------------------------------------------
import tensorrt as trt


def build_trt_engine_from_onnx(
    onnx_path: Union[str, os.PathLike],
    engine_path: Union[str, os.PathLike],
    min_hw=(240, 320),
    opt_hw=(320, 480),
    max_hw=(480, 640),
    fp16=True,
    workspace_mb=512,
    builder_optimization_level=1,
    static_hw=(320, 480),      # static fallback (H, W)
):
    """Build a TensorRT 10 engine from an ONNX file with memory-friendly settings."""

    onnx_path = os.fspath(onnx_path)
    engine_path = os.fspath(engine_path)

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, namespace="")

    # Helper: add ONE optimization profile that covers ALL dynamic inputs
    def add_profiles(builder, config, input_names, minh, minw, opth, optw, maxh, maxw):
        prof = builder.create_optimization_profile()
        for name in input_names:
            prof.set_shape(
                name,
                min=(1, 1, minh, minw),
                opt=(1, 1, opth, optw),
                max=(1, 1, maxh, maxw),
            )
        config.add_optimization_profile(prof)

    # Helper: build once (dynamic or static depending on shapes you pass)
    def _do_build(shapes_tuple, dynamic=True):
        with trt.Builder(logger) as builder, \
             builder.create_builder_config() as config, \
             builder.create_network(flags=int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, logger) as parser:

            # Parse ONNX
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print("[TRT][ParserError]", parser.get_error(i))
                    raise RuntimeError("Failed to parse ONNX")

            # Introspect inputs so we can verify names
            n_inputs = network.num_inputs
            in_names = [network.get_input(i).name for i in range(n_inputs)]
            in_shapes = [tuple(network.get_input(i).shape) for i in range(n_inputs)]
            print("[TRT] Network inputs:")
            for i, (n, s) in enumerate(zip(in_names, in_shapes)):
                print(f"  - {i}: name='{n}', dtype={network.get_input(i).dtype}, shape={s}, is_shape_tensor={network.get_input(i).is_shape_tensor}")

            # Expect two inputs called image0/image1; fall back to "first two" if names differ
            expected = ("image0", "image1")
            if not set(expected).issubset(set(in_names)) and len(in_names) >= 2:
                print("[WARN] ONNX input names differ; using first two:", in_names[:2])
                input_names = in_names[:2]
            else:
                input_names = list(expected)

            # Dynamic profile or static profile
            if dynamic:
                (minh, minw), (opth, optw), (maxh, maxw) = shapes_tuple
                add_profiles(builder, config, input_names, minh, minw, opth, optw, maxh, maxw)
            else:
                (h, w) = shapes_tuple
                # static == min=opt=max
                add_profiles(builder, config, input_names, h, w, h, w, h, w)

            # Cap workspace memory (TRT 8.6+ / 10 API)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * (1 << 20))

            # Prefer FP16 where legal
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # Keep search space small on 4GB GPUs
            config.builder_optimization_level = int(builder_optimization_level)

            # (Optional) restrict tactic sources (cast to int to avoid enum/int TypeError)
            mask = 0
            for src_name in ("CUBLAS", "CUBLAS_LT", "CUDNN"):
                if hasattr(trt.TacticSource, src_name):
                    mask |= int(getattr(trt.TacticSource, src_name))
            try:
                if mask:
                    config.set_tactic_sources(mask)
            except Exception as e:
                print("[WARN] set_tactic_sources failed:", e)

            # Build serialized engine
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("Failed to build serialized network")

            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            with open(engine_path, "wb") as f:
                f.write(engine_bytes)
            print(f"[OK] Saved TensorRT engine -> {engine_path}")

    # Try dynamic first
    try:
        _do_build((min_hw, opt_hw, max_hw), dynamic=True)
        return
    except Exception as e:
        print("[WARN] Dynamic build failed:", e)
        print(f"[INFO] Retrying as *static* engine at {static_hw} to avoid OOM/tactic/profile issues…")

    # Static fallback (often succeeds on 4GB GPUs)
    _do_build(static_hw, dynamic=False)


# -----------------------------------------
# 4) (Optional) quick runtime sanity checker
# -----------------------------------------
def run_tensorrt_inference(engine_path, image0_np, image1_np):
    """
    Minimal inference to demonstrate bindings and dynamic shapes.
    image*_np: numpy arrays [N,1,H,W], float32 in [~standardized range]
    Returns: dict with output arrays: mkpts0, mkpts1, mconf
    """
    import tensorrt as trt
    import cuda  # from pycuda or cuda-python depending on your env; adapt if needed

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Set dynamic input shapes on the context
    b, c, h, w = image0_np.shape
    context.set_binding_shape(0, (b, c, h, w))  # image0
    context.set_binding_shape(1, (b, c, h, w))  # image1

    # Allocate device buffers
    import pycuda.driver as cuda_drv
    import pycuda.autoinit  # noqa: F401  (initializes CUDA)

    def dev_alloc(arr):
        d = cuda_drv.mem_alloc(arr.nbytes)
        cuda_drv.memcpy_htod(d, arr)
        return d

    d_image0 = dev_alloc(image0_np)
    d_image1 = dev_alloc(image1_np)

    # Query output shapes now that inputs are set
    # Bindings order: [image0, image1, mkpts0, mkpts1, mconf] (assuming export order)
    out_shapes = [tuple(context.get_binding_shape(i)) for i in range(2, engine.num_bindings)]
    out_sizes  = [int(np.prod(s)) for s in out_shapes]
    d_outputs  = [cuda_drv.mem_alloc(sz * 4) for sz in out_sizes]  # float32

    bindings = [int(d_image0), int(d_image1)] + [int(p) for p in d_outputs]
    context.execute_v2(bindings)

    # Copy back
    host_outs = [np.empty(sz, dtype=np.float32) for sz in out_sizes]
    for host, dev in zip(host_outs, d_outputs):
        cuda_drv.memcpy_dtoh(host, dev)

    # Reshape
    mkpts0 = host_outs[0].reshape(out_shapes[0])
    mkpts1 = host_outs[1].reshape(out_shapes[1])
    mconf  = host_outs[2].reshape(out_shapes[2])
    return {"mkpts0": mkpts0, "mkpts1": mkpts1, "mconf": mconf}
