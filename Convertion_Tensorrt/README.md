# MatchAnything TensorRT Conversion

This directory contains tools to convert the MatchAnything model from PyTorch to TensorRT for optimized inference on NVIDIA GPUs.

## Files Overview

- `matchanything_to_trt_full.py` - Main conversion script (PyTorch → ONNX)
- `run_ma_trt.py` - TensorRT inference script
- `weight_adapter.py` - Checkpoint weight remapping utilities
- `encoders_trt_full.py` - TensorRT-optimized encoder implementation
- `gp_trt.py` - Gaussian Process matcher for TensorRT
- `matcher_trt_full.py` - Additional matching components (currently unused)
- `roma_models_trt_full.py` - Main TensorRT model definition
- `build_tensorrt.sh` - Automated build script
- `out/` - Output directory for ONNX and TensorRT files

## Prerequisites

### Software Requirements
- NVIDIA GPU with Compute Capability 6.0+
- CUDA 11.0+ or 12.0+
- TensorRT 8.6.0+
- Python 3.8+

### Python Dependencies
Install the required packages:
```bash
pip install -r ../requirements.txt
```

The key TensorRT-specific dependencies are:
- `onnx>=1.12.0`
- `onnx-graphsurgeon`
- `tensorrt>=8.6.0`
- `pycuda>=2022.1`

## Quick Start

### Step 0: Environment Check
First, verify your environment is ready:
```bash
python3 setup_environment.py
```
This will check for CUDA, TensorRT, Python dependencies, and ROMA installation.

**Note**: It's normal for checks to "fail" in development environments without GPU support. The key requirement is that ROMA is found and accessible.

### Option 1: Automated Build (Recommended)
```bash
# Basic conversion (no checkpoint)
./build_tensorrt.sh

# With checkpoint
./build_tensorrt.sh --ckpt /path/to/matchanything_roma.ckpt

# Custom input size
./build_tensorrt.sh --height 672 --width 672 --workspace 4096
```

### Option 2: Example Workflow
For a complete walkthrough:
```bash
python3 example_usage.py --checkpoint /path/to/checkpoint.ckpt --image1 img1.jpg --image2 img2.jpg
```

### Option 3: Manual Steps

#### Step 1: Export to ONNX
```bash
python3 matchanything_to_trt_full.py \
    --onnx out/roma_dino_gp_dynamic.onnx \
    --H 448 --W 448 \
    --ckpt /path/to/matchanything_roma.ckpt
```

#### Step 2: Build TensorRT Engine
```bash
trtexec \
    --onnx=out/roma_dino_gp_dynamic.onnx \
    --saveEngine=out/roma_dino_gp.plan \
    --explicitBatch --fp16 --workspace=2048 \
    --minShapes=image0:1x3x224x224,image1:1x3x224x224 \
    --optShapes=image0:1x3x448x448,image1:1x3x448x448 \
    --maxShapes=image0:1x3x896x896,image1:1x3x896x896 \
    --buildOnly
```

## Running Inference

Once you have built the TensorRT engine, run inference:

```bash
python3 run_ma_trt.py \
    --engine out/roma_dino_gp.plan \
    --image0 /path/to/image1.jpg \
    --image1 /path/to/image2.jpg \
    --H 448 --W 448
```

### Inference Options
- `--norm imagenet` - Use ImageNet normalization (recommended for most cases)
- `--budget 1000` - Number of top matches to extract
- `--outdir /path/out` - Output directory for visualization

## Model Architecture

The TensorRT-optimized model consists of:

1. **CNNandDinov2TRT** (`encoders_trt_full.py`)
   - DINOv2 ViT-L/14 backbone with ONNX-safe positional encoding
   - Outputs coarse features at 1/14 resolution

2. **GPMatchEncoderTRT** (`gp_trt.py`)
   - Lightweight Gaussian Process-based matcher
   - Computes correspondences and certainty maps

3. **RoMaTRTCoreFull** (`roma_models_trt_full.py`)
   - Main model combining encoder and matcher
   - Outputs dense warp field and certainty map

## Input/Output Format

### Inputs
- `image0`: RGB image tensor `[B, 3, H, W]` in range [0, 1]
- `image1`: RGB image tensor `[B, 3, H, W]` in range [0, 1]

### Outputs
- `warp`: Dense correspondence field `[B, 4, H, W]` containing `(x0, y0, x1, y1)` in normalized coordinates [-1, 1]
- `cert`: Certainty map `[B, 1, H, W]` in range [0, 1]

## Performance Tips

1. **Input Size**: Use multiples of 14 (patch size) for optimal performance
2. **Batch Size**: Start with batch size 1, increase if memory permits
3. **Precision**: FP16 provides ~2x speedup with minimal accuracy loss
4. **Memory**: Increase workspace size for better optimization (2048-4096 MB)

## Troubleshooting

### Common Issues

**ONNX Export Fails**
- Check ROMA path in `encoders_trt_full.py`
- Ensure all dependencies are installed
- Verify input dimensions are multiples of 14

**TensorRT Build Fails**
- Check CUDA/TensorRT installation
- Increase workspace size
- Try without FP16 first

**Runtime Errors**
- Ensure input images exist and are readable
- Check input dimensions match engine expectations
- Verify CUDA device availability

### Memory Requirements

| Input Size | VRAM (FP32) | VRAM (FP16) |
|------------|-------------|-------------|
| 448×448    | ~4GB        | ~2GB        |
| 672×672    | ~8GB        | ~4GB        |
| 896×896    | ~14GB       | ~7GB        |

## Advanced Usage

### Custom Checkpoints
The weight adapter supports various checkpoint formats:
- Lightning checkpoints
- Raw state dictionaries
- Model-wrapped checkpoints

### ONNX Graph Surgery
The conversion automatically removes problematic `EyeLike` operators for better TensorRT compatibility.

### Dynamic Shapes
The engine supports dynamic input shapes within the specified ranges:
- Minimum: 224×224
- Optimal: 448×448 (or specified)
- Maximum: 896×896

## Citation

If you use this TensorRT conversion in your research, please cite the original MatchAnything paper:

```bibtex
@article{matchanything2024,
  title={MatchAnything: Universal Cross-modal Image Matching with Large-scale Pre-training},
  author={...},
  journal={...},
  year={2024}
}
```