# MatchAnything TensorRT Conversion

This directory contains tools to convert the MatchAnything model from PyTorch to TensorRT for optimized inference on NVIDIA GPUs.

## 🎯 **Accurate Implementation**

This repository provides an accurate TensorRT port that maintains exact compatibility with the [original HuggingFace space](https://huggingface.co/spaces/LittleFrog/MatchAnything). The conversion replicates the full preprocessing and postprocessing pipeline to ensure identical results.

## Files Overview

### 🎯 **Accurate Implementation**
- `accurate_matchanything_trt.py` - **Exact replica** of original MatchAnything
- `convert_accurate_matchanything.py` - Conversion script for accurate version
- `run_accurate_matchanything_trt.py` - TensorRT inference with exact accuracy
- `build_accurate_tensorrt.sh` - Automated build for accurate version
- `validate_accuracy.py` - Script to validate TensorRT vs PyTorch accuracy

### 🔧 **Shared Components**
- `weight_adapter.py` - Checkpoint weight remapping utilities
- `encoders_trt_full.py` - TensorRT-optimized encoder implementation
- `gp_trt.py` - Gaussian Process matcher for TensorRT
- `setup_environment.py` - Environment verification
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

The ONNX export scripts generate a model file (`.onnx`) alongside a
companion weights file (`.onnx.data`). Keep **both** files together when
building or deploying with TensorRT; missing the `.data` file will cause
weight-loading errors during `trtexec`.

### Step 0: Environment Check
First, verify your environment is ready:
```bash
python3 setup_environment.py
```
This will check for CUDA, TensorRT, Python dependencies, and ROMA installation.

**Note**: It's normal for checks to "fail" in development environments without GPU support. The key requirement is that ROMA is found and accessible.

### Build Accurate TensorRT Engine
```bash
# Build accurate TensorRT engine that matches original exactly
./build_accurate_tensorrt.sh

# Python helper (same behaviour as the shell script)
python3 build_accurate_tensorrt.py

# Custom settings
./build_accurate_tensorrt.sh \
    --model matchanything_roma \
    --height 832 --width 832 \
    --match_threshold 0.1 \
    --workspace 4096
```

## Running Inference

```bash
python3 run_accurate_matchanything_trt.py \
    --engine out/accurate_matchanything_roma.plan \
    --image0 /path/to/image1.jpg \
    --image1 /path/to/image2.jpg \
    --confidence_threshold 0.1
```

## 🔍 **Accuracy Validation**

Validate that your TensorRT implementation produces identical results:

```bash
python3 validate_accuracy.py \
    --model matchanything_roma \
    --image0 test_img1.jpg \
    --image1 test_img2.jpg \
    --ckpt /path/to/checkpoint.ckpt
```

This script compares the TensorRT implementation against the original PyTorch model and reports accuracy metrics.

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

3. **AccurateMatchAnythingTRT** (`accurate_matchanything_trt.py`)
   - Combines encoder and matcher with original preprocessing/postprocessing
   - Outputs keypoints and confidence scores matching PyTorch

## Input/Output Format

### Inputs
- `image0`: RGB image tensor `[B, 3, H, W]` in range [0, 1]
- `image1`: RGB image tensor `[B, 3, H, W]` in range [0, 1]

### Outputs
- `keypoints0`: Matched keypoints from image0 `[N, 2]`
- `keypoints1`: Matched keypoints from image1 `[N, 2]`
- `mconf`: Confidence scores `[N]`

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
