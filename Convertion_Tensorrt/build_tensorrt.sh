#!/bin/bash

# MatchAnything TensorRT Conversion Script
# This script automates the conversion from PyTorch to TensorRT

set -e

# Configuration
CKPT_PATH=""
H=448
W=448
WORKSPACE_MB=2048
USE_FP16=true

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --height)
            H="$2"
            shift 2
            ;;
        --width)
            W="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE_MB="$2"
            shift 2
            ;;
        --no-fp16)
            USE_FP16=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ckpt PATH        Path to matchanything_roma.ckpt"
            echo "  --height HEIGHT    Input height (default: 448)"
            echo "  --width WIDTH      Input width (default: 448)"
            echo "  --workspace MB     TensorRT workspace size in MB (default: 2048)"
            echo "  --no-fp16          Disable FP16 precision"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== MatchAnything TensorRT Conversion ==="
echo "Input size: ${H}x${W}"
echo "Workspace: ${WORKSPACE_MB}MB"
echo "FP16: ${USE_FP16}"
if [[ -n "$CKPT_PATH" ]]; then
    echo "Checkpoint: $CKPT_PATH"
fi
echo

# Create output directory
mkdir -p "${OUT_DIR}"

# Step 1: Export to ONNX
echo "Step 1: Exporting PyTorch model to ONNX..."
ONNX_PATH="${OUT_DIR}/roma_dino_gp_dynamic.onnx"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 "${SCRIPT_DIR}/matchanything_to_trt_full.py" \
    --onnx "${ONNX_PATH}" \
    --H ${H} \
    --W ${W} \
    $(if [[ -n "$CKPT_PATH" ]]; then echo "--ckpt $CKPT_PATH"; fi) \
    --verbose

if [[ ! -f "$ONNX_PATH" ]]; then
    echo "Error: ONNX export failed"
    exit 1
fi

echo "ONNX export completed: $ONNX_PATH"
echo

# Step 2: Convert ONNX to TensorRT
echo "Step 2: Converting ONNX to TensorRT engine..."

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "Error: trtexec not found. Please install TensorRT and add trtexec to PATH"
    exit 1
fi

# Build TensorRT engine
ENGINE_PATH="${OUT_DIR}/roma_dino_gp.plan"

TRTEXEC_ARGS=(
    --onnx="${ONNX_PATH}"
    --saveEngine="${ENGINE_PATH}"
    --explicitBatch
    --workspace=${WORKSPACE_MB}
    --minShapes=image0:1x3x224x224,image1:1x3x224x224
    --optShapes=image0:1x3x${H}x${W},image1:1x3x${H}x${W}
    --maxShapes=image0:1x3x896x896,image1:1x3x896x896
    --buildOnly
    --verbose
)

if [[ "$USE_FP16" == "true" ]]; then
    TRTEXEC_ARGS+=(--fp16)
fi

echo "Running trtexec with the following arguments:"
echo "trtexec ${TRTEXEC_ARGS[*]}"
echo

trtexec "${TRTEXEC_ARGS[@]}"

if [[ ! -f "$ENGINE_PATH" ]]; then
    echo "Error: TensorRT engine build failed"
    exit 1
fi

echo "TensorRT engine created: $ENGINE_PATH"
echo

# Step 3: Display usage instructions
echo "=== Conversion Complete ==="
echo "Engine file: $ENGINE_PATH"
echo
echo "To run inference with the TensorRT engine:"
echo "python3 ${SCRIPT_DIR}/run_ma_trt.py \\"
echo "    --engine \"$ENGINE_PATH\" \\"
echo "    --image0 /path/to/image1.jpg \\"
echo "    --image1 /path/to/image2.jpg \\"
echo "    --H ${H} --W ${W}"
echo
echo "Optional parameters:"
echo "  --norm imagenet     # Use ImageNet normalization"
echo "  --budget 1000       # Number of top matches to extract"
echo "  --outdir /path/out  # Output directory for results"
echo