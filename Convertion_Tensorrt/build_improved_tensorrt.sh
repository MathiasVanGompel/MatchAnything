#!/bin/bash

# Improved MatchAnything TensorRT Conversion Script
# This script fixes weight loading and image size handling issues

set -e

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
WEIGHTS_DIR="${SCRIPT_DIR}/../imcui/third_party/MatchAnything/weights"

# Configuration
CKPT_PATH="${WEIGHTS_DIR}/matchanything_roma.ckpt"
MODEL="matchanything_roma"
H=832
W=832
MATCH_THRESHOLD=0.1
WORKSPACE_MB=4096
USE_FP16=true
DOWNLOAD_WEIGHTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
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
        --match_threshold)
            MATCH_THRESHOLD="$2"
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
        --download-weights)
            DOWNLOAD_WEIGHTS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ckpt PATH            Path to MatchAnything checkpoint"
            echo "  --model NAME           Model variant (matchanything_roma, matchanything_eloftr)"
            echo "  --height HEIGHT        Input height (default: 832)"
            echo "  --width WIDTH          Input width (default: 832)"
            echo "  --match_threshold THR  Match confidence threshold (default: 0.1)"
            echo "  --workspace MB         TensorRT workspace size in MB (default: 4096)"
            echo "  --no-fp16              Disable FP16 precision"
            echo "  --download-weights     Download weights from Google Drive if missing"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Improved MatchAnything TensorRT Conversion ==="
echo "Model: $MODEL"
echo "Input size: ${H}x${W}"
echo "Match threshold: $MATCH_THRESHOLD"
echo "Workspace: ${WORKSPACE_MB}MB"
echo "FP16: ${USE_FP16}"
echo "Checkpoint: $CKPT_PATH"
echo "Download weights: $DOWNLOAD_WEIGHTS"
echo

# Create output directory
mkdir -p "${OUT_DIR}"
mkdir -p "${WEIGHTS_DIR}"

# Step 0: Download weights if requested and not present
if [[ "$DOWNLOAD_WEIGHTS" == "true" ]] || [[ ! -f "$CKPT_PATH" ]]; then
    echo "Step 0: Downloading MatchAnything pretrained weights..."
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
    
    python3 "${SCRIPT_DIR}/download_weights.py" \
        --output_dir "${WEIGHTS_DIR}" \
        --verify
    
    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "Error: Failed to download weights to $CKPT_PATH"
        exit 1
    fi
    echo "Weights downloaded successfully"
    echo
fi

# Step 1: Export to ONNX with improved weight loading
echo "Step 1: Exporting improved PyTorch model to ONNX..."
ONNX_PATH="${OUT_DIR}/improved_${MODEL}.onnx"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

CONVERT_ARGS=(
    "python3" "${SCRIPT_DIR}/convert_accurate_matchanything.py"
    "--onnx" "${ONNX_PATH}"
    "--model" "${MODEL}"
    "--H" "${H}"
    "--W" "${W}"
    "--match_threshold" "${MATCH_THRESHOLD}"
    "--verbose"
)

if [[ -f "$CKPT_PATH" ]]; then
    CONVERT_ARGS+=("--ckpt" "$CKPT_PATH")
    echo "Using checkpoint: $CKPT_PATH"
else
    echo "Warning: No checkpoint found at $CKPT_PATH"
    echo "Proceeding with random initialization..."
fi

"${CONVERT_ARGS[@]}"

if [[ ! -f "$ONNX_PATH" ]]; then
    echo "Error: ONNX export failed"
    exit 1
fi

echo "ONNX export completed: $ONNX_PATH"
echo

# Step 2: Convert ONNX to TensorRT with dynamic shapes
echo "Step 2: Converting ONNX to TensorRT engine with dynamic shapes..."

# Check if trtexec is available
TRTEXEC_PATH="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC_PATH" ]; then
    if ! command -v trtexec &> /dev/null; then
        echo "Error: trtexec not found at $TRTEXEC_PATH or in PATH"
        echo "Please check your TensorRT installation"
        exit 1
    else
        TRTEXEC_PATH="trtexec"
    fi
fi

echo "Using trtexec from: $TRTEXEC_PATH"

# Build TensorRT engine
ENGINE_PATH="${OUT_DIR}/improved_${MODEL}.plan"

# Calculate shape ranges with better dynamic support
MIN_H=224  # Minimum supported size
MIN_W=224
OPT_H=$H   # Optimal size (user specified)
OPT_W=$W
MAX_H=$((H * 2))  # Maximum size
MAX_W=$((W * 2))

# Ensure dimensions are multiples of 14 (DINOv2 patch size)
MIN_H=$(((MIN_H + 13) / 14 * 14))
MIN_W=$(((MIN_W + 13) / 14 * 14))
OPT_H=$(((OPT_H + 13) / 14 * 14))
OPT_W=$(((OPT_W + 13) / 14 * 14))
MAX_H=$(((MAX_H + 13) / 14 * 14))
MAX_W=$(((MAX_W + 13) / 14 * 14))

echo "Shape ranges:"
echo "  Min: ${MIN_H}x${MIN_W}"
echo "  Opt: ${OPT_H}x${OPT_W}"
echo "  Max: ${MAX_H}x${MAX_W}"

TRTEXEC_ARGS=(
    "--onnx=${ONNX_PATH}"
    "--saveEngine=${ENGINE_PATH}"
    "--memPoolSize=workspace:${WORKSPACE_MB}M"
    "--minShapes=image0:1x3x${MIN_H}x${MIN_W},image1:1x3x${MIN_H}x${MIN_W}"
    "--optShapes=image0:1x3x${OPT_H}x${OPT_W},image1:1x3x${OPT_H}x${OPT_W}"
    "--maxShapes=image0:1x3x${MAX_H}x${MAX_W},image1:1x3x${MAX_H}x${MAX_W}"
    "--skipInference"
    "--verbose"
    "--profilingVerbosity=detailed"
    "--builderOptimizationLevel=5"
    "--avgTiming=8"
)

if [[ "$USE_FP16" == "true" ]]; then
    TRTEXEC_ARGS+=(--fp16)
fi

echo "Running trtexec with the following arguments:"
echo "$TRTEXEC_PATH ${TRTEXEC_ARGS[*]}"
echo

"$TRTEXEC_PATH" "${TRTEXEC_ARGS[@]}"

if [[ ! -f "$ENGINE_PATH" ]]; then
    echo "Error: TensorRT engine build failed"
    exit 1
fi

echo "TensorRT engine created: $ENGINE_PATH"
echo

# Step 3: Display usage instructions
echo "=== Conversion Complete ==="
echo "Engine file: $ENGINE_PATH"
echo "ONNX file: $ONNX_PATH (keep both .onnx and .onnx.data files together)"
echo
echo "To run improved inference with the TensorRT engine:"
echo "python3 ${SCRIPT_DIR}/run_accurate_matchanything_trt.py \\"
echo "    --engine \"$ENGINE_PATH\" \\"
echo "    --image0 /path/to/image1.jpg \\"
echo "    --image1 /path/to/image2.jpg \\"
echo "    --confidence_threshold ${MATCH_THRESHOLD}"
echo
echo "The improved version includes:"
echo "- ✅ Automatic weight downloading from Google Drive"
echo "- ✅ Better weight loading and compatibility checking" 
echo "- ✅ Improved image size handling (no more crashes on different sizes)"
echo "- ✅ Dynamic shape support with proper padding"
echo "- ✅ Enhanced preprocessing with aspect ratio preservation"
echo
echo "Performance improvements:"
echo "- Dynamic input shapes (${MIN_H}x${MIN_W} to ${MAX_H}x${MAX_W})"
echo "- Optimized TensorRT build settings"
echo "- Better memory management"