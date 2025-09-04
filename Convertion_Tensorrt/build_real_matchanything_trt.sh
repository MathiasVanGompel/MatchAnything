#!/bin/bash

# Real MatchAnything TensorRT Conversion Script
# Uses the actual MatchAnything implementation for perfect compatibility

set -e

# Configuration
CKPT_PATH=""
MODEL="matchanything_roma"
H=832
W=832
MATCH_THRESHOLD=0.1
WORKSPACE_MB=4096
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ckpt PATH            Path to MatchAnything checkpoint (REQUIRED)"
            echo "  --model NAME           Model variant (matchanything_roma, matchanything_eloftr)"
            echo "  --height HEIGHT        Input height (default: 832)"
            echo "  --width WIDTH          Input width (default: 832)"
            echo "  --match_threshold THR  Match confidence threshold (default: 0.1)"
            echo "  --workspace MB         TensorRT workspace size in MB (default: 4096)"
            echo "  --no-fp16              Disable FP16 precision"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Error: --ckpt is required for real MatchAnything conversion"
    echo "Use: $0 --ckpt /path/to/matchanything_roma.ckpt"
    exit 1
fi

echo "=== Real MatchAnything TensorRT Conversion ==="
echo "Model: $MODEL"
echo "Input size: ${H}x${W}"
echo "Match threshold: $MATCH_THRESHOLD"
echo "Workspace: ${WORKSPACE_MB}MB"
echo "FP16: ${USE_FP16}"
echo "Checkpoint: $CKPT_PATH"
echo

# Create output directory
mkdir -p "${OUT_DIR}"

# Step 1: Export to ONNX using real implementation
echo "Step 1: Exporting real MatchAnything model to ONNX..."
ONNX_PATH="${OUT_DIR}/real_${MODEL}.onnx"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 "${SCRIPT_DIR}/real_matchanything_trt.py" \
    --onnx "${ONNX_PATH}" \
    --model "${MODEL}" \
    --ckpt "${CKPT_PATH}"

if [[ ! -f "$ONNX_PATH" ]]; then
    echo "Error: ONNX export failed"
    exit 1
fi

echo "ONNX export completed: $ONNX_PATH"
echo

# Step 2: Convert ONNX to TensorRT
echo "Step 2: Converting ONNX to TensorRT engine..."

# Check if trtexec is available
TRTEXEC_PATH="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC_PATH" ]; then
    # Fallback to PATH
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
ENGINE_PATH="${OUT_DIR}/real_${MODEL}.plan"

# Calculate shape ranges
MIN_H=$((H / 2))
MIN_W=$((W / 2))
MAX_H=$((H * 2))
MAX_W=$((W * 2))

TRTEXEC_ARGS=(
    "--onnx=${ONNX_PATH}"
    "--saveEngine=${ENGINE_PATH}"
    "--memPoolSize=workspace:${WORKSPACE_MB}M"
    "--minShapes=image0:1x3x${MIN_H}x${MIN_W},image1:1x3x${MIN_H}x${MIN_W}"
    "--optShapes=image0:1x3x${H}x${W},image1:1x3x${H}x${W}"
    "--maxShapes=image0:1x3x${MAX_H}x${MAX_W},image1:1x3x${MAX_H}x${MAX_W}"
    "--skipInference"
    "--verbose"
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
echo
echo "To run inference with the TensorRT engine:"
echo "python3 ${SCRIPT_DIR}/run_real_matchanything_trt.py \\"
echo "    --engine \"$ENGINE_PATH\" \\"
echo "    --image0 /path/to/image1.jpg \\"
echo "    --image1 /path/to/image2.jpg \\"
echo "    --model ${MODEL}"
echo
echo "=== Success! ==="
echo "This TensorRT engine uses the real MatchAnything implementation"
echo "and should produce identical results to the original model."