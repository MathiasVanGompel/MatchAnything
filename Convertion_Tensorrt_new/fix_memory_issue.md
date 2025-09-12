# Fix for MatchAnything ONNX Export Memory Issue

## Problem Analysis

The "Killed" error during ONNX export is caused by memory exhaustion. The process successfully:
1. Loads 91.5% of weights (270/295 parameters)
2. Passes forward test
3. But gets killed during ONNX export due to memory usage

## Root Causes

1. **Large Model Size**: MatchAnything with DINOv2 backbone is memory-intensive
2. **High Resolution**: 840x840 input creates large intermediate tensors
3. **ONNX Export Overhead**: The export process requires additional memory for graph tracing
4. **No Swap Space**: System has 0B swap, so no memory overflow protection

## Solutions (In Order of Preference)

### Solution 1: Use Smaller Input Resolution
```bash
python3 export_fixed_weights.py \
    --onnx output/matchanything_fixed_weights.onnx \
    --checkpoint /home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt \
    --H 560 \
    --W 560
```

### Solution 2: Use the Optimized Export Script
I created `export_fixed_weights_optimized.py` with memory optimizations:
- Explicit garbage collection
- Gradient disabling
- Memory-mapped checkpoint loading
- External data format for ONNX

```bash
python3 export_fixed_weights_optimized.py \
    --onnx output/matchanything_optimized.onnx \
    --checkpoint /home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt \
    --H 560 \
    --W 560
```

### Solution 3: Use Minimal Memory Export
I created `export_minimal_memory.py` with aggressive optimizations:
- Smaller default input size (560x560)
- Minimal weight loading
- Disabled constant folding
- Lower ONNX opset version

```bash
python3 export_minimal_memory.py \
    --onnx output/matchanything_minimal.onnx \
    --checkpoint /home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt \
    --H 420 \
    --W 420
```

### Solution 4: Add Swap Space (System-level fix)
```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Then retry the original command
python3 export_fixed_weights.py \
    --onnx output/matchanything_fixed_weights.onnx \
    --checkpoint /home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt \
    --H 840 \
    --W 840
```

### Solution 5: Environment Variable Tuning
```bash
# Limit PyTorch memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=1

# Then run export
python3 export_fixed_weights.py \
    --onnx output/matchanything_fixed_weights.onnx \
    --checkpoint /home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt \
    --H 560 \
    --W 560
```

## Recommended Approach

1. **Start with Solution 1**: Try smaller input resolution (560x560) with the original script
2. **If still failing**: Use Solution 2 (optimized script) with 420x420 resolution
3. **If still failing**: Use Solution 3 (minimal script) with 280x280 resolution
4. **Last resort**: Add swap space (Solution 4)

## Key Optimizations Implemented

### In `export_fixed_weights_optimized.py`:
- `torch.no_grad()` context for export
- Explicit `gc.collect()` calls
- `weights_only=False` for checkpoint loading
- External data format for large models
- Memory cleanup after each major operation

### In `export_minimal_memory.py`:
- Smaller default input size
- `torch.set_grad_enabled(False)`
- `requires_grad=False` for all parameters
- Lower ONNX opset version (11 vs 17)
- Disabled constant folding
- Immediate memory cleanup

## Testing the Fix

The optimized scripts should prevent the "Killed" error by:
1. Using less memory during export
2. Cleaning up intermediate tensors
3. Using smaller input dimensions
4. Optimizing the export process

You can verify the fix worked if:
- The export completes without being killed
- You see "✅ ONNX export successful"
- The output file is created with reasonable size