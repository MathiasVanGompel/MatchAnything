#!/usr/bin/env python3
import onnx
import os

try:
    from onnx import external_data_utils
except ImportError:
    from onnx import external_data_helper as external_data_utils


def fix_onnx_external_data(input_onnx, output_onnx):
    """Convert embedded ONNX to external data format for TensorRT"""

    print(f"[FIX] Loading {input_onnx}")
    print(f"[FIX] Input size: {os.path.getsize(input_onnx) / (1024*1024):.1f} MB")

    # Load the ONNX model
    model = onnx.load(input_onnx, load_external_data=False)

    # Count embedded parameters
    embedded_params = 0
    for init in model.graph.initializer:
        if init.raw_data or init.float_data or init.int64_data:
            param_count = 1
            for dim in init.dims:
                param_count *= dim
            embedded_params += param_count

    print(f"[FIX] Found {embedded_params:,} embedded parameters")

    # Convert to external data format
    print("[FIX] Converting to external data format...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_onnx) or ".", exist_ok=True)

    # Convert all weights to external data
    external_data_utils.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_onnx) + ".data",
        size_threshold=1024,  # Externalize anything > 1KB
    )

    # Save the model
    onnx.save_model(model, output_onnx, save_as_external_data=True)

    # Check results
    onnx_size = os.path.getsize(output_onnx) / (1024 * 1024)
    data_file = output_onnx + ".data"

    print(f"[FIX] Output ONNX size: {onnx_size:.1f} MB")

    if os.path.exists(data_file):
        data_size = os.path.getsize(data_file) / (1024 * 1024)
        print(f"[FIX] Output data size: {data_size:.1f} MB")
        print(f"[FIX] Total size: {onnx_size + data_size:.1f} MB")

        if data_size > 100:
            print("✅ SUCCESS! External data file created properly")
            return True
        else:
            print("❌ External data file too small")
            return False
    else:
        print("❌ No external data file created")
        return False


if __name__ == "__main__":
    success = fix_onnx_external_data(
        "out/matchanything_complete.onnx", "out/matchanything_external.onnx"
    )
    print(f"Conversion {'succeeded' if success else 'failed'}")
