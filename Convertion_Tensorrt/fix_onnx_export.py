#!/usr/bin/env python3
"""
Fixed ONNX export that properly saves weights to external data file.
"""

import torch
import onnx
import os
from pathlib import Path

try:
    from onnx import external_data_utils
except ImportError:
    from onnx import external_data_helper as external_data_utils


def export_with_proper_weights(model, dummy_inputs, onnx_path):
    """Export ONNX with proper weight handling"""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    print(f"[FIXED] Exporting to {onnx_path}")

    # First, export normally (this will create multiple weight files)
    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        input_names=["image0", "image1"],
        output_names=["keypoints0", "keypoints1", "mconf"],
        dynamic_axes={
            "image0": {0: "B", 2: "H", 3: "W"},
            "image1": {0: "B", 2: "H", 3: "W"},
            "keypoints0": {0: "num_matches"},
            "keypoints1": {0: "num_matches"},
            "mconf": {0: "num_matches"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    print("[FIXED] Initial export complete")

    # Load and re-save with proper external data handling
    print("[FIXED] Consolidating external data...")

    try:
        # Load the model with all external data
        model_proto = onnx.load(onnx_path, load_external_data=True)

        # Convert to single external data file
        data_filename = os.path.basename(onnx_path) + ".data"
        external_data_utils.convert_model_to_external_data(
            model_proto,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=1024,  # 1KB threshold
        )

        # Save the consolidated model
        onnx.save_model(
            model_proto,
            onnx_path,
            save_as_external_data=True,
        )

        print("[FIXED] Model consolidated successfully")

    except Exception as e:
        print(f"[FIXED] Error during consolidation: {e}")
        print("[FIXED] Trying alternative approach...")

        # Alternative: Load without external data, then convert
        try:
            model_proto = onnx.load(onnx_path, load_external_data=False)

            # Find all external data files
            out_dir = Path(onnx_path).parent
            external_files = list(out_dir.glob("onnx__*"))
            print(f"[FIXED] Found {len(external_files)} external data files")

            if external_files:
                # Load with external data
                model_proto = onnx.load(onnx_path, load_external_data=True)

                # Convert to single file
                data_filename = os.path.basename(onnx_path) + ".data"
                external_data_utils.convert_model_to_external_data(
                    model_proto,
                    all_tensors_to_one_file=True,
                    location=data_filename,
                    size_threshold=0,  # Include all tensors
                )

                # Save
                onnx.save_model(
                    model_proto,
                    onnx_path,
                    save_as_external_data=True,
                )
                print("[FIXED] Alternative consolidation successful")
            else:
                print("[FIXED] No external data files found - weights may be embedded")

        except Exception as e2:
            print(f"[FIXED] Alternative approach also failed: {e2}")

    # Clean up temporary files
    out_dir = Path(onnx_path).parent
    temp_files = list(out_dir.glob("onnx__*"))
    if temp_files:
        print(f"[FIXED] Cleaning up {len(temp_files)} temporary files...")
        for temp_file in temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass

    # Check final file sizes
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    data_file = onnx_path + ".data"

    print("[FIXED] Export complete!")
    print(f"  - ONNX file: {onnx_size:.1f} MB")

    if os.path.exists(data_file):
        data_size = os.path.getsize(data_file) / (1024 * 1024)
        print(f"  - Data file: {data_size:.1f} MB")

        if data_size > 100:
            print("  ✅ Success! Data file size indicates weights are properly saved")
            return True
        else:
            print("  ❌ Warning: Data file still too small")
            return False
    else:
        print("  ❌ No data file created - weights may be embedded in ONNX")
        if onnx_size > 100:
            print("  ✅ But ONNX file is large, so weights might be embedded")
            return True
        else:
            print("  ❌ ONNX file is also small - weights are missing")
            return False


if __name__ == "__main__":
    print("Fixed ONNX export utility")
