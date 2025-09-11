#!/usr/bin/env python3
"""
Simple working ONNX export that avoids consolidation issues.
"""

import torch
import os
from unified_weight_loader_complete import apply_unified_weight_loading
from accurate_matchanything_trt import AccurateMatchAnythingTRT


def export_working_onnx(onnx_path, checkpoint_path, H=832, W=832):
    """Simple working ONNX export"""
    print(f"[WORKING] Exporting to {onnx_path}")
    model = AccurateMatchAnythingTRT(model_name="matchanything_roma")
    if checkpoint_path:
        loadable = apply_unified_weight_loading(
            checkpoint_path, model.state_dict(), load_dinov2_components=True
        )
        model.load_state_dict(loadable, strict=False)
        print(f"[WORKING] Loaded {len(loadable)} weights")
    model.eval()
    x1 = torch.rand(1, 3, H, W)
    x2 = torch.rand(1, 3, H, W)
    with torch.no_grad():
        result = model(x1, x2)
        print(f"[WORKING] Forward pass OK: {result['keypoints0'].shape}")
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        (x1, x2),
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
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[WORKING] Export complete: {onnx_size:.1f} MB")
    if onnx_size > 400:
        print("✅ SUCCESS! Large ONNX file created!")
        return True
    else:
        print("❌ ONNX file smaller than expected")
        return False


if __name__ == "__main__":
    success = export_working_onnx(
        "out/matchanything_working.onnx",
        "/home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt",
    )
    print(f"Export {'succeeded' if success else 'failed'}")
