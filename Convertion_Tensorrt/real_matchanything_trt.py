#!/usr/bin/env python3
"""
Real MatchAnything TensorRT conversion using the actual implementation.
This ensures perfect compatibility by using the original model.
"""
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Optional

# Add MatchAnything to path
MATCHANYTHING_PATH = Path(__file__).parent / "../imcui/third_party/MatchAnything"
sys.path.append(str(MATCHANYTHING_PATH))

# Import the real MatchAnything components
try:
    from src.lightning.lightning_loftr import PL_LoFTR
    from src.config.default import get_cfg_defaults
    MATCHANYTHING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MatchAnything components: {e}")
    MATCHANYTHING_AVAILABLE = False

class RealMatchAnythingTRT(nn.Module):
    """
    TensorRT wrapper around the real MatchAnything model.
    This uses the actual implementation for perfect compatibility.
    """
    
    def __init__(self, model_name: str = "matchanything_roma", 
                 img_resize: int = 832, match_threshold: float = 0.1):
        super().__init__()
        
        if not MATCHANYTHING_AVAILABLE:
            raise ImportError("MatchAnything components not available")
        
        self.model_name = model_name
        self.img_resize = img_resize
        self.match_threshold = match_threshold
        
        # Load the real MatchAnything configuration
        config = get_cfg_defaults()
        
        if model_name == 'matchanything_roma':
            config_path = MATCHANYTHING_PATH / 'configs/models/roma_model.py'
            config.merge_from_file(str(config_path))
            # Disable FP16 for ONNX export stability
            config.LOFTR.FP16 = False
            config.ROMA.MODEL.AMP = False
        elif model_name == 'matchanything_eloftr':
            config_path = MATCHANYTHING_PATH / 'configs/models/eloftr_model.py'
            config.merge_from_file(str(config_path))
            # Configure for specific image size
            if config.LOFTR.COARSE.ROPE:
                assert config.DATASET.NPE_NAME is not None
            if config.DATASET.NPE_NAME is not None:
                if config.DATASET.NPE_NAME == 'megadepth':
                    config.LOFTR.COARSE.NPE = [832, 832, img_resize, img_resize]
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        
        config.METHOD = model_name
        config.LOFTR.MATCH_COARSE.THR = match_threshold
        
        # Create the model without loading checkpoint (we'll load it separately)
        self.net = PL_LoFTR(config, pretrained_ckpt=None, test_mode=True).matcher
        self.net.eval()
        
        # Apply ONNX-compatibility patches
        from patch_dinov2_for_onnx import patch_all_dinov2_models
        patch_all_dinov2_models(self.net)
        
        print(f"[MODEL] Created {model_name} model with ONNX patches")
    
    def load_checkpoint(self, ckpt_path: str):
        """Load checkpoint into the real MatchAnything model"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        print(f"[CKPT] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # The checkpoint should be compatible with PL_LoFTR
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Load weights - this should work since we're using the original model
        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        
        print(f"[WEIGHTS] Summary:")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        if missing:
            print(f"  Sample missing: {missing[:5]}")
        if unexpected:
            print(f"  Sample unexpected: {unexpected[:5]}")
        
        return len(missing) == 0 and len(unexpected) == 0
    
    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass using the real MatchAnything model.
        """
        # Convert RGB to grayscale for LoFTR
        img0_gray = 0.299 * image0[:, 0:1] + 0.587 * image0[:, 1:2] + 0.114 * image0[:, 2:3]
        img1_gray = 0.299 * image1[:, 0:1] + 0.587 * image1[:, 1:2] + 0.114 * image1[:, 2:3]
        
        # Create batch dictionary with all required fields for ROMA
        batch = {
            'image0': img0_gray,
            'image1': img1_gray,
            'image0_rgb_origin': image0,  # ROMA needs this
            'image1_rgb_origin': image1,  # ROMA needs this
        }
        
        # Run the real MatchAnything model
        self.net(batch)
        
        # Extract results (LoFTR stores results in batch)
        if 'mkpts0_f' in batch and 'mkpts1_f' in batch:
            keypoints0 = batch['mkpts0_f']
            keypoints1 = batch['mkpts1_f'] 
            mconf = batch.get('mconf', torch.ones(len(keypoints0), device=keypoints0.device))
        else:
            # Fallback for empty results
            device = image0.device
            keypoints0 = torch.empty((0, 2), dtype=torch.float32, device=device)
            keypoints1 = torch.empty((0, 2), dtype=torch.float32, device=device)
            mconf = torch.empty((0,), dtype=torch.float32, device=device)
        
        return {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "mconf": mconf,
        }

def export_real_matchanything_onnx(onnx_path: str, model_name: str = "matchanything_roma",
                                  H: int = 832, W: int = 832, 
                                  match_threshold: float = 0.1,
                                  ckpt: Optional[str] = None):
    """
    Export the real MatchAnything model to ONNX format.
    """
    device = "cpu"
    
    try:
        model = RealMatchAnythingTRT(
            model_name=model_name,
            img_resize=H,  # Use H as img_resize
            match_threshold=match_threshold
        ).to(device).eval()
        
        # Load checkpoint if provided
        if ckpt:
            success = model.load_checkpoint(ckpt)
            if success:
                print("[SUCCESS] Checkpoint loaded successfully")
            else:
                print("[WARNING] Checkpoint loading had issues, but proceeding...")
        
        # Create dummy inputs
        x1 = torch.rand(1, 3, H, W, device=device)
        x2 = torch.rand(1, 3, H, W, device=device)
        
        # Test forward pass
        with torch.no_grad():
            result = model(x1, x2)
            print("Dry run OK:")
            print(f"  keypoints0: {result['keypoints0'].shape}")
            print(f"  keypoints1: {result['keypoints1'].shape}")
            print(f"  mconf: {result['mconf'].shape}")

        # Export to ONNX
        dynamic_axes = {
            "image0": {0: "B", 2: "H", 3: "W"},
            "image1": {0: "B", 2: "H", 3: "W"},
            "keypoints0": {0: "num_matches"},
            "keypoints1": {0: "num_matches"},
            "mconf": {0: "num_matches"},
        }
        
        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
        
        torch.onnx.export(
            model, (x1, x2), onnx_path,
            input_names=["image0", "image1"],
            output_names=["keypoints0", "keypoints1", "mconf"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            verbose=False
        )
        print(f"[ONNX] Exported real MatchAnything model -> {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"[ERROR] Failed to export real MatchAnything model: {e}")
        print("[INFO] This might be due to missing dependencies or configuration issues")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="out/real_matchanything.onnx")
    parser.add_argument("--model", default="matchanything_roma")
    parser.add_argument("--ckpt", help="Checkpoint path")
    args = parser.parse_args()
    
    export_real_matchanything_onnx(
        args.onnx, args.model, ckpt=args.ckpt
    )