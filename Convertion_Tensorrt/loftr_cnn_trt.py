#!/usr/bin/env python3
"""
LoFTR CNN-based TensorRT implementation.
This matches the checkpoint structure we observed: matcher.model.encoder.cnn.*
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class CNNEncoder(nn.Module):
    """
    CNN encoder that matches the checkpoint structure.
    Based on the checkpoint keys: matcher.model.encoder.cnn.layers.*
    """
    def __init__(self):
        super().__init__()
        
        # Create CNN layers based on observed checkpoint structure
        # The checkpoint shows layers 0,1,3,4,7,8,10,11,14 etc.
        self.layers = nn.ModuleList([
            # Layer 0: Conv2d
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # grayscale input
            # Layer 1: BatchNorm2d  
            nn.BatchNorm2d(64),
            # Layer 2: ReLU (not in checkpoint, inplace)
            nn.ReLU(inplace=True),
            # Layer 3: Conv2d
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # Layer 4: BatchNorm2d
            nn.BatchNorm2d(64),
            # Layer 5: ReLU
            nn.ReLU(inplace=True),
            # Layer 6: MaxPool2d
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 7: Conv2d
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # Layer 8: BatchNorm2d
            nn.BatchNorm2d(128),
            # Layer 9: ReLU
            nn.ReLU(inplace=True),
            # Layer 10: Conv2d
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # Layer 11: BatchNorm2d
            nn.BatchNorm2d(128),
            # Layer 12: ReLU
            nn.ReLU(inplace=True),
            # Layer 13: MaxPool2d
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 14: Conv2d
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        ])
    
    def forward(self, x):
        """Forward pass through CNN layers"""
        for layer in self.layers:
            x = layer(x)
        return x

class LoFTRCNNTRT(nn.Module):
    """
    LoFTR CNN-based model for TensorRT that matches the checkpoint structure.
    """
    def __init__(self, match_threshold: float = 0.1):
        super().__init__()
        self.match_threshold = match_threshold
        self.encoder = CNNEncoder()
        
        # Simple matcher head (placeholder - will be refined based on checkpoint)
        self.matcher_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 256*2 from concat
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),  # confidence map
        )
    
    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass matching LoFTR-style processing.
        
        Args:
            image0: RGB image [B, 3, H, W] in range [0, 1]  
            image1: RGB image [B, 3, H, W] in range [0, 1]
            
        Returns:
            Dictionary with keypoints0, keypoints1, mconf
        """
        device = image0.device
        B, C, H, W = image0.shape
        
        # Convert to grayscale
        img0_gray = 0.299 * image0[:, 0:1] + 0.587 * image0[:, 1:2] + 0.114 * image0[:, 2:3]
        img1_gray = 0.299 * image1[:, 0:1] + 0.587 * image1[:, 1:2] + 0.114 * image1[:, 2:3]
        
        # Extract features
        feat0 = self.encoder(img0_gray)  # [B, 256, H/4, W/4]
        feat1 = self.encoder(img1_gray)  # [B, 256, H/4, W/4]
        
        # Simple correlation-based matching
        # This is a placeholder - real LoFTR has more complex attention mechanism
        _, C_feat, H_feat, W_feat = feat0.shape
        
        # Flatten features for correlation
        feat0_flat = feat0.view(B, C_feat, -1)  # [B, C, HW]
        feat1_flat = feat1.view(B, C_feat, -1)  # [B, C, HW]
        
        # Normalize features
        feat0_norm = F.normalize(feat0_flat, dim=1)
        feat1_norm = F.normalize(feat1_flat, dim=1)
        
        # Compute correlation matrix
        correlation = torch.bmm(feat0_norm.transpose(1, 2), feat1_norm)  # [B, HW, HW]
        
        # Find best matches
        max_corr, max_indices = torch.max(correlation, dim=2)  # [B, HW]
        
        # Apply confidence threshold
        conf_mask = max_corr > self.match_threshold
        
        matches_found = []
        confidences_found = []
        kpts0_found = []
        kpts1_found = []
        
        for b in range(B):
            # Get confident matches for this batch
            batch_mask = conf_mask[b]
            if not torch.any(batch_mask):
                continue
                
            # Get indices of confident matches
            confident_indices = torch.nonzero(batch_mask, as_tuple=False).squeeze(1)
            
            # Convert linear indices to 2D coordinates
            y0_coords = confident_indices // W_feat
            x0_coords = confident_indices % W_feat
            
            # Get corresponding points in image1
            corr_indices = max_indices[b][confident_indices]
            y1_coords = corr_indices // W_feat
            x1_coords = corr_indices % W_feat
            
            # Scale to full resolution (4x upsampling)
            scale_factor = H / H_feat  # Should be 4.0
            
            kpts0 = torch.stack([x0_coords.float() * scale_factor, 
                                y0_coords.float() * scale_factor], dim=1)
            kpts1 = torch.stack([x1_coords.float() * scale_factor,
                                y1_coords.float() * scale_factor], dim=1)
            
            confidences = max_corr[b][confident_indices]
            
            kpts0_found.append(kpts0)
            kpts1_found.append(kpts1)
            confidences_found.append(confidences)
        
        # Concatenate all matches
        if kpts0_found:
            all_kpts0 = torch.cat(kpts0_found, dim=0)
            all_kpts1 = torch.cat(kpts1_found, dim=0)
            all_conf = torch.cat(confidences_found, dim=0)
        else:
            all_kpts0 = torch.empty((0, 2), dtype=torch.float32, device=device)
            all_kpts1 = torch.empty((0, 2), dtype=torch.float32, device=device)
            all_conf = torch.empty((0,), dtype=torch.float32, device=device)
        
        return {
            "keypoints0": all_kpts0,
            "keypoints1": all_kpts1,
            "mconf": all_conf,
        }

def export_loftr_cnn_onnx(onnx_path: str, H: int = 832, W: int = 832,
                         match_threshold: float = 0.1, ckpt: Optional[str] = None):
    """
    Export LoFTR CNN model to ONNX.
    """
    device = "cpu"
    model = LoFTRCNNTRT(match_threshold=match_threshold).to(device).eval()
    
    # Load checkpoint if provided
    if ckpt:
        print(f"[CKPT] Loading checkpoint: {ckpt}")
        try:
            from weight_adapter import remap_and_load
            
            # Update weight adapter rules for CNN structure
            loaded_weights = remap_and_load(model, ckpt_path=ckpt, save_sanitized=None)
            if len(loaded_weights) > 0:
                print(f"[SUCCESS] Loaded {len(loaded_weights)} weight tensors")
            else:
                print("[WARNING] No weights loaded - using random initialization")
        except Exception as e:
            print(f"[ERROR] Checkpoint loading failed: {e}")
            print("[INFO] Using random initialization")
    
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
    
    import os
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
    print(f"[ONNX] Exported LoFTR CNN model -> {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    # Test export
    export_loftr_cnn_onnx("out/loftr_cnn_test.onnx")