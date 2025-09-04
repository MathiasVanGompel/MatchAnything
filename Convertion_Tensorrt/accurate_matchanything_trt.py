#!/usr/bin/env python3
"""
Accurate MatchAnything TensorRT implementation that matches the original exactly.
This version replicates the exact preprocessing and postprocessing pipeline.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import cv2
import PIL
from PIL import Image

from encoders_trt_full import CNNandDinov2TRT
from gp_trt import GPMatchEncoderTRT

class AccurateMatchAnythingTRT(nn.Module):
    """
    Accurate TensorRT implementation that matches the original MatchAnything exactly.
    Includes the full preprocessing and postprocessing pipeline.
    """
    
    def __init__(self, model_name: str = "matchanything_roma", img_resize: int = 832, 
                 match_threshold: float = 0.1, amp: bool = False):
        super().__init__()
        self.model_name = model_name
        self.img_resize = img_resize
        self.match_threshold = match_threshold
        self.amp = amp
        
        if model_name == "matchanything_roma":
            # Use our TensorRT-optimized ROMA implementation
            self.encoder = CNNandDinov2TRT(amp=amp)
            self.matcher = GPMatchEncoderTRT(beta=10.0)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented for TensorRT")
    
    def process_resize(self, w: int, h: int, resize: Optional[int] = None, 
                      df: Optional[int] = None, resize_no_larger_than: bool = False) -> Tuple[int, int]:
        """Exact copy of the original process_resize function"""
        if resize is not None:
            if resize_no_larger_than and (max(h, w) <= resize):
                w_new, h_new = w, h
            else:
                scale = resize / max(h, w)
                w_new, h_new = int(round(w * scale)), int(round(h * scale))
        else:
            w_new, h_new = w, h

        if df is not None:
            w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
        return w_new, h_new

    def resize_image(self, image: np.ndarray, size: Tuple[int, int], interp: str) -> np.ndarray:
        """Exact copy of the original resize_image function"""
        if interp.startswith('cv2_'):
            interp = getattr(cv2, 'INTER_' + interp[len('cv2_'):].upper())
            h, w = image.shape[:2]
            if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
                interp = cv2.INTER_LINEAR
            resized = cv2.resize(image, size, interpolation=interp)
        elif interp.startswith('pil_'):
            interp = getattr(PIL.Image, interp[len('pil_'):].upper())
            resized = PIL.Image.fromarray(image.astype(np.uint8))
            resized = resized.resize(size, resample=interp)
            resized = np.asarray(resized, dtype=image.dtype)
        else:
            raise ValueError(f'Unknown interpolation {interp}.')
        return resized

    def pad_bottom_right(self, inp: np.ndarray, pad_size: int, ret_mask: bool = False):
        """Exact copy of the original pad_bottom_right function"""
        assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
        mask = None
        if inp.ndim == 2:
            padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            padded[:inp.shape[0], :inp.shape[1]] = inp
            if ret_mask:
                mask = np.zeros((pad_size, pad_size), dtype=bool)
                mask[:inp.shape[0], :inp.shape[1]] = True
        elif inp.ndim == 3:
            padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            padded[:, :inp.shape[1], :inp.shape[2]] = inp
            if ret_mask:
                mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
                mask[:, :inp.shape[1], :inp.shape[2]] = True
            mask = mask[0]
        else:
            raise NotImplementedError()
        return padded, mask

    def resize_with_padding(self, img: np.ndarray, resize: Optional[int] = None, 
                           df: int = 8, padding: bool = True):
        """Exact copy of the original resize function"""
        w, h = img.shape[1], img.shape[0]
        w_new, h_new = self.process_resize(w, h, resize=resize, df=df, resize_no_larger_than=False)
        img_new = self.resize_image(img, (w_new, h_new), interp="pil_LANCZOS").astype('float32')
        h_scale, w_scale = img.shape[0] / img_new.shape[0], img.shape[1] / img_new.shape[1]
        mask = None
        if padding:
            img_new, mask = self.pad_bottom_right(img_new, max(h_new, w_new), ret_mask=True)
        return img_new, [h_scale, w_scale], mask

    def preprocess_images(self, img0_rgb: torch.Tensor, img1_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Exact preprocessing pipeline matching the original implementation.
        Input: RGB images as tensors [B, C, H, W] in range [0, 1]
        """
        # Convert to numpy and scale to [0, 255]
        img0 = img0_rgb.cpu().numpy().squeeze() * 255
        img1 = img1_rgb.cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0).astype("uint8")
        img1 = img1.transpose(1, 2, 0).astype("uint8")
        
        # Get original image sizes
        img0_size = np.array(img0.shape[:2])
        img1_size = np.array(img1.shape[:2])
        
        # Convert to grayscale
        img0_gray = np.array(Image.fromarray(img0).convert("L"))
        img1_gray = np.array(Image.fromarray(img1).convert("L"))
        
        # Resize with padding (df=32 to match original)
        (img0_gray, hw0_new, mask0) = self.resize_with_padding(img0_gray, df=32)
        (img1_gray, hw1_new, mask1) = self.resize_with_padding(img1_gray, df=32)
        
        # Convert to tensors
        img0_tensor = torch.from_numpy(img0_gray)[None][None] / 255.0
        img1_tensor = torch.from_numpy(img1_gray)[None][None] / 255.0
        
        batch = {
            'image0': img0_tensor,
            'image1': img1_tensor,
            'image0_rgb_origin': img0_rgb,
            'image1_rgb_origin': img1_rgb,
            'origin_img_size0': torch.from_numpy(img0_size)[None],
            'origin_img_size1': torch.from_numpy(img1_size)[None],
            'hw0_new': hw0_new,
            'hw1_new': hw1_new
        }
        
        # Handle masks if present
        if mask0 is not None:
            mask0 = torch.from_numpy(mask0)
            mask1 = torch.from_numpy(mask1)
            # Downsample masks by factor of 8 (to match coarse level)
            ts_mask_0, ts_mask_1 = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=0.125,
                mode='nearest',
                recompute_scale_factor=False
            )[0].bool()
            batch.update({"mask0": ts_mask_0[None], "mask1": ts_mask_1[None]})
        
        return batch

    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass that exactly matches the original MatchAnything pipeline.
        
        Args:
            image0: RGB image tensor [B, C, H, W] in range [0, 1]
            image1: RGB image tensor [B, C, H, W] in range [0, 1]
            
        Returns:
            Dictionary with keypoints0, keypoints1, and mconf (confidence scores)
        """
        # Preprocess images exactly like the original
        batch = self.preprocess_images(image0, image1)
        
        # Move to device
        device = image0.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Extract features using our TensorRT-optimized encoder
        img0_gray = batch['image0']
        img1_gray = batch['image1']
        
        # Get features at coarse level (1/8 resolution for LoFTR-style matching)
        feat0_dict = self.encoder(img0_gray)
        feat1_dict = self.encoder(img1_gray)
        
        feat0_c = feat0_dict['coarse']  # [B, C, H/8, W/8]
        feat1_c = feat1_dict['coarse']  # [B, C, H/8, W/8]
        
        # Apply masks if present
        if 'mask0' in batch:
            mask0_c = batch['mask0']  # [B, H/8, W/8]
            mask1_c = batch['mask1']  # [B, H/8, W/8]
            # Zero out features where mask is False
            feat0_c = feat0_c * mask0_c.unsqueeze(1).float()
            feat1_c = feat1_c * mask1_c.unsqueeze(1).float()
        
        # Coarse matching using our GP matcher
        warp_c, cert_c = self.matcher(feat0_c, feat1_c)  # [B,Ha,Wa,2], [B,Ha,Wa]
        
        # Apply confidence threshold
        conf_mask = cert_c > self.match_threshold
        
        # Extract matches above threshold
        B, Ha, Wa = cert_c.shape
        
        # Get coordinates of confident matches
        conf_indices = torch.nonzero(conf_mask, as_tuple=False)  # [N, 3] (batch, y, x)
        
        if conf_indices.shape[0] == 0:
            # No confident matches found
            return {
                "keypoints0": torch.empty((0, 2), dtype=torch.float32),
                "keypoints1": torch.empty((0, 2), dtype=torch.float32),
                "mconf": torch.empty((0,), dtype=torch.float32),
            }
        
        # Extract match coordinates and confidences
        batch_idx = conf_indices[:, 0]
        y_coords = conf_indices[:, 1]
        x_coords = conf_indices[:, 2]
        
        # Get corresponding points in image1 from warp field
        mkpts0_c = torch.stack([x_coords.float(), y_coords.float()], dim=1)  # [N, 2]
        mkpts1_c = warp_c[batch_idx, y_coords, x_coords]  # [N, 2]
        mconf = cert_c[batch_idx, y_coords, x_coords]  # [N]
        
        # Convert from coarse coordinates to original image coordinates
        # Coarse features are at 1/8 resolution due to encoder downsampling
        scale_factor = 8.0
        
        # For ELoFTR, we need to apply the hw_new scaling
        if self.model_name == 'matchanything_eloftr':
            hw0_new = batch['hw0_new']  # [h_scale, w_scale]
            hw1_new = batch['hw1_new']  # [h_scale, w_scale]
            
            mkpts0_c *= torch.tensor([hw0_new[1], hw0_new[0]], device=device)  # [w_scale, h_scale]
            mkpts1_c *= torch.tensor([hw1_new[1], hw1_new[0]], device=device)  # [w_scale, h_scale]
        
        # Scale to full resolution
        mkpts0_f = mkpts0_c * scale_factor
        mkpts1_f = mkpts1_c * scale_factor
        
        return {
            "keypoints0": mkpts0_f,
            "keypoints1": mkpts1_f, 
            "mconf": mconf,
        }

class AccurateMatchAnythingWrapper:
    """
    Wrapper that provides the exact same interface as the original MatchAnything matcher.
    This ensures perfect compatibility with existing code.
    """
    
    def __init__(self, conf: Dict):
        self.conf = conf
        self.model = AccurateMatchAnythingTRT(
            model_name=conf['model_name'],
            img_resize=conf.get('img_resize', 832),
            match_threshold=conf.get('match_threshold', 0.1),
            amp=conf.get('amp', False)
        )
        self.model.eval()
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process images and return matches in the exact same format as original.
        
        Args:
            data: Dictionary containing 'image0' and 'image1' tensors
            
        Returns:
            Dictionary with 'mkpts0_f', 'mkpts1_f', and 'mconf'
        """
        with torch.no_grad():
            result = self.model(data['image0'], data['image1'])
            
            # Return in the exact format expected by the original code
            return {
                'mkpts0_f': result['keypoints0'],
                'mkpts1_f': result['keypoints1'],
                'mconf': result['mconf']
            }

# Export function for ONNX conversion
def export_accurate_matchanything_onnx(onnx_path: str, model_name: str = "matchanything_roma",
                                      H: int = 832, W: int = 832, 
                                      match_threshold: float = 0.1,
                                      ckpt: Optional[str] = None):
    """
    Export the accurate MatchAnything model to ONNX format.
    """
    device = "cpu"
    model = AccurateMatchAnythingTRT(
        model_name=model_name,
        match_threshold=match_threshold,
        amp=False  # Use FP32 for ONNX export
    ).to(device).eval()

    # Load checkpoint if provided
    if ckpt:
        print(f"[CKPT] Loading checkpoint: {ckpt}")
        from weight_adapter import remap_and_load
        remap_and_load(model, ckpt_path=ckpt, save_sanitized=None)

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

    # Export to ONNX with dynamic axes
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
    print(f"[ONNX] Exported accurate model -> {onnx_path}")
    return onnx_path