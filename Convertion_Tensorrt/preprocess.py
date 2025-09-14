import math
import torch
import numpy as np
import cv2

def pad_to_multiple(img, multiple=14):
    h, w = img.shape[-2:]
    nh = int(math.ceil(h / multiple) * multiple)
    nw = int(math.ceil(w / multiple) * multiple)
    pad_h, pad_w = nh - h, nw - w
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0, 0, 0)
    img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
    return img, (0, pad_w, 0, pad_h)

def preprocess_rgb(image_bchw_float01, multiple=14):
    img, pads = pad_to_multiple(image_bchw_float01, multiple=multiple)
    return img, pads

def unpad_like(out, pads):
    _, pw, _, ph = pads
    if ph or pw:
        return out[..., : -ph or None, : -pw or None]
    return out

