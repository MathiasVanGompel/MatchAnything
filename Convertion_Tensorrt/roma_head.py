# -*- coding: utf-8 -*-
"""
A compact RoMa-style coarse head in PyTorch:
- L2-normalize features
- correlation (N x N) via matmul
- softmax -> soft-argmax to get continuous coords on the other image
- certainty from max probability (or low entropy)

Inputs: f0,f1 as numpy [1, C, Hc, Wc] from TRT encoder.
Outputs:
  warp_c: [1, 2, Hc, Wc] (x,y) in *cell* coordinates (multiply by 14 for pixel coords)
  cert_c: [1, 1, Hc, Wc] confidence [0..1]
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

def _grid_xy(Hc: int, Wc: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(Hc, device=device).float()
    xs = torch.arange(Wc, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy  # [Hc,Wc]

@torch.no_grad()
def coarse_from_features(f0_np: np.ndarray, f1_np: np.ndarray, temperature: float = 0.07):
    """
    f0_np, f1_np: [1, C, Hc, Wc], float32/float16 ok
    Returns:
      warp_c: [1,2,Hc,Wc] (x,y)
      cert_c: [1,1,Hc,Wc]
    """
    assert f0_np.ndim == 4 and f1_np.ndim == 4
    _, C, Hc, Wc = f0_np.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f0 = torch.from_numpy(f0_np).to(device=device, dtype=torch.float32)
    f1 = torch.from_numpy(f1_np).to(device=device, dtype=torch.float32)

    # L2-normalize along channel
    f0 = F.normalize(f0, dim=1)
    f1 = F.normalize(f1, dim=1)

    # flatten: [1,C,N]
    N = Hc * Wc
    f0v = f0.view(1, C, N)
    f1v = f1.view(1, C, N)

    # correlation: [1,N,N] = f0^T @ f1  (via bmm)
    # (1,N,C) @ (1,C,N) -> (1,N,N)
    sim = torch.bmm(f0v.transpose(1, 2), f1v)  # cosine since normalized

    # softmax with temperature on dim=2 (over positions in image1)
    prob = F.softmax(sim / max(1e-6, temperature), dim=2)  # [1,N,N]

    # expected (x,y) in image1 for each location in image0
    xx, yy = _grid_xy(Hc, Wc, device)
    xp = xx.flatten()[None, None, :]  # [1,1,N]
    yp = yy.flatten()[None, None, :]

    # [1,N,N] @ [1,N] -> [1,N]
    x1_hat = torch.bmm(prob, xp.transpose(1, 2)).squeeze(2)  # [1,N]
    y1_hat = torch.bmm(prob, yp.transpose(1, 2)).squeeze(2)  # [1,N]

    # certainty: use max probability per row (peakiness)
    conf, _ = prob.max(dim=2)  # [1,N]

    # reshape back to maps
    warp_x = x1_hat.view(1, 1, Hc, Wc)
    warp_y = y1_hat.view(1, 1, Hc, Wc)
    warp_c = torch.cat([warp_x, warp_y], dim=1)  # [1,2,Hc,Wc]
    cert_c = conf.view(1, 1, Hc, Wc)

    return warp_c.cpu().numpy().astype(np.float32), cert_c.cpu().numpy().astype(np.float32)
