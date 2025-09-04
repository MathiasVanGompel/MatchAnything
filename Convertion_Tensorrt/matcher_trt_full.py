from typing import Dict, Optional, Tuple
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- ONNX-friendly local correlation ----------
def local_correlation_onnx(feat0: torch.Tensor,
                           feat1: torch.Tensor,
                           local_radius: int,
                           flow: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes correlation in a (2r+1)x(2r+1) window.
    Output: (B, K, H, W), K=(2r+1)^2
    """
    B, C, H, W = feat0.shape
    r = local_radius
    K = (2 * r + 1) ** 2

    if flow is None:
        patches = F.unfold(feat1, kernel_size=2*r+1, padding=r)  # (B, C*K, H*W)
        patches = patches.view(B, C, K, H, W)
        feat0_exp = feat0.unsqueeze(2)  # (B,C,1,H,W)
        corr = (feat0_exp * patches).sum(dim=1) / (C ** 0.5)  # (B,K,H,W)
        return corr

    ys = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=feat0.device, dtype=feat0.dtype)
    xs = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=feat0.device, dtype=feat0.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, H, W, 2)

    off_y = torch.linspace(-2*r/H, 2*r/H, 2*r+1, device=feat0.device, dtype=feat0.dtype)
    off_x = torch.linspace(-2*r/W, 2*r/W, 2*r+1, device=feat0.device, dtype=feat0.dtype)
    wy, wx = torch.meshgrid(off_y, off_x, indexing="ij")
    window = torch.stack([wx, wy], dim=-1).view(1, 1, 1, K, 2)

    flow_xy = flow.permute(0, 2, 3, 1)
    coords = base + 0.0*flow_xy
    coords = coords.unsqueeze(3) + window
    coords = coords.view(B, H, W*K, 2)

    sampled = F.grid_sample(feat1, coords, align_corners=False, mode="bilinear")  # (B,C,H,W*K)
    sampled = sampled.view(B, C, H, W, K).permute(0,4,1,2,3)  # (B,K,C,H,W)
    corr = (feat0.unsqueeze(1) * sampled).sum(dim=2) / (C ** 0.5)
    return corr

# ---------- ConvRefiner ----------
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=5, use_bn=True, bias=True):
        super().__init__()
        pad = k // 2
        layers = [nn.Conv2d(in_dim, out_dim, k, 1, pad, bias=bias)]
        if use_bn: layers += [nn.BatchNorm2d(out_dim, momentum=0.1)]
        layers += [nn.ReLU(inplace=True), nn.Conv2d(out_dim, out_dim, 1, 1, 0)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# Light-mode switch (optional) for low-VRAM
_LIGHT_MODE = bool(int(os.getenv("MA_LIGHT_MODE", "0")))
def _hidden_mult(c: int) -> int:
    return max(1, (1 if _LIGHT_MODE else 2))

class ConvRefiner(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 3,
                 local_radius: Optional[int] = None, concat_logits: bool = True):
        super().__init__()
        self.block1 = ConvBlock(in_dim, hidden_dim, k=5, use_bn=True, bias=True)
        self.res = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, k=5, use_bn=True, bias=True),
            ConvBlock(hidden_dim, hidden_dim, k=5, use_bn=True, bias=True),
        )
        self.out = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.local_radius = local_radius
        self.concat_logits = concat_logits

    def forward(self, fA: torch.Tensor, fB: torch.Tensor, flow: torch.Tensor, logits: Optional[torch.Tensor] = None):
        xhat = F.grid_sample(fB, flow.permute(0,2,3,1), align_corners=False, mode="bilinear")
        x = torch.cat([fA, xhat], dim=1)
        if logits is not None and self.concat_logits:
            x = torch.cat([x, logits], dim=1)
        x = self.block1(x)
        x = self.res(x)
        y = self.out(x).to(flow.dtype)
        delta, cert = y[:, :2], y[:, 2:3]
        return delta, cert

# ---------- GP cosine kernel + CG solver (no inverse) ----------
class CosKernel(nn.Module):
    def __init__(self, T: float = 1.0, learn_temperature: bool = False):
        super().__init__()
        self.learn = learn_temperature
        if learn_temperature:
            self.T = nn.Parameter(torch.tensor(float(T)))
        else:
            self.register_buffer("T_buf", torch.tensor(float(T)))

    def forward(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        num = torch.einsum("bnd,bmd->bnm", X, Y)
        den = (X.norm(dim=-1, keepdim=True) * Y.norm(dim=-1).unsqueeze(1) + eps)
        c = num / den
        T = (self.T.abs() + 0.01) if self.learn else self.T_buf
        return torch.exp((c - 1.0) / T)

def cg_solve(A: torch.Tensor, B: torch.Tensor, iters: int = 12, eps: float = 1e-9) -> torch.Tensor:
    """
    Solve A X = B for SPD A using Conjugate Gradient with fixed iterations.
    A: (B,M,M), B: (B,M,D) -> X: (B,M,D)
    """
    X = torch.zeros_like(B)
    R = B - torch.matmul(A, X)
    P = R.clone()
    rsold = (R * R).sum(dim=1, keepdim=True)
    for _ in range(iters):
        AP = torch.matmul(A, P)
        pAp = (P * AP).sum(dim=1, keepdim=True) + eps
        alpha = rsold / pAp
        X = X + alpha * P
        R = R - alpha * AP
        rsnew = (R * R).sum(dim=1, keepdim=True)
        beta = rsnew / (rsold + eps)
        P = R + beta * P
        rsold = rsnew
    return X

def _make_base_grid(h=16, w=16) -> torch.Tensor:
    ys = torch.linspace(-1 + 1/h, 1 - 1/h, h)
    xs = torch.linspace(-1 + 1/w, 1 - 1/w, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    # (1,2,h,w) in CPU float32; moved/typed on demand
    return torch.stack([gx, gy], dim=0).unsqueeze(0)

class GP(nn.Module):
    """
    Gaussian Process head with dynamic, TRT-friendly shape handling.
    """
    def __init__(self, gp_dim: int = 64, covar_size: int = 5, sigma_noise: float = 0.1, with_cov: bool = False):
        super().__init__()
        self.kernel = CosKernel(T=1.0, learn_temperature=False)
        self.pos_conv = nn.Conv2d(2, gp_dim, 1, 1, 0)
        self.gp_dim = gp_dim
        self.covar_size = covar_size
        self.sigma_noise = sigma_noise
        self.with_cov = with_cov
        # Small fixed grid we upsample later (avoid data-dependent grid sizes).
        self.register_buffer("base_grid", _make_base_grid(16, 16), persistent=False)

    def pos_enc(self, H: int, W: int, device, dtype) -> torch.Tensor:
        # Upsample a small (1,2,16,16) grid to (1,2,H,W), then 1x1-conv and cos.
        g = self.base_grid.to(device=device, dtype=dtype)
        g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
        return torch.cos(8 * math.pi * self.pos_conv(g))  # (1,D,H,W)

    def forward(self, fA: torch.Tensor, fB: torch.Tensor) -> torch.Tensor:
        B, C, H1, W1 = fA.shape
        _, _, H2, W2 = fB.shape

        # Coarse positional enc. sized to fB
        Phi = self.pos_enc(H2, W2, fB.device, fB.dtype)       # (1,D,H2,W2)

        # Flatten features to sequences
        X = fA.flatten(2).transpose(1, 2).contiguous()        # (B,N,C)
        Y = fB.flatten(2).transpose(1, 2).contiguous()        # (B,M,C)
        Fpos = Phi.expand(B, -1, -1, -1).flatten(2).transpose(1, 2).contiguous()  # (B,M,D)

        # Kernel stuff
        Kyy = self.kernel(Y, Y)                               # (B,M,M)
        Kxy = self.kernel(X, Y)                               # (B,N,M)

        M = Y.size(1)
        I = torch.eye(M, device=fB.device, dtype=fB.dtype).unsqueeze(0)   # (1,M,M)
        Kyyn = Kyy + self.sigma_noise * I                                 # (B,M,M)
        Xsol = cg_solve(Kyyn, Fpos, iters=12)                             # (B,M,D)

        mu = torch.matmul(Kxy, Xsol)                                      # (B,N,D)
        mu = mu.transpose(1, 2).contiguous()                              # (B,D,N)

        # ---- DYNAMIC RESHAPE (critical) ----
        # Use a dynamic reference shaped like (B, gp_dim, H1, W1)
        # so ONNX exports Reshape(shape=Shape(ref)) instead of hard-coded ints.
        ref = fA[:, : self.gp_dim, :, :]                                  # (B,gp_dim,H1,W1)
        mu = mu.reshape_as(ref)                                           # (B,gp_dim,H1,W1)
        return mu  # covariance omitted

# ---------- Embedding decoder head (flow + certainty) ----------
class EmbeddingDecoder(nn.Module):
    """
    in_ch should equal gp_dim + proj(A)_channels at GP scales.
    """
    def __init__(self, in_ch: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_classifier = False
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 3, 1, 1, 0),
        )

    def scales(self): return [16, 8]

    def forward(self, gp: torch.Tensor, fA: torch.Tensor, old: torch.Tensor, new_scale: str):
        x = torch.cat([gp, fA], dim=1)  # channel dim must match in_ch
        y = self.head(x)
        flow = y[:, :2]
        cert = y[:, 2:3]
        return flow, cert, old

# ---------- Decoder (multi-scale) ----------
class Decoder(nn.Module):
    def __init__(self,
                 embedding_decoder: EmbeddingDecoder,
                 gps: Dict[str, GP],
                 proj: Dict[str, nn.Sequential],
                 conv_refiner: Dict[str, ConvRefiner],
                 flow_upsample_mode: str = "bilinear"):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.gps = nn.ModuleDict(gps)
        self.proj = nn.ModuleDict(proj)
        self.conv_refiner = nn.ModuleDict(conv_refiner)
        self.flow_upsample_mode = flow_upsample_mode
        # Base normalized grid for dynamic init at the coarsest scale
        self.register_buffer("base_grid", _make_base_grid(16, 16), persistent=False)

    def _grid_like(self, B: int, H: int, W: int, device, dtype) -> torch.Tensor:
        g = self.base_grid.to(device=device, dtype=dtype)
        g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)  # (1,2,H,W)
        return g.expand(B, -1, -1, -1)  # (B,2,H,W)

    def forward(self, f1: Dict[str, torch.Tensor], f2: Dict[str, torch.Tensor], sizes: Dict[str, Tuple[int,int]]):
        ordered = ["16", "8", "4", "2", "1"]
        scales = [s for s in ordered if s in f1]
        coarsest = scales[0]

        Hc, Wc = sizes[coarsest]
        B = f1[coarsest].size(0)
        flow = self._grid_like(B, Hc, Wc, f1[coarsest].device, f1[coarsest].dtype)  # start as coords in [-1,1]
        cert = torch.zeros(B, 1, Hc, Wc, device=flow.device, dtype=flow.dtype)
        old = torch.zeros(B, self.embedding_decoder.hidden_dim, Hc, Wc, device=flow.device, dtype=flow.dtype)

        for s in scales:
            A = f1[s]; B2 = f2[s]
            if s in self.proj:
                A = self.proj[s](A); B2 = self.proj[s](B2)

            if s in self.gps:
                gp = self.gps[s](A, B2)
                gflow, gcert, old = self.embedding_decoder(gp, A, old, s)
                flow = gflow.detach()
                cert = gcert

            if s in self.conv_refiner:
                dflow, dcert = self.conv_refiner[s](A, B2, flow, cert)
                flow = flow + dflow
                cert = cert + dcert

            idx = scales.index(s)
            if idx + 1 < len(scales):
                ns = scales[idx + 1]
                Hn, Wn = sizes[ns]
                flow = F.interpolate(flow, size=(Hn, Wn), mode=self.flow_upsample_mode, align_corners=False)
                cert = F.interpolate(cert, size=(Hn, Wn), mode=self.flow_upsample_mode, align_corners=False)

        return {"flow": flow, "certainty": cert}