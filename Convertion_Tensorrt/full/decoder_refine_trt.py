# decoder_refine_trt.py
# TRT-friendly RoMa-style match decoder + optional fine refinement CNN

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers ----------
class FP32Softmax(nn.Module):
    """Always computes softmax in float32 and RETURNS float32."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x.float(), dim=self.dim)  # keep fp32

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossAttention(nn.Module):
    """Q from A, KV from B — Linear/MatMul/Softmax for TRT."""
    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=True)
        self.to_k = nn.Linear(dim, inner, bias=True)
        self.to_v = nn.Linear(dim, inner, bias=True)
        self.to_out = nn.Linear(inner, dim, bias=True)
        self.softmax = FP32Softmax(dim=-1)

    def forward(self, qx: torch.Tensor, kx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # qx: [B, Na, C], kx: [B, Nb, C]
        B, Na, _ = qx.shape
        Nb = kx.shape[1]
        q = self.to_q(qx)
        k = self.to_k(kx)
        v = self.to_v(kx)
        H = self.heads
        q = q.view(B, Na, H, -1).transpose(1, 2)   # [B,H,Na,hd]
        k = k.view(B, Nb, H, -1).transpose(1, 2)   # [B,H,Nb,hd]
        v = v.view(B, Nb, H, -1).transpose(1, 2)   # [B,H,Nb,hd]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Na,Nb]
        attn = self.softmax(attn)                                  # **fp32**
        out = torch.matmul(attn, v.float())                        # promote v -> fp32 for matmul
        out = out.transpose(1, 2).contiguous().view(B, Na, -1)     # [B,Na,H*hd]
        return self.to_out(out), attn  # to_out runs fine with fp32

class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64, ff_mult: int = 4):
        super().__init__()
        self.xattn = PreNorm(dim, CrossAttention(dim, heads, head_dim))
        self.ff = PreNorm(dim, FeedForward(dim, dim * ff_mult))
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        x, attn = self.xattn(A, B)
        A = A + x
        A = A + self.ff(A)
        return A, attn

class MatchDecoderTRT(nn.Module):
    """
    RoMa-style match decoder: predicts anchor probabilities over B for each A token,
    then converts them to a coarse warp and certainty.
    """
    def __init__(self, dim: int, depth: int = 4, heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, heads, head_dim) for _ in range(depth)])
        self.softmax = FP32Softmax(dim=-1)

    def forward_logits(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            A, _ = blk(A, B)
        C = A.shape[-1]
        logits = (A @ B.transpose(1, 2)) / (C ** 0.5)  # [B,Na,Nb]
        return logits

    @torch.no_grad()
    def forward(self, fA: torch.Tensor, fB: torch.Tensor, Ha: int, Wa: int, Hb: int, Wb: int):
        # fA/fB: [B,C,Ha,Wa] / [B,C,Hb,Wb]
        B = fA.shape[0]
        A = fA.flatten(2).transpose(1, 2).contiguous()    # [B,Na,C]
        Btok = fB.flatten(2).transpose(1, 2).contiguous() # [B,Nb,C]
        logits = self.forward_logits(A, Btok)
        attn = self.softmax(logits)                       # **fp32**

        # coords in **fp32**
        ys = torch.arange(Hb, device=fA.device, dtype=torch.float32).view(Hb, 1).repeat(1, Wb)
        xs = torch.arange(Wb, device=fA.device, dtype=torch.float32).view(1, Wb).repeat(Hb, 1)
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).unsqueeze(0)  # [1,Nb,2]
        tgt = attn @ coords                              # [B,Na,2]  (fp32)

        warp = tgt.view(B, Ha * Wa, 2).transpose(1, 2).view(B, 2, Ha, Wa)
        cert = attn.max(dim=2)[0].view(B, 1, Ha, Wa)
        # cast OUTS back to fp16 if encoder ran in fp16
        return warp.to(fA.dtype), cert.to(fA.dtype)

class RefineCNNTRT(nn.Module):
    """
    Tiny refinement CNN. Consumes (warp, certainty[, aux]) and predicts residual flow.
    ONNX/TRT-friendly; not a RoMa replica but aligned with coarse-to-fine idea.
    """
    def __init__(self, in_ch: int = 3, hidden: int = 64, iters: int = 1):
        super().__init__()
        self.iters = iters
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, 3, 3, padding=1)  # 2 for delta, 1 for cert delta
        self.act = nn.GELU()
        self.sig = nn.Sigmoid()

    @torch.no_grad()
    def forward(self, warp: torch.Tensor, certainty: torch.Tensor, aux: Optional[torch.Tensor] = None):
        x = torch.cat([warp, certainty], dim=1) if aux is None else torch.cat([warp, certainty, aux], dim=1)
        for _ in range(self.iters):
            y = self.act(self.conv1(x))
            y = self.act(self.conv2(y))
            upd = self.conv3(y)
            delta = upd[:, :2]
            dc = upd[:, 2:3]
            warp = warp + delta
            certainty = torch.clamp(certainty + self.sig(dc) - 0.5, 0.0, 1.0)
            x = torch.cat([warp, certainty], dim=1) if aux is None else torch.cat([warp, certainty, aux], dim=1)
        return warp, certainty
