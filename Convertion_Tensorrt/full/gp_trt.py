import torch, torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
    def forward(self, x):
        return x / (x.norm(p=2, dim=self.dim, keepdim=True) + self.eps)

class GPMatchEncoderTRT(nn.Module):
    def __init__(self, beta: float = 14.285714285714286):  # 1/0.07
        super().__init__()
        self.beta = float(beta)
        self.l2 = L2Norm(dim=1, eps=1e-6)

    @torch.no_grad()
    def forward(self, f0: torch.Tensor, f1: torch.Tensor):
        # f*: [B,C,Hc,Wc] -> [B,Na,C] and [B,Nb,C]
        B, C, Ha, Wa = f0.shape
        Hb, Wb = f1.shape[2], f1.shape[3]
        Na, Nb = Ha * Wa, Hb * Wb

        a = self.l2(f0.view(B, C, Na).transpose(1, 2))      # [B,Na,C]
        b = self.l2(f1.view(B, C, Nb).transpose(1, 2))      # [B,Nb,C]

        # Compute sim in FP32 for stable softmax
        sim = (a.float() @ b.float().transpose(1, 2))        # [B,Na,Nb]
        sim = (self.beta * sim).contiguous()

        attn = F.softmax(sim, dim=2)                         # FP32 softmax
        # Build coordinate grid in FP32
        ys = torch.arange(Hb, device=f0.device, dtype=torch.float32).view(Hb, 1).repeat(1, Wb)
        xs = torch.arange(Wb, device=f0.device, dtype=torch.float32).view(1, Wb).repeat(Hb, 1)
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).unsqueeze(0)  # [1,Nb,2]
        tgt = attn @ coords                                  # [B,Na,2] in FP32

        warp = tgt.view(B, Ha, Wa, 2).permute(0, 3, 1, 2)    # [B,2,Ha,Wa]
        cert = attn.max(dim=2)[0].view(B, 1, Ha, Wa)         # [B,1,Ha,Wa]

        # Cast back to original dtype
        warp = warp.to(f0.dtype); cert = cert.to(f0.dtype)
        return warp, cert
