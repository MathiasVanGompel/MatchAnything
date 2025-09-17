import torch, torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
    def forward(self, x):
        return x / (x.norm(p=2, dim=self.dim, keepdim=True) + self.eps)

class GPMatchEncoderTRT(nn.Module):
    def __init__(self, beta: float = 14.285714285714286):  # Beta defaults to 1/0.07.
        super().__init__()
        self.beta = float(beta)
        self.l2 = L2Norm(dim=1, eps=1e-6)

    @torch.no_grad()
    def forward(self, f0: torch.Tensor, f1: torch.Tensor):
        # Flatten coarse features from [B, C, Hc, Wc] to [B, N, C] for both images.
        B, C, Ha, Wa = f0.shape
        Hb, Wb = f1.shape[2], f1.shape[3]
        Na, Nb = Ha * Wa, Hb * Wb

        a = self.l2(f0.view(B, C, Na).transpose(1, 2))      # Normalized tokens for image0 with shape [B, Na, C].
        b = self.l2(f1.view(B, C, Nb).transpose(1, 2))      # Normalized tokens for image1 with shape [B, Nb, C].

        # Compute similarities in float32 for numerical stability.
        sim = (a.float() @ b.float().transpose(1, 2))        # Pairwise scores with shape [B, Na, Nb].
        sim = (self.beta * sim).contiguous()

        attn = F.softmax(sim, dim=2)                         # Softmax stays in float32.
        # Build the coordinate grid in float32.
        ys = torch.arange(Hb, device=f0.device, dtype=torch.float32).view(Hb, 1).repeat(1, Wb)
        xs = torch.arange(Wb, device=f0.device, dtype=torch.float32).view(1, Wb).repeat(Hb, 1)
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).unsqueeze(0)  # [1,Nb,2]
        tgt = attn @ coords                                  # Weighted coordinates with shape [B, Na, 2].

        warp = tgt.view(B, Ha, Wa, 2).permute(0, 3, 1, 2)    # Warp grid shaped as [B, 2, Ha, Wa].
        cert = attn.max(dim=2)[0].view(B, 1, Ha, Wa)         # Certainty map shaped as [B, 1, Ha, Wa].

        # Cast outputs back to the original dtype.
        warp = warp.to(f0.dtype); cert = cert.to(f0.dtype)
        return warp, cert
