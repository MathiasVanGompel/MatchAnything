#!/usr/bin/env python3
"""
ONNX-compatible attention layer to replace MemEffAttention
"""

import torch
import torch.nn as nn
from torch import Tensor


class ONNXCompatibleAttention(nn.Module):
    """
    ONNX-compatible version of the attention mechanism.
    This replaces the MemEffAttention to avoid xformers dependency.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        """
        ONNX-compatible forward pass using standard PyTorch operations only.
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        
        # Split into q, k, v using indexing (ONNX compatible)
        q = qkv[0] * self.scale  # [B, num_heads, N, head_dim]
        k = qkv[1]               # [B, num_heads, N, head_dim]
        v = qkv[2]               # [B, num_heads, N, head_dim]
        
        # Attention computation
        attn = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        
        # Apply attention bias if provided
        if attn_bias is not None:
            attn = attn + attn_bias
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x