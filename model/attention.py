# implementing causal self-attention which uses single head math, multi-head module and grouped query attention
import math

import torch
from torch import nn


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # we take q, k, v: [B, H, T, d] and return output [B, H, T, d] and weights [B, H, T, T]
    d = q.size(-1)
    T = q.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = (
            self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )  # [B, H, T, hd]
        k = (
            self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )  # [B, KV, T, hd]
        v = (
            self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )  # [B, KV, T, hd]

        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)  # [B, H, T, hd]
        v = v.repeat_interleave(group, dim=1)

        y, _ = causal_attention(q, k, v)  # [B, H, T, hd]
        y = (
            y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        )  # [B, T, C]
        return self.o_proj(y)
