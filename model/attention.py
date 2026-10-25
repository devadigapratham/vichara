# implementing causal self-attention which uses single head math, multi-head module and grouped query attention
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional import apply_rope, rope_cache


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    # q: [B, H, Tq, d]; k, v: [B, H, Tk, d] with Tk >= Tq (the queries are the LAST Tq positions)
    d = q.size(-1)
    Tq, Tk = q.size(-2), k.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    future = torch.triu(
        torch.ones(Tq, Tk, dtype=torch.bool, device=q.device), diagonal=Tk - Tq + 1
    )
    if key_mask is not None:  # [B, Tk], True = real token
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
    scores = scores.masked_fill(future, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


class KVCache:
    def __init__(
        self, batch_size, n_kv_heads, max_len, head_dim, device, dtype=torch.float32
    ):
        shape = (batch_size, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def update(self, k_new, v_new):
        start, end = self.length, self.length + k_new.size(2)
        self.k[:, :, start:end] = k_new
        self.v[:, :, start:end] = v_new
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_seq_len: int = 1024,
        rope_base: float = 10000.0,
        use_sdpa: bool = False,
        qkv_bias: bool = False,
    ):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_sdpa = use_sdpa
        cos, sin = rope_cache(max_seq_len, self.head_dim, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        # interpretability hooks (plain attributes: not parameters, not saved in checkpoints)
        self.head_mask: torch.Tensor | None = None  # [n_heads]; 0 switches a head off
        self.record_attention = False
        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = (
            self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )  # [B, H, T, hd]
        k = (
            self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )  # [B, KV, T, hd]
        v = (
            self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )  # [B, KV, T, hd]
        if position_ids is None:
            start = cache.length if cache is not None else 0
            cos, sin = self.rope_cos[start : start + T], self.rope_sin[start : start + T]
        else:
            cos, sin = self.rope_cos[position_ids][:, None], self.rope_sin[position_ids][:, None]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)

        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)  # [B, H, T, hd]
        v = v.repeat_interleave(group, dim=1)

        if self.use_sdpa and not self.record_attention and key_mask is None and k.size(2) == T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y, weights = causal_attention(q, k, v, key_mask)
            if self.record_attention:
                self.last_attention = weights.detach()

        if self.head_mask is not None:
            y = y * self.head_mask.view(1, -1, 1, 1).to(y.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)
