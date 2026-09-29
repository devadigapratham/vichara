"""Checker for model/attention.py. The first test replays the worked example from Lesson 4."""

import pytest
import torch
import torch.nn.functional as F

from model.attention import MultiHeadAttention, causal_attention


def lesson_qkv():
    # the / cat / sat, head size 2, shaped [batch=1, heads=1, T=3, d=2]
    q = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    k = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    return q[None, None], k[None, None], v[None, None]


def test_worked_example():
    out, weights = causal_attention(*lesson_qkv())
    expected = torch.tensor([[1.000, 0.000, 0.000],
                             [0.330, 0.670, 0.000],
                             [0.140, 0.576, 0.284]])
    assert torch.allclose(weights[0, 0], expected, atol=1e-3)
    assert torch.allclose(out[0, 0, 2], torch.tensor([0.282, 0.718]), atol=1e-3)


def test_weights_are_causal_distributions():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    _, weights = causal_attention(q, k, v)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 6))
    assert torch.all(weights.triu(diagonal=1) == 0)  # no weight on the future


def test_matches_pytorch_sdpa():
    # our simple implementation vs. PyTorch's optimized one (FlashAttention-style kernels on GPU)
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    ours, _ = causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])  # MHA, GQA, MQA
def test_module_shape(n_kv_heads):
    attn = MultiHeadAttention(d_model=32, n_heads=4, n_kv_heads=n_kv_heads)
    assert attn(torch.randn(2, 10, 32)).shape == (2, 10, 32)


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_no_future_leakage(n_kv_heads):
    # changing token t must not change any output at positions < t
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=32, n_heads=4, n_kv_heads=n_kv_heads)
    x = torch.randn(1, 10, 32)
    x_changed = x.clone()
    x_changed[:, 6] += 10.0
    y, y_changed = attn(x), attn(x_changed)
    assert torch.allclose(y[:, :6], y_changed[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6], y_changed[:, 6])


def test_gqa_shrinks_kv_projections():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    gqa = MultiHeadAttention(d_model=32, n_heads=4, n_kv_heads=1)
    assert mha.k_proj.weight.shape == (32, 32)
    assert gqa.k_proj.weight.shape == (8, 32)  # one shared K head of size 8


def test_attention_ignores_order_of_the_past():
    # Lesson 4 Q2: without positional information, shuffling earlier tokens
    # does not change the last position's output. (Lesson 5 adds RoPE and flips this.)
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=16, n_heads=2)
    x = torch.randn(1, 5, 16)
    shuffled = x[:, [2, 0, 3, 1, 4]]  # shuffle positions 0-3, keep the last in place
    assert torch.allclose(attn(x)[:, -1], attn(shuffled)[:, -1], atol=1e-6)
