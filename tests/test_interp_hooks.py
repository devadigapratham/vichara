"""Checker for the interpretability hooks in model/attention.py (Lesson 10)."""

import torch

from model.attention import MultiHeadAttention


def attn_and_input():
    torch.manual_seed(0)
    return MultiHeadAttention(d_model=32, n_heads=4, n_kv_heads=2).eval(), torch.randn(2, 6, 32)


def test_hooks_are_off_by_default():
    attn, x = attn_and_input()
    assert attn.head_mask is None and attn.record_attention is False
    attn(x)
    assert attn.last_attention is None


def test_all_ones_mask_changes_nothing():
    attn, x = attn_and_input()
    baseline = attn(x)
    attn.head_mask = torch.ones(4)
    assert torch.allclose(attn(x), baseline)


def test_masking_every_head_silences_the_layer():
    attn, x = attn_and_input()
    attn.head_mask = torch.zeros(4)
    assert torch.allclose(attn(x), torch.zeros(2, 6, 32))


def test_masking_one_head_removes_exactly_its_contribution():
    # o_proj is linear, so: full output = sum of each head's contribution.
    # output(without head 1) must equal output(all heads) - output(only head 1)
    attn, x = attn_and_input()
    full = attn(x)
    attn.head_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    without = attn(x)
    attn.head_mask = torch.tensor([0.0, 1.0, 0.0, 0.0])
    only = attn(x)
    assert torch.allclose(without, full - only, atol=1e-6)
    assert not torch.allclose(without, full)        # positive control: it did change something


def test_recorded_attention_is_a_causal_distribution():
    attn, x = attn_and_input()
    attn.record_attention = True
    attn(x)
    w = attn.last_attention
    assert w.shape == (2, 4, 6, 6)                  # [B, heads, query, key]
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-6)
    assert torch.all(w.triu(diagonal=1) == 0)
    assert not w.requires_grad                       # stored detached: no graph kept alive
