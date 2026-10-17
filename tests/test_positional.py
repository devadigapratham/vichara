"""Checker for model/positional.py (RoPE). The first test replays the worked example from Lesson 5."""

import math

import torch

from model.positional import apply_rope, rope_cache, rotate_half


def test_worked_example_2d():
    # head_dim=2 has one pair with frequency 1, so position m rotates by m radians
    cos, sin = rope_cache(seq_len=4, head_dim=2)
    q = torch.tensor([1.0, 0.0])
    rotated = apply_rope(q, cos[1], sin[1])
    assert torch.allclose(rotated, torch.tensor([math.cos(1.0), math.sin(1.0)]), atol=1e-6)


def test_rotate_half():
    assert torch.equal(rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_position_zero_is_identity():
    cos, sin = rope_cache(seq_len=8, head_dim=16)
    x = torch.randn(16)
    assert torch.allclose(apply_rope(x, cos[0], sin[0]), x)


def test_rotation_preserves_length():
    torch.manual_seed(0)
    cos, sin = rope_cache(seq_len=32, head_dim=16)
    x = torch.randn(32, 16)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_scores_depend_only_on_relative_position():
    # the key property: q at m, k at n -> the score depends only on m - n
    torch.manual_seed(0)
    cos, sin = rope_cache(seq_len=64, head_dim=16)
    q, k = torch.randn(16), torch.randn(16)

    def score(m, n):
        return apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n])

    assert math.isclose(score(5, 2), score(40, 37), rel_tol=1e-4)   # both distance 3
    assert not math.isclose(score(5, 2), score(5, 4), rel_tol=1e-2)  # different distance


def test_shapes_broadcast_over_batch_and_heads():
    cos, sin = rope_cache(seq_len=10, head_dim=8)
    x = torch.randn(2, 4, 10, 8)  # [B, H, T, hd]
    assert apply_rope(x, cos, sin).shape == x.shape
