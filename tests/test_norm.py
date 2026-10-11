"""Checker for model/norm.py. The first test replays the worked example from Lesson 5."""

import torch
import torch.nn as nn

from model.norm import RMSNorm


def test_worked_example():
    # x = [3, 4]: mean of squares = 12.5, rms = 3.5355 -> [0.8485, 1.1314]
    out = RMSNorm(2)(torch.tensor([3.0, 4.0]))
    assert torch.allclose(out, torch.tensor([0.8485, 1.1314]), atol=1e-4)


def test_output_has_unit_rms():
    torch.manual_seed(0)
    out = RMSNorm(64)(torch.randn(4, 10, 64) * 7.0)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_scale_invariance():
    # multiplying the input by any positive number does not change the output
    torch.manual_seed(0)
    norm, x = RMSNorm(16), torch.randn(3, 16)
    assert torch.allclose(norm(x), norm(100.0 * x), atol=1e-5)


def test_zero_input_is_finite():
    assert torch.isfinite(RMSNorm(8)(torch.zeros(2, 8))).all()


def test_weight_is_learnable_parameter():
    norm = RMSNorm(16)
    params = dict(norm.named_parameters())
    assert list(params) == ["weight"] and params["weight"].shape == (16,)


def test_matches_pytorch_rmsnorm():
    torch.manual_seed(0)
    ours, ref = RMSNorm(32, eps=1e-6), nn.RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(32))
        ref.weight.copy_(ours.weight)
    x = torch.randn(2, 5, 32)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_keeps_input_dtype():
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    assert RMSNorm(16)(x).dtype == torch.bfloat16
