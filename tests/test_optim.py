"""Checker for training/optim.py (Lesson 6.1). The first tests replay the worked examples."""

import math

import pytest
import torch
import torch.nn as nn

from model.transformer import GPT, GPTConfig
from training.optim import AdamW, lr_at, param_groups


def test_first_step_is_lr_times_sign():
    # Lesson 6: after bias correction, step 1 moves every parameter by exactly lr,
    # whether its gradient is tiny (rare) or large (common)
    p = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    p.grad = torch.tensor([0.01, 1.0, -5.0])
    AdamW([p], lr=0.1, weight_decay=0.0).step()
    assert torch.allclose(p.data, torch.tensor([0.9, 0.9, 1.1]), atol=1e-6)


def test_decoupled_weight_decay_shrinks_weights():
    # zero gradient: only the decay acts, p <- p * (1 - lr * wd)
    p = nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([0.0])
    AdamW([p], lr=0.1, weight_decay=0.1).step()
    assert torch.allclose(p.data, torch.tensor([0.99]))


def test_matches_pytorch_adamw():
    torch.manual_seed(0)
    init = torch.randn(4, 3)
    ours, ref = nn.Parameter(init.clone()), nn.Parameter(init.clone())
    opt_ours = AdamW([ours], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opt_ref = torch.optim.AdamW([ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(10):
        g = torch.randn(4, 3)
        ours.grad, ref.grad = g.clone(), g.clone()
        opt_ours.step()
        opt_ref.step()
    assert torch.allclose(ours.data, ref.data, atol=1e-6)


def test_param_groups_decay_only_matrices():
    model = GPT(GPTConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64))
    decay, no_decay = param_groups(model, weight_decay=0.1)
    assert all(p.dim() >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() == 1 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0
    n_grouped = sum(p.numel() for g in (decay, no_decay) for p in g["params"])
    assert n_grouped == model.num_params()  # every parameter in exactly one group (tied weight once)


@pytest.mark.parametrize(("step", "expected"), [
    (0, 1e-5),        # warmup starts at max_lr / warmup_steps, not 0
    (49, 5e-4),       # halfway through warmup
    (99, 1e-3),       # end of warmup = max_lr
    (100, 1e-3),      # cosine starts at the top
    (550, 5.5e-4),    # halfway through decay: midpoint between max and min
    (1000, 1e-4),     # end: min_lr
    (5000, 1e-4),     # past the end: stays at min_lr
])
def test_schedule_worked_example(step, expected):
    lr = lr_at(step, max_lr=1e-3, min_lr=1e-4, warmup_steps=100, total_steps=1000)
    assert math.isclose(lr, expected, rel_tol=1e-6)
