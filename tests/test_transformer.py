"""Checker for model/mlp.py and model/transformer.py (Lesson 5.3)."""

import math

import pytest
import torch
import torch.nn.functional as F

from model.mlp import SwiGLU
from model.transformer import GPT, Block, GPTConfig

TINY = GPTConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_silu_worked_example():
    # Lesson 5.3: SiLU(z) = z * sigmoid(z)
    out = F.silu(torch.tensor([2.0, -2.0, 0.0]))
    assert torch.allclose(out, torch.tensor([1.7616, -0.2384, 0.0]), atol=1e-4)


def test_swiglu_shape_and_params():
    mlp = SwiGLU(d_model=32, hidden=64)
    assert mlp(torch.randn(2, 5, 32)).shape == (2, 5, 32)
    assert sum(p.numel() for p in mlp.parameters()) == 3 * 32 * 64


def test_block_preserves_shape():
    assert Block(TINY)(torch.randn(2, 10, 32)).shape == (2, 10, 32)


def test_forward_shapes_and_loss():
    model = GPT(TINY)
    ids = torch.randint(0, 64, (2, 10))
    logits, loss = model(ids, targets=ids)
    assert logits.shape == (2, 10, 64)
    assert loss.shape == ()
    assert model(ids)[1] is None  # no targets -> no loss


def test_initial_loss_is_near_uniform():
    # Lesson 1 sanity check: an untrained model should be close to ln(vocab_size)
    # (random targets: with tied embeddings an untrained model slightly favours
    # repeating the current token, so targets=ids would give a misleadingly low loss)
    torch.manual_seed(0)
    model = GPT(GPTConfig())
    ids = torch.randint(0, 4096, (4, 64))
    targets = torch.randint(0, 4096, (4, 64))
    _, loss = model(ids, targets=targets)
    assert abs(loss.item() - math.log(4096)) < 0.2


def test_embeddings_are_tied():
    model = GPT(TINY)
    assert model.lm_head.weight is model.tok_emb.weight


def hand_count(cfg: GPTConfig) -> int:
    d, hd = cfg.d_model, cfg.d_model // cfg.n_heads
    embedding = cfg.vocab_size * d                      # shared with lm_head (tied), counted once
    attention = 2 * d * d + 2 * d * (cfg.n_kv_heads * hd)  # q, o full size; k, v shrunk by GQA
    mlp = 3 * d * cfg.mlp_hidden                       # gate, up, down
    norms = 2 * d                                      # attn_norm + mlp_norm
    return embedding + cfg.n_layers * (attention + mlp + norms) + d  # + final norm


@pytest.mark.parametrize("cfg", [TINY, GPTConfig()])
def test_param_count_matches_hand_formula(cfg):
    assert GPT(cfg).num_params() == hand_count(cfg)


def test_no_future_leakage_through_whole_model():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    ids = torch.randint(0, 64, (1, 12))
    changed = ids.clone()
    changed[0, 8] = (changed[0, 8] + 1) % 64
    a, b = model(ids)[0], model(changed)[0]
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8], b[:, 8])  # positive control


def test_every_parameter_gets_a_gradient():
    model = GPT(TINY)
    ids = torch.randint(0, 64, (2, 10))
    model(ids, targets=ids)[1].backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert dead == []


def test_can_overfit_a_single_batch():
    # the classic first check of a new model: it must be able to memorize one tiny batch
    torch.manual_seed(0)
    model = GPT(TINY)
    ids = torch.randint(0, 64, (2, 12))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(100):
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
