"""Checker for training/data.py and training/pretrain.py (Lesson 6.2)."""

import json

import numpy as np
import pytest
import torch

import training.data as data_mod
from training.data import get_batch
from training.pretrain import TrainConfig, clip_grad_norm, train


def test_batch_targets_are_inputs_shifted_by_one():
    tokens = np.arange(100, dtype=np.uint16)
    x, y = get_batch(tokens, batch_size=4, seq_len=8, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.int64                       # embedding lookups need int64
    assert torch.equal(y[:, :-1], x[:, 1:])             # y is x moved one step ahead
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_batch_windows_stay_in_bounds():
    tokens = np.arange(20, dtype=np.uint16)
    x, y = get_batch(tokens, batch_size=1000, seq_len=10, rng=np.random.default_rng(0))
    assert x.min() >= 0 and y.max() <= 19               # never reads past the end
    assert y.max() == 19                                # ...but can reach the last token


def test_batches_are_reproducible_from_seed():
    tokens = np.arange(1000, dtype=np.uint16)
    a = get_batch(tokens, 8, 16, np.random.default_rng(42))
    b = get_batch(tokens, 8, 16, np.random.default_rng(42))
    assert torch.equal(a[0], b[0])


def test_clip_grad_norm_matches_pytorch():
    torch.manual_seed(0)
    p1, p2 = torch.nn.Parameter(torch.randn(10)), torch.nn.Parameter(torch.randn(10))
    q1, q2 = torch.nn.Parameter(p1.detach().clone()), torch.nn.Parameter(p2.detach().clone())
    for p, q in ((p1, q1), (p2, q2)):
        p.grad = torch.randn(10) * 5
        q.grad = p.grad.clone()
    ours = clip_grad_norm([p1, p2], max_norm=1.0)
    ref = torch.nn.utils.clip_grad_norm_([q1, q2], max_norm=1.0)
    assert ours == pytest.approx(ref.item(), rel=1e-5)          # returns the norm BEFORE clipping
    assert torch.allclose(p1.grad, q1.grad, atol=1e-6)


def test_clip_leaves_small_gradients_alone():
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([0.1, 0.2, 0.2])                       # norm 0.3 < 1.0
    clip_grad_norm([p], max_norm=1.0)
    assert torch.allclose(p.grad, torch.tensor([0.1, 0.2, 0.2]))


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    # a fake dataset: a repeating pattern a tiny model can learn quickly
    ds = tmp_path / "tokens" / "pattern"
    ds.mkdir(parents=True)
    pattern = np.tile(np.arange(16, dtype=np.uint16), 400)
    pattern.tofile(ds / "train.bin")
    pattern[:1000].tofile(ds / "val.bin")
    (ds / "meta.json").write_text(json.dumps({"vocab_size": 16}))
    monkeypatch.setattr(data_mod, "TOKENS", tmp_path / "tokens")
    return tmp_path


def tiny_config(out_dir) -> TrainConfig:
    return TrainConfig(dataset="pattern", out_dir=str(out_dir), d_model=32, n_layers=2, n_heads=4,
                       n_kv_heads=2, mlp_hidden=64, seq_len=16, batch_size=8, max_steps=20,
                       max_lr=1e-2, min_lr=1e-3, warmup_steps=5, eval_every=5, eval_batches=2,
                       log_every=1000, device="cpu")


def test_training_reduces_loss(tiny_dataset):
    result = train(tiny_config(tiny_dataset / "run"))
    assert result["val"] < 1.0                                   # started near ln(16) = 2.77
    assert (tiny_dataset / "run" / "best.pt").exists()
    assert (tiny_dataset / "run" / "last.pt").exists()


def test_resume_gives_identical_result(tiny_dataset):
    # run A: 20 steps straight. Run B: "crash" after step 12, resume, finish.
    # B resumes from the last checkpoint (step 10) and must end bit-for-bit where A ended.
    a = train(tiny_config(tiny_dataset / "a"))["model"]
    train(tiny_config(tiny_dataset / "b"), stop_after=12)
    b = train(tiny_config(tiny_dataset / "b"), resume=True)["model"]
    for (name, pa), pb in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.allclose(pa, pb, atol=1e-6), name


def test_gradient_accumulation_matches_bigger_batch(tiny_dataset):
    # one step with batch 8 should equal one step with 2 x batch 4 (same 8 sequences)
    cfg = tiny_config(tiny_dataset / "x")
    cfg.max_steps, cfg.eval_every = 1, 1000
    big = train(cfg)["model"]
    cfg2 = tiny_config(tiny_dataset / "y")
    cfg2.max_steps, cfg2.eval_every, cfg2.batch_size, cfg2.grad_accum_steps = 1, 1000, 4, 2
    acc = train(cfg2)["model"]
    for pa, pb in zip(big.parameters(), acc.parameters()):
        assert torch.allclose(pa, pb, atol=1e-5)
