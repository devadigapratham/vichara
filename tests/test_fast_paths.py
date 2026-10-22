import json

import numpy as np
import pytest
import torch

import training.data as data_mod
from inference.generate import generate
from model.transformer import GPT, GPTConfig
from training.pretrain import TrainConfig, parse_args, train


def models():
    cfg = dict(vocab_size=32, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=24)
    torch.manual_seed(0)
    manual = GPT(GPTConfig(**cfg)).eval()
    fast = GPT(GPTConfig(**cfg, use_sdpa=True)).eval()
    fast.load_state_dict(manual.state_dict())
    return manual, fast


def test_sdpa_matches_manual_attention():
    manual, fast = models()
    ids = torch.randint(0, 32, (3, 20))
    assert torch.allclose(manual(ids)[0], fast(ids)[0], atol=1e-5)


def test_sdpa_gradients_match():
    manual, fast = models()
    ids = torch.randint(0, 32, (2, 12))
    manual.train(), fast.train()
    manual(ids, ids)[1].backward()
    fast(ids, ids)[1].backward()
    for (name, a), b in zip(manual.named_parameters(), fast.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-5), name


def test_sdpa_model_generates_same_tokens_with_kv_cache():
    manual, fast = models()
    prompt = torch.tensor([[1, 2, 3]])
    a = generate(manual, prompt, 15, temperature=0)
    b = generate(fast, prompt, 15, temperature=0)
    assert torch.equal(a, b)


def test_recording_attention_still_works_with_sdpa():
    _, fast = models()
    attn = fast.blocks[0].attn
    attn.record_attention = True
    fast(torch.randint(0, 32, (1, 6)))
    assert attn.last_attention.shape == (1, 4, 6, 6)


def test_head_mask_works_with_sdpa():
    _, fast = models()
    ids = torch.randint(0, 32, (1, 8))
    baseline = fast(ids)[0]
    fast.blocks[0].attn.head_mask = torch.zeros(4)
    assert not torch.allclose(fast(ids)[0], baseline)


def test_old_configs_without_sdpa_field_still_load():
    old = dict(vocab_size=32, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, mlp_hidden=64,
               max_seq_len=16, rope_base=10000.0)
    assert GPTConfig(**old).use_sdpa is False


def test_bool_flags_parse_correctly():
    cfg, _ = parse_args([])
    assert cfg.sdpa is False and cfg.compile is False and cfg.precision == "fp32"
    cfg, _ = parse_args(["--sdpa", "--precision", "bf16"])
    assert cfg.sdpa is True and cfg.precision == "bf16"
    cfg, _ = parse_args(["--no-sdpa"])
    assert cfg.sdpa is False


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    ds = tmp_path / "tokens" / "pattern"
    ds.mkdir(parents=True)
    pattern = np.tile(np.arange(16, dtype=np.uint16), 400)
    pattern.tofile(ds / "train.bin")
    pattern[:1000].tofile(ds / "val.bin")
    (ds / "meta.json").write_text(json.dumps({"vocab_size": 16}))
    monkeypatch.setattr(data_mod, "TOKENS", tmp_path / "tokens")
    return tmp_path


@pytest.mark.parametrize("precision, sdpa", [("fp32", True), ("bf16", False), ("bf16", True)])
def test_training_works_in_every_mode(tiny_dataset, precision, sdpa):
    cfg = TrainConfig(dataset="pattern", out_dir=str(tiny_dataset / f"{precision}{sdpa}"), d_model=32, n_layers=2,
                      n_heads=4, n_kv_heads=2, mlp_hidden=64, seq_len=16, batch_size=8, max_steps=20,
                      max_lr=1e-2, min_lr=1e-3, warmup_steps=5, eval_every=10, eval_batches=2, log_every=1000,
                      device="cpu", precision=precision, sdpa=sdpa)
    assert train(cfg)["val"] < 1.0
