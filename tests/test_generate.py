"""Checker for inference/generate.py (Lesson 8). The first tests replay the worked examples."""

import torch

from inference.generate import filter_logits, generate, sample_next
from model.transformer import GPT, GPTConfig

LOGITS = torch.tensor([2.0, 1.0, 0.0])


def probs(logits):
    return logits.softmax(dim=-1)


def test_temperature_worked_example():
    assert torch.allclose(probs(filter_logits(LOGITS, temperature=1.0)), torch.tensor([0.665, 0.245, 0.090]), atol=1e-3)
    assert torch.allclose(probs(filter_logits(LOGITS, temperature=0.5)), torch.tensor([0.867, 0.117, 0.016]), atol=1e-3)
    assert torch.allclose(probs(filter_logits(LOGITS, temperature=2.0)), torch.tensor([0.506, 0.307, 0.186]), atol=1e-3)


def test_top_k_keeps_k_largest():
    p = probs(filter_logits(torch.tensor([3.0, 1.0, 2.0, 0.0]), top_k=2))
    assert (p > 0).tolist() == [True, False, True, False]
    assert torch.isclose(p.sum(), torch.tensor(1.0))


def test_top_p_worked_example():
    # probs [0.665, 0.245, 0.090]: 0.665 + 0.245 = 0.910 >= 0.9, so token 2 is dropped
    p = probs(filter_logits(LOGITS, top_p=0.9))
    assert torch.allclose(p, torch.tensor([0.731, 0.269, 0.0]), atol=1e-3)


def test_top_p_always_keeps_the_top_token():
    p = probs(filter_logits(torch.tensor([5.0, 0.0, 0.0]), top_p=0.01))
    assert p[0] == 1.0


def test_filters_work_on_batches():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    p = probs(filter_logits(logits, top_p=0.9))
    assert torch.allclose(p[1], torch.tensor([0.0, 0.269, 0.731]), atol=1e-3)


def test_temperature_zero_is_greedy():
    assert sample_next(torch.tensor([[0.1, 3.0, 0.2]]), temperature=0).item() == 1


def tiny_model(max_seq_len=8):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                         max_seq_len=max_seq_len))


def test_generate_appends_tokens_and_keeps_prompt():
    prompt = torch.tensor([[1, 2, 3]])
    out = generate(tiny_model(), prompt, max_new_tokens=5, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], prompt)


def test_generate_beyond_context_window():
    # 20 new tokens with max_seq_len 8: the loop must crop the context, not crash
    out = generate(tiny_model(max_seq_len=8), torch.tensor([[1]]), max_new_tokens=20,
                   generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 21)


def test_generation_is_reproducible_with_seed():
    model = tiny_model()
    a = generate(model, torch.tensor([[1]]), 10, generator=torch.Generator().manual_seed(7))
    b = generate(model, torch.tensor([[1]]), 10, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
