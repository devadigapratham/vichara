"""Checker for the KV cache (Lesson 8.2): cached generation must match uncached generation exactly."""

import torch

from inference.generate import generate
from model.attention import KVCache, causal_attention
from model.transformer import GPT, GPTConfig

CFG = GPTConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_attention_with_fewer_queries_than_keys():
    # 2 new queries attending over 5 keys must equal the last 2 rows of full 5x5 causal attention
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    full, _ = causal_attention(q, k, v)
    last_two, _ = causal_attention(q[:, :, -2:], k, v)
    assert torch.allclose(last_two, full[:, :, -2:], atol=1e-6)


def test_cache_update_appends():
    cache = KVCache(batch_size=1, n_kv_heads=2, max_len=8, head_dim=4, device="cpu")
    k1, v1 = torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4)
    k, v = cache.update(k1, v1)
    assert k.shape == (1, 2, 3, 4) and cache.length == 3
    k, v = cache.update(2 * torch.ones(1, 2, 1, 4), 2 * torch.ones(1, 2, 1, 4))
    assert k.shape == (1, 2, 4, 4) and cache.length == 4
    assert torch.equal(k[:, :, 3], 2 * torch.ones(1, 2, 4))  # new entry lands after the old ones


def test_prefill_then_decode_matches_full_forward():
    # the core correctness test: logits computed token-by-token with the cache
    # must equal the logits of one full forward pass over the whole sequence
    m = model()
    ids = torch.randint(0, 32, (2, 12))
    full_logits, _ = m(ids)

    caches = m.new_caches(batch_size=2, device="cpu")
    prefill_logits, _ = m(ids[:, :5], caches=caches)                  # prefill 5 tokens at once
    step_logits = [m(ids[:, t : t + 1], caches=caches)[0] for t in range(5, 12)]   # then 1 at a time
    cached = torch.cat([prefill_logits, *step_logits], dim=1)
    assert torch.allclose(cached, full_logits, atol=1e-5)


def test_cached_generation_equals_uncached():
    m = model()
    prompt = torch.tensor([[1, 2, 3]])
    a = generate(m, prompt, 10, temperature=0, use_cache=False)
    b = generate(m, prompt, 10, temperature=0, use_cache=True)
    assert torch.equal(a, b)


def test_cached_generation_beyond_context_window():
    out = generate(model(), torch.tensor([[1]]), 40, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 41)


def test_cache_memory_formula():
    caches = model().new_caches(batch_size=3, device="cpu")
    n_bytes = sum(c.k.numel() * c.k.element_size() + c.v.numel() * c.v.element_size() for c in caches)
    # 2 (K and V) x layers x batch x kv_heads x max_len x head_dim x 4 bytes
    assert n_bytes == 2 * 2 * 3 * 2 * 16 * 8 * 4
