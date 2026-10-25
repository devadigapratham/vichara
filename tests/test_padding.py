import torch

from inference.generate import generate, generate_batch
from model.transformer import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=40, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                         max_seq_len=64)).eval()


def test_left_padding_does_not_change_logits():
    model = tiny()
    seq = torch.tensor([[5, 6, 7, 8]])
    padded = torch.tensor([[0, 0, 0, 5, 6, 7, 8]])
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    a = model(seq)[0]
    b = model(padded, attention_mask=mask)[0][:, 3:]
    assert torch.allclose(a, b, atol=1e-5)


def test_all_ones_mask_matches_no_mask():
    model = tiny()
    ids = torch.randint(0, 40, (2, 9))
    assert torch.allclose(model(ids)[0], model(ids, attention_mask=torch.ones(2, 9, dtype=torch.long))[0], atol=1e-6)


def test_padded_rows_produce_no_nan():
    model = tiny()
    ids = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert torch.isfinite(model(ids, attention_mask=mask)[0]).all()


def test_batched_generation_matches_one_at_a_time():
    model = tiny()
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9], [10], [11, 12, 13, 14]]
    batched = generate_batch(model, prompts, 12, eos_id=39)
    for p, out in zip(prompts, batched):
        single = generate(model, torch.tensor([p]), 12, temperature=0)[0, len(p):].tolist()
        single = single[: single.index(39) + 1] if 39 in single else single
        assert out == single


def test_generation_stops_at_eos():
    model = tiny()
    eos = generate(model, torch.tensor([[1, 2, 3]]), 1, temperature=0)[0, -1].item()   # whatever comes first
    out = generate_batch(model, [[1, 2, 3], [7, 8]], 10, eos_id=eos)
    assert out[0] == [eos]
    assert eos not in out[1][:-1]


def test_bf16_model_generates_with_matching_cache_dtype():
    model = tiny().to(torch.bfloat16)
    assert model.new_caches(1, "cpu")[0].k.dtype == torch.bfloat16
    out = generate_batch(model, [[1, 2, 3], [4, 5]], 6, eos_id=39)
    assert len(out) == 2


def test_last_only_matches_last_position_of_full_logits():
    model = tiny()
    ids = torch.randint(0, 40, (2, 7))
    full = model(ids)[0]
    last = model(ids, last_only=True)[0]
    assert last.shape == (2, 1, 40)
    assert torch.allclose(last, full[:, -1:], atol=1e-6)
