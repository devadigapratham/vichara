import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from model.hf_import import load_qwen2, qwen2_key_map
from model.transformer import GPT, GPTConfig

QWEN = Path("checkpoints/qwen2.5-0.5b")


def test_qkv_bias_is_off_by_default():
    model = GPT(GPTConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, mlp_hidden=64))
    assert model.blocks[0].attn.q_proj.bias is None


def test_qkv_bias_adds_exactly_the_bias_parameters():
    cfg = dict(vocab_size=32, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64)
    plain, biased = GPT(GPTConfig(**cfg)), GPT(GPTConfig(**cfg, qkv_bias=True))
    assert biased.num_params() - plain.num_params() == 2 * (32 + 16 + 16)    # q + k + v biases per layer


def fake_qwen_dir(tmp_path):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=48,
                    rope_base=1_000_000.0, qkv_bias=True)
    src = GPT(cfg).eval()
    for b in src.blocks:
        for name in ("q_proj", "k_proj", "v_proj"):
            torch.nn.init.normal_(getattr(b.attn, name).bias)
    ours_to_hf = {v: k for k, v in qwen2_key_map(cfg.n_layers).items()}
    state = {ours_to_hf[k]: v.clone() for k, v in src.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "vocab_size": 50, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 48, "rope_theta": 1_000_000.0}))
    return src


def test_load_roundtrip_gives_identical_logits(tmp_path):
    src = fake_qwen_dir(tmp_path)
    loaded = load_qwen2(tmp_path, max_seq_len=1024)
    ids = torch.randint(0, 50, (2, 9))
    assert torch.allclose(loaded(ids)[0], src(ids)[0], atol=1e-6)
    assert loaded.lm_head.weight is loaded.tok_emb.weight


def test_unknown_keys_are_rejected(tmp_path):
    fake_qwen_dir(tmp_path)
    from safetensors.torch import load_file
    state = load_file(str(tmp_path / "model.safetensors"))
    state["model.layers.0.self_attn.q_norm.weight"] = torch.ones(8)
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        load_qwen2(tmp_path)


@pytest.mark.skipif(not (QWEN / "model.safetensors").exists(), reason="Qwen weights not downloaded")
def test_matches_huggingface_qwen():
    transformers = pytest.importorskip("transformers")
    ours = load_qwen2(QWEN)
    ref = transformers.AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval()
    ids = torch.tensor([[9707, 11, 1879, 0, 3555, 374, 220, 17, 10, 17, 30]])
    with torch.no_grad():
        a, b = ours(ids)[0], ref(ids).logits
    assert (a - b).abs().max() < 1e-2
    assert torch.equal(a.argmax(-1), b.argmax(-1))
