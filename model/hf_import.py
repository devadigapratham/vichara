import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from model.transformer import GPT, GPTConfig


def qwen2_config(path, max_seq_len: int = 2048) -> GPTConfig:
    c = json.loads((Path(path) / "config.json").read_text())
    return GPTConfig(vocab_size=c["vocab_size"], d_model=c["hidden_size"], n_layers=c["num_hidden_layers"],
                     n_heads=c["num_attention_heads"], n_kv_heads=c["num_key_value_heads"],
                     mlp_hidden=c["intermediate_size"], max_seq_len=max_seq_len, rope_base=c["rope_theta"],
                     qkv_bias=True)


def qwen2_key_map(n_layers: int) -> dict[str, str]:
    keys = {"model.embed_tokens.weight": "tok_emb.weight", "model.norm.weight": "norm.weight"}
    for i in range(n_layers):
        hf, ours = f"model.layers.{i}.", f"blocks.{i}."
        keys[hf + "input_layernorm.weight"] = ours + "attn_norm.weight"
        keys[hf + "post_attention_layernorm.weight"] = ours + "mlp_norm.weight"
        for name in ("q_proj", "k_proj", "v_proj"):
            keys[hf + f"self_attn.{name}.weight"] = ours + f"attn.{name}.weight"
            keys[hf + f"self_attn.{name}.bias"] = ours + f"attn.{name}.bias"
        keys[hf + "self_attn.o_proj.weight"] = ours + "attn.o_proj.weight"
        for name in ("gate_proj", "up_proj", "down_proj"):
            keys[hf + f"mlp.{name}.weight"] = ours + f"mlp.{name}.weight"
    return keys


def load_qwen2(path, max_seq_len: int = 2048, dtype=torch.float32) -> GPT:
    cfg = qwen2_config(path, max_seq_len)
    model = GPT(cfg)
    hf_state = load_file(str(Path(path) / "model.safetensors"))
    keys = qwen2_key_map(cfg.n_layers)
    unknown = [k for k in hf_state if k not in keys]
    if unknown:
        raise ValueError(f"unexpected keys in checkpoint: {unknown[:5]}")
    state = {keys[k]: v.to(dtype) for k, v in hf_state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    if [m for m in missing if m != "lm_head.weight"]:
        raise ValueError(f"missing weights: {missing[:5]}")
    model.lm_head.weight = model.tok_emb.weight
    return model.to(dtype).eval()
