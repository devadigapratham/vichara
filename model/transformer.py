# full gpt model : using its config, transformer black and the model
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import KVCache, MultiHeadAttention
from model.mlp import SwiGLU
from model.norm import RMSNorm


@dataclass
class GPTConfig:
    vocab_size: int = 4096  # we'll be using tinystory tokenizer for simplicity
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    mlp_hidden: int = 1024
    max_seq_len: int = 512
    rope_base: float = 10000.0
    use_sdpa: bool = False
    qkv_bias: bool = False


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.max_seq_len,
            cfg.rope_base,
            cfg.use_sdpa,
            cfg.qkv_bias,
        )
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None,
                position_ids: torch.Tensor | None = None, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x), cache, position_ids, key_mask
        )  # gather information from other positions
        x = x + self.mlp(self.mlp_norm(x))  # process it, per position
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying: one matrix, two uses

        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(
            2 * cfg.n_layers
        )  # the layers that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
        last_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        position_ids = key_mask = None
        if attention_mask is not None:  # [B, all keys so far], 1 = real token, 0 = left padding
            key_mask = attention_mask.bool()
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)[:, -ids.size(1):]
        x = self.tok_emb(ids)  # [B, T] → [B, T, C]
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i], position_ids, key_mask)
        if last_only:  # generation only needs the next-token logits; [B, T, V] is huge for big vocabs
            x = x[:, -1:]
        logits = self.lm_head(self.norm(x))  # [B, T, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def new_caches(self, batch_size: int, device) -> list[KVCache]:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        return [
            KVCache(
                batch_size, self.cfg.n_kv_heads, self.cfg.max_seq_len, head_dim, device,
                dtype=self.tok_emb.weight.dtype,
            )
            for _ in self.blocks
        ]
