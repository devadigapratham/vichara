# full gpt model : using its config, transformer black and the model
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
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


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len, cfg.rope_base
        )
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))  # gather information from other positions
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
        self, ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_emb(ids)  # [B, T] → [B, T, C]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))  # [B, T, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
