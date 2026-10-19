# text generation using sampling strategies and the autoregressive loop

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # must run BEFORE the imports below
from model.transformer import GPT, GPTConfig
from training.data import load_meta


def filter_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    logits = logits / temperature

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        mass_before = (
            probs.cumsum(-1) - probs
        )  # mass of the tokens ranked ABOVE this one
        sorted_logits = sorted_logits.masked_fill(mass_before > top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(
            -1, order, sorted_logits
        )

    return logits


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)  # greedy
    probs = filter_logits(logits, temperature, top_k, top_p).softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate(
    model: GPT,
    ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        context = ids[:, -model.cfg.max_seq_len :]  # crop to the context window
        logits, _ = model(context)  # [B, T, V]
        next_id = sample_next(
            logits[:, -1, :], temperature, top_k, top_p, generator
        )  # only the LAST position
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def load_model(path: str, device: str = "cpu") -> tuple[GPT, dict]:
    ckpt = torch.load(
        path, map_location=device, weights_only=False
    )  # our own checkpoint only
    model = GPT(GPTConfig(**ckpt["model_config"])).to(
        device
    )  # rebuild the exact architecture
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/shakespeare_char/best.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, ckpt = load_model(args.ckpt)
    itos = load_meta(ckpt["train_config"]["dataset"])["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}
    ids = torch.tensor([[stoi[ch] for ch in args.prompt]])
    g = torch.Generator().manual_seed(args.seed)
    out = generate(
        model, ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p, g
    )
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
