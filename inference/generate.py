# text generation using sampling strategies and the autoregressive loop

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # must run BEFORE the imports below
from model.transformer import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer
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
def generate(model, ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None, use_cache=True):
    model.eval()
    max_len = model.cfg.max_seq_len
    if not use_cache:
        for _ in range(max_new_tokens):
            logits, _ = model(ids[:, -max_len:])
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    caches = model.new_caches(ids.size(0), ids.device)
    logits, _ = model(ids[:, -max_len:], caches=caches)
    for i in range(max_new_tokens):
        next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        ids = torch.cat([ids, next_id], dim=1)
        if i == max_new_tokens - 1:
            break
        if caches[0].length == max_len:
            caches = model.new_caches(ids.size(0), ids.device)
            logits, _ = model(ids[:, -(max_len // 2):], caches=caches)
        else:
            logits, _ = model(next_id, caches=caches)
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


REPO = Path(__file__).resolve().parents[1]


def get_codec(meta: dict):
    """Return (encode, decode) for a dataset: character-level, or our BPE tokenizer."""
    if "itos" in meta:
        itos = meta["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return (lambda s: [stoi[ch] for ch in s]), (lambda ids: "".join(itos[i] for i in ids))
    tok = BPETokenizer.load(REPO / meta["tokenizer"])
    return tok.encode_ordinary, tok.decode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/shakespeare_char/best.pt")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, ckpt = load_model(args.ckpt)
    meta = load_meta(ckpt["train_config"]["dataset"])
    encode, decode = get_codec(meta)
    if args.prompt is None:                          # no prompt: start like a fresh document
        start = [meta["eot_id"]] if "eot_id" in meta else encode("\n")
    else:
        start = encode(args.prompt)
    ids = torch.tensor([start])
    g = torch.Generator().manual_seed(args.seed)
    out = generate(
        model, ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p, g
    )
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()


@torch.no_grad()
def generate_batch(model, prompts: list[list[int]], max_new_tokens: int, eos_id: int, pad_id: int = 0,
                   temperature: float = 0.0, top_p: float | None = None, generator=None,
                   n_valid: int | None = None) -> list[list[int]]:
    model.eval()
    device = next(model.parameters()).device
    batch, width = len(prompts), max(len(p) for p in prompts)
    assert width + max_new_tokens <= model.cfg.max_seq_len, "prompt + new tokens exceed max_seq_len"
    ids = torch.full((batch, width), pad_id, dtype=torch.long)
    mask = torch.zeros(batch, width, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, width - len(p):] = torch.tensor(p)
        mask[i, width - len(p):] = 1
    ids, mask = ids.to(device), mask.to(device)

    caches = model.new_caches(batch, device)
    logits, _ = model(ids, caches=caches, attention_mask=mask, last_only=True)
    out = []
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        nxt = sample_next(logits[:, -1, :n_valid].float(), temperature, top_p=top_p, generator=generator)
        nxt = nxt.masked_fill(done[:, None], pad_id)
        out.append(nxt)
        mask = torch.cat([mask, (~done).long()[:, None]], dim=1)
        done = done | (nxt[:, 0] == eos_id)
        if done.all() or step == max_new_tokens - 1:
            break
        logits, _ = model(nxt, caches=caches, attention_mask=mask)
    out = torch.cat(out, dim=1).tolist()
    results = []
    for row in out:
        results.append(row[: row.index(eos_id) + 1] if eos_id in row else row)
    return results
