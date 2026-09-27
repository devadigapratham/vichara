import math
import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root, so the data is importable
from data.download import get


def build_vocab(text):
    # converting index to string and string to index
    itos = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def count_bigrams(ids, vocab_size):
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
    counts.index_put_(
        (ids[:-1], ids[1:]),
        torch.ones(len(ids) - 1, dtype=torch.float64),
        accumulate=True,
    )

    return counts


def bigram_probs(counts, alpha=0.0):
    c = counts + alpha
    return c / c.sum(dim=1, keepdim=True)


def avg_nll(probs, ids):
    p = probs[
        ids[:-1], ids[1:]
    ]  # P(actual next char | actual previous char), one per position in the text, except the first one (which has no previous char)
    return (
        (-torch.log(p)).mean().item()
    )  # NLL Formula L = -1/N * sum(log(p(x_i | x_{i-1})))


def sample(probs, start, n, seed=42):
    # creating an autoregressive loop where we lookup the row for the last token, draw the next token from that distribution, append it and repeat it.
    g = torch.Generator().manual_seed(seed)
    out = [start]
    for _ in range(n):
        out.append(torch.multinomial(probs[out[-1]].float(), 1, generator=g).item())
    return out


def main():
    text = get("shakespeare").read_text()
    stoi, itos = build_vocab(text)
    vocab_size = len(itos)
    ids = torch.tensor([stoi[ch] for ch in text])

    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    counts = count_bigrams(train_ids, vocab_size)
    print(f"chars: {len(ids):,}   vocab size: {vocab_size}\n")
    print(f"{'model':<24}{'train nll':>12}{'val nll':>12}{'val ppl':>10}")

    uniform_nll = math.log(vocab_size)
    print(f"{'uniform':<24}{uniform_nll:>12.4f}{uniform_nll:>12.4f}{vocab_size:>10.2f}")

    for alpha in (0.0, 0.01, 1.0):
        probs = bigram_probs(counts, alpha)
        train_nll = avg_nll(probs, train_ids)
        val_nll = avg_nll(probs, val_ids)
        print(
            f"{f'bigram alpha={alpha}':<24}{train_nll:>12.4f}{val_nll:>12.4f}{math.exp(val_nll):>10.2f}"
        )

    probs = bigram_probs(counts, alpha=0.01)
    generated = sample(probs, start=stoi["\n"], n=300)
    print("\nsample:\n" + "".join(itos[i] for i in generated))


if __name__ == "__main__":
    main()
