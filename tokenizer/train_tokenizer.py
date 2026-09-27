"""Train the BPE tokenizer on TinyStories and analyze what it learned."""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # must come before the imports below
from data.download import (
    get,
)
from tokenizer.bpe import BPETokenizer

EOT = "<|endoftext|>"


def show(tok, BPETokenizer, i: int) -> str:
    return repr(tok.vocab[i].decode("utf-8", errors="replace"))


def analyze(tok: BPETokenizer, text: str) -> None:
    ids = tok.encode(text, allow_special=True)
    n_bytes = len(text.encode("utf-8"))
    print(
        f"\ncompression: {n_bytes} bytes -> {len(ids):,} tokens = {n_bytes / len(ids):.2f} bytes/token"
    )
    print("\nfirst 20 merges:", " ".join(show(tok, i) for i in range(256, 276)))
    longest = sorted(
        tok.merges.values(), key=lambda i: len(tok.vocab[i]), reverse=True
    )[:10]
    print("\nlongest merges:", " ".join(show(tok, i) for i in longest))

    freq = Counter(ids)
    print(
        "most frequent:  ",
        " ".join(f"{show(tok, i)}:{c}" for i, c in freq.most_common(10)),
    )
    unused = [i for i in tok.vocab if i not in freq]
    print(f"vocab entries never used in this text: {len(unused)} of {len(tok.vocab)}")

    print()
    for s in [
        "Once upon a time",
        "12345",
        "🙂",
        "Supercalifragilistic",
        "Bonjour mon ami",
        "The END.<|endoftext|>",
    ]:
        pieces = [show(tok, i) for i in tok.encode(s, allow_special=True)]
        print(f"{s!r:28} -> {len(pieces):2} tokens: {' '.join(pieces)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--train-chars", type=int, default=20_000_000)
    parser.add_argument("--out", default="tokenizer/models/tinystories_4096.json")
    args = parser.parse_args()

    with get("tinystories_train").open(encoding="utf-8") as f:
        train_text = f.read(
            args.train_chars
        )  # a sample is enough: word statistics saturate quickly

    t0 = time.perf_counter()
    tok = BPETokenizer.train(train_text, args.vocab_size, special_tokens=[EOT])
    print(
        f"trained {len(tok.merges)} merges on {len(train_text):,} chars in {time.perf_counter() - t0:.1f}s"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)
    print(f"saved -> {out}")

    val_text = get("tinystories_valid").read_text(encoding="utf-8")[
        :5_000_000
    ]  # held-out text
    analyze(tok, val_text)


if __name__ == "__main__":
    main()
