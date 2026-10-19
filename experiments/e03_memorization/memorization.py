# Experiment 3: does the overfit checkpoint copy its training data verbatim?

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.generate import generate, load_model
from training.data import load_meta, load_split

SPAN = 50  # characters; a 50-character exact match is essentially impossible by chance


def copied_fraction(
    text: str, reference: str, span: int = SPAN, stride: int = 10
) -> float:
    """Fraction of `span`-character windows of `text` that appear verbatim in `reference`."""
    windows = [text[i : i + span] for i in range(0, len(text) - span, stride)]
    return sum(w in reference for w in windows) / len(windows)


def main() -> None:
    itos = load_meta("shakespeare_char")["itos"]
    train_text = "".join(itos[i] for i in load_split("shakespeare_char", "train"))
    val_text = "".join(itos[i] for i in load_split("shakespeare_char", "val"))

    # control: real Shakespeare the model never trained on. How much of IT is "found" in train?
    print(
        f"control (real val text): {copied_fraction(val_text[:5000], train_text):.1%}"
    )

    for name in ("best", "last"):
        model, _ = load_model(f"checkpoints/shakespeare_char/{name}.pt")
        fractions = []
        t0 = time.perf_counter()
        for seed in range(5):
            out = generate(
                model,
                torch.tensor([[0]]),
                1000,
                temperature=0.8,
                generator=torch.Generator().manual_seed(seed),
            )
            sample = "".join(itos[i] for i in out[0].tolist())
            fractions.append(copied_fraction(sample, train_text))
        mean = sum(fractions) / len(fractions)
        print(
            f"{name}.pt: {[f'{f:.0%}' for f in fractions]}  mean {mean:.1%}  "
            f"({time.perf_counter() - t0:.0f}s for 5 x 1000 tokens)"
        )


if __name__ == "__main__":
    main()
