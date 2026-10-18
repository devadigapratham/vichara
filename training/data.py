"""Load token files and sample training batches."""

import json
from pathlib import Path

import numpy as np
import torch

TOKENS = Path(__file__).resolve().parents[1] / "data" / "tokens"


def load_split(dataset: str, split: str) -> np.ndarray:
    return np.memmap(TOKENS / dataset / f"{split}.bin", dtype=np.uint16, mode="r")


def load_meta(dataset: str) -> dict:
    return json.loads((TOKENS / dataset / "meta.json").read_text())


def get_batch(
    tokens: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator
):
    starts = rng.integers(0, len(tokens) - seq_len, size=batch_size)
    windows = np.stack([tokens[s : s + seq_len + 1] for s in starts]).astype(np.int64)
    windows = torch.from_numpy(windows)
    return windows[:, :-1], windows[:, 1:]
