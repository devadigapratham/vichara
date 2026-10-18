"""Turn raw text into token-id files for training: <split>.bin (uint16) + meta.json."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `data` is importable
from data.download import get  # noqa: E402

TOKENS = Path(__file__).parent / "tokens"


def write_tokens(out_dir: Path, split: str, ids) -> None:
    np.asarray(ids, dtype=np.uint16).tofile(out_dir / f"{split}.bin")


def prepare_shakespeare_char() -> Path:
    text = get("shakespeare").read_text()
    itos = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(itos)}
    ids = [stoi[ch] for ch in text]
    split = int(0.9 * len(ids))

    out = TOKENS / "shakespeare_char"
    out.mkdir(parents=True, exist_ok=True)
    write_tokens(out, "train", ids[:split])
    write_tokens(out, "val", ids[split:])
    (out / "meta.json").write_text(json.dumps({"vocab_size": len(itos), "itos": itos}))
    print(f"wrote {len(ids):,} tokens to {out}")
    return out


if __name__ == "__main__":
    prepare_shakespeare_char()
