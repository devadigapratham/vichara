"""Turn raw text into token-id files for training: <split>.bin (uint16) + meta.json."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `data` is importable
from data.download import get  # noqa: E402
from tokenizer.bpe import BPETokenizer  # noqa: E402

TOKENS = Path(__file__).parent / "tokens"
TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "models" / "tinystories_4096.json"
EOT = "<|endoftext|>"


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


def encode_file(src: Path, dst: Path, tok: BPETokenizer, eot_id: int, flush_every: int = 10_000_000) -> int:
    buffer: list[int] = []
    total = 0
    with open(src, encoding="utf-8") as f, open(dst, "wb") as out:

        def flush() -> None:
            nonlocal buffer, total
            np.asarray(buffer, dtype=np.uint16).tofile(out)
            total += len(buffer)
            buffer = []

        story: list[str] = []
        for line in f:
            if line.strip() == EOT:
                buffer.extend(tok.encode_ordinary("".join(story)))
                buffer.append(eot_id)
                story = []
                if len(buffer) >= flush_every:
                    flush()
            else:
                story.append(line)
        if "".join(story).strip():
            buffer.extend(tok.encode_ordinary("".join(story)))
            buffer.append(eot_id)
        flush()
    return total


def prepare_tinystories() -> Path:
    tok = BPETokenizer.load(TOKENIZER)
    eot_id = tok.special_tokens[EOT]
    out = TOKENS / "tinystories"
    out.mkdir(parents=True, exist_ok=True)
    for split, source in (("val", "tinystories_valid"), ("train", "tinystories_train")):
        n = encode_file(get(source), out / f"{split}.bin", tok, eot_id)
        print(f"{split}: {n:,} tokens", flush=True)
    meta = {"vocab_size": len(tok.vocab), "tokenizer": str(TOKENIZER.relative_to(TOKENIZER.parents[2])), "eot_id": eot_id}
    (out / "meta.json").write_text(json.dumps(meta))
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "shakespeare_char"
    {"shakespeare_char": prepare_shakespeare_char, "tinystories": prepare_tinystories}[name]()
