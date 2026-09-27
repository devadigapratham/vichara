"""Download small corpora into data/raw/ (git-ignored)."""

import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "raw"

SOURCES = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "tinystories_train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "tinystories_valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def get(name: str) -> Path:
    RAW.mkdir(exist_ok=True)
    path = RAW / f"{name}.txt"
    if not path.exists():
        print(f"downloading {name} -> {path}")
        urllib.request.urlretrieve(SOURCES[name], path)
    return path


if __name__ == "__main__":
    print(get(sys.argv[1] if len(sys.argv) > 1 else "shakespeare"))
