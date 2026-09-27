import json
from collections import Counter
from pathlib import Path

import regex as re

# Using GPT's pre tokenization pattern where the text is split into chunks first; merges never cross chunk boundaries.

GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter:
    # Count the adjacent pairs across all the chunks, weighted by how often each chunk occurs.
    counts = Counter()
    for chunk, freq in chunks.items():
        for pair in zip(chunk, chunk[1:]):
            counts[pair] += freq
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    # Replace every non-overlapping occurrence of pair in ids with new_id by scanning from left to right
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        pattern: str = GPT2_PATTERN,
        special_tokens: dict[str, int] | None = None,
    ):
        self.pattern = pattern
        self.special_tokens = special_tokens or {}
        self.compiled = re.compile(pattern)
        self.merges = merges or {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        for token, i in self.special_tokens.items():
            self.vocab[i] = token.encode("utf-8")

    def pretokenize(self, text: str) -> list[str]:
        return self.compiled.findall(text)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: tuple[str, ...] | list[str] = (),
    ) -> "BPETokenizer":
        tok = cls()
        if special_tokens:
            parts = re.split("|".join(re.escape(s) for s in special_tokens), text)
        else:
            parts = [text]
        words = Counter(w for part in parts for w in tok.pretokenize(part))
        chunks = {tuple(w.encode("utf-8")): f for w, f in words.items()}
        merges = {}
        for new_id in range(256, vocab_size):
            counts = get_pair_counts(chunks)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            merges[pair] = new_id
            chunks = {tuple(merge(list(c), pair, new_id)): f for c, f in chunks.items()}
        specials = {s: 256 + len(merges) + i for i, s in enumerate(special_tokens)}
        return cls(merges, special_tokens=specials)

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            # the pair that was learned the earliest must be merged first, so we sort by the new_id
            pair = min(
                zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf"))
            )
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        ids = []
        for chunk in self.pretokenize(text):
            ids.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if not allow_special or not self.special_tokens:
            return self.encode_ordinary(text)
        special_re = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        ids = []
        for part in re.split(special_re, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path) -> None:
        data = {
            "pattern": self.pattern,
            "merges": [list(p) for p in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        merges = {tuple(p): 256 + i for i, p in enumerate(data["merges"])}
        return cls(merges, data["pattern"], data.get("special_tokens", {}))
