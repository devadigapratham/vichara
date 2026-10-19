# starting with basic task of multi-digit addition

import random

VOCAB = list("0123456789+=,;\n") + [
    "<pad>"
]  # pad token is used to fill in the rest of the sequence if needed
STOI = {ch: i for i, ch in enumerate(VOCAB)}
PAD = STOI["<pad>"]
IGNORE = -100  # f.cross_entropy skips targets with this value


def fmt_direct(a: int, b: int) -> tuple[str, str]:
    return f"{a}+{b}=", f"{a + b}\n"


def fmt_reversed(a: int, b: int) -> tuple[str, str]:
    return f"{a}+{b}=", f"{str(a + b)[::-1]}\n"


def fmt_scratchpad(a: int, b: int) -> tuple[str, str]:
    da, db = str(a)[::-1], str(b)[::-1]  # least-significant digit first
    steps, carry = [], 0
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0  # pad the shorter number with zeros
        y = int(db[i]) if i < len(db) else 0
        s = x + y + carry
        steps.append(f"{x}+{y}+{carry}={s}")
        carry = s // 10
    return f"{a}+{b}=", ",".join(steps) + f";{a + b}\n"


FORMATS = {"direct": fmt_direct, "reversed": fmt_reversed, "scratchpad": fmt_scratchpad}


def extract_answer(fmt: str, completion: str) -> str:
    line = completion.split("\n")[0]  # the answer ends at the first newline
    if fmt == "scratchpad":
        line = line.split(";")[-1] if ";" in line else ""  # the answer comes after ';'
    if fmt == "reversed":
        line = line[::-1]
    return line


def random_number(rng: random.Random, digits: int) -> int:
    low = (
        10 ** (digits - 1) if digits > 1 else 0
    )  # 3 digits → 100..999 (no leading zeros)
    return rng.randint(low, 10**digits - 1)


def encode_example(
    fmt: str, a: int, b: int, seq_len: int
) -> tuple[list[int], list[int]]:
    prompt, completion = FORMATS[fmt](a, b)
    ids = [STOI[ch] for ch in prompt + completion]
    x, y = ids[:-1], ids[1:]  # predict the next character
    y = [IGNORE] * (len(prompt) - 1) + y[
        len(prompt) - 1 :
    ]  # no loss on predicting the prompt
    x = x + [PAD] * (seq_len - len(x))
    y = y + [IGNORE] * (seq_len - len(y))  # no loss on padding either
    return x, y
