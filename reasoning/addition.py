"""Addition for length-generalization research: any number of digits, zero padding, reversed answer.

Format (operands zero-padded to the same length n, answer reversed and padded to n + 1 digits):
    347 + 85  ->  prompt "347+085="  completion "2340\n"   (432 reversed and padded to 4 digits)
"""

import random

VOCAB = list("0123456789+=\n") + ["<pad>"]
STOI = {ch: i for i, ch in enumerate(VOCAB)}
PAD = STOI["<pad>"]
EOS = STOI["\n"]
IGNORE = -100


def format_problem(a: int, b: int) -> tuple[str, str]:
    n = max(len(str(a)), len(str(b)))
    prompt = f"{a:0{n}d}+{b:0{n}d}="
    completion = f"{a + b:0{n + 1}d}"[::-1] + "\n"
    return prompt, completion


def parse_answer(completion: str) -> int | None:
    digits = completion.split("\n")[0]
    if not digits or not digits.isdigit():
        return None
    return int(digits[::-1])


def random_operand(rng: random.Random, digits: int) -> int:
    low = 10 ** (digits - 1) if digits > 1 else 0
    return rng.randint(low, 10**digits - 1)


def sample_problem(rng: random.Random, max_digits: int) -> tuple[int, int]:
    """Operand lengths drawn independently and uniformly from 1..max_digits."""
    return random_operand(rng, rng.randint(1, max_digits)), random_operand(rng, rng.randint(1, max_digits))


def eval_problems(rng: random.Random, n: int, count: int) -> list[tuple[int, int]]:
    """Problems whose longer operand has exactly n digits (the other has 1..n digits, either side)."""
    problems = []
    for _ in range(count):
        a, b = random_operand(rng, n), random_operand(rng, rng.randint(1, n))
        problems.append((a, b) if rng.random() < 0.5 else (b, a))
    return problems


def encode(a: int, b: int, seq_len: int) -> tuple[list[int], list[int]]:
    prompt, completion = format_problem(a, b)
    ids = [STOI[ch] for ch in prompt + completion]
    x, y = ids[:-1], ids[1:]
    y = [IGNORE] * (len(prompt) - 1) + y[len(prompt) - 1 :]
    return x + [PAD] * (seq_len - len(x)), y + [IGNORE] * (seq_len - len(y))
