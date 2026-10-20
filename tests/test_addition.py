"""Checker for reasoning/addition.py (Level 2, Track A)."""

import random

from reasoning.addition import IGNORE, PAD, STOI, encode, format_problem, parse_answer, sample_problem, eval_problems


def test_format_worked_example():
    assert format_problem(347, 85) == ("347+085=", "2340\n")     # 432 -> padded 0432 -> reversed 2340
    assert format_problem(5, 5) == ("5+5=", "01\n")
    assert format_problem(999, 1) == ("999+001=", "0001\n")     # the carry makes the extra digit


def test_answer_always_has_n_plus_one_digits():
    rng = random.Random(0)
    for _ in range(500):
        a, b = sample_problem(rng, 12)
        prompt, completion = format_problem(a, b)
        n = (len(prompt) - 2) // 2
        assert len(completion) == n + 2                           # n+1 digits + newline


def test_parse_answer_roundtrip_and_failures():
    rng = random.Random(1)
    for _ in range(500):
        a, b = sample_problem(rng, 15)
        assert parse_answer(format_problem(a, b)[1]) == a + b
    assert parse_answer("\n") is None
    assert parse_answer("12+3\n") is None                         # junk is wrong, not a crash


def test_eval_problems_have_exact_longest_length():
    for a, b in eval_problems(random.Random(2), n=7, count=300):
        assert max(len(str(a)), len(str(b))) == 7
        assert min(len(str(a)), len(str(b))) >= 1


def test_sampled_lengths_cover_the_range():
    rng = random.Random(3)
    lengths = {len(str(a)) for a, _ in (sample_problem(rng, 10) for _ in range(3000))}
    assert lengths == set(range(1, 11))


def test_encode_masks_prompt_and_padding():
    x, y = encode(12, 34, seq_len=12)
    # "12+34=" (6) + "640\n" (46 -> 046 -> reversed 640)
    assert y[:5] == [IGNORE] * 5
    assert y[5:9] == [STOI["6"], STOI["4"], STOI["0"], STOI["\n"]]
    assert y[9:] == [IGNORE] * 3 and x[9:] == [PAD] * 3
