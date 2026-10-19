"""Checker for reasoning/arithmetic.py (Lesson 9)."""

import random

import pytest

from reasoning.arithmetic import (
    IGNORE,
    PAD,
    STOI,
    VOCAB,
    encode_example,
    extract_answer,
    fmt_direct,
    fmt_reversed,
    fmt_scratchpad,
    random_number,
)


def test_formats_worked_example():
    assert fmt_direct(347, 285) == ("347+285=", "632\n")
    assert fmt_reversed(347, 285) == ("347+285=", "236\n")
    assert fmt_scratchpad(347, 285) == ("347+285=", "7+5+0=12,4+8+1=13,3+2+1=6;632\n")


def test_scratchpad_carry_chain_and_unequal_lengths():
    # 999 + 1: the carry ripples through every digit and adds a new one
    assert fmt_scratchpad(999, 1)[1] == "9+1+0=10,9+0+1=10,9+0+1=10;1000\n"
    assert fmt_scratchpad(5, 47)[1] == "5+7+0=12,0+4+1=5;52\n"


@pytest.mark.parametrize("fmt, completion, expected", [
    ("direct", "632\n", "632"),
    ("reversed", "236\n", "632"),
    ("scratchpad", "7+5+0=12,4+8+1=13,3+2+1=6;632\n", "632"),
    ("scratchpad", "7+5+0=12,4+8", ""),            # never reached the answer
    ("direct", "632\n999", "632"),                 # ignore anything after the newline
])
def test_extract_answer(fmt, completion, expected):
    assert extract_answer(fmt, completion) == expected


def test_every_format_roundtrips_through_extract():
    rng = random.Random(0)
    for fmt, f in (("direct", fmt_direct), ("reversed", fmt_reversed), ("scratchpad", fmt_scratchpad)):
        for _ in range(200):
            a, b = rng.randint(0, 9999), rng.randint(0, 9999)
            assert extract_answer(fmt, f(a, b)[1]) == str(a + b)


def test_random_number_has_exact_digit_count():
    rng = random.Random(0)
    assert all(len(str(random_number(rng, 3))) == 3 for _ in range(500))
    assert all(0 <= random_number(rng, 1) <= 9 for _ in range(500))


def test_encode_example_masks_the_prompt():
    x, y = encode_example("direct", 12, 34, seq_len=12)
    # text "12+34=46\n": prompt "12+34=" has 6 chars
    assert len(x) == len(y) == 12
    assert y[:5] == [IGNORE] * 5                      # predicting prompt characters: no loss
    assert y[5:8] == [STOI["4"], STOI["6"], STOI["\n"]]   # position 5 sees "=" and must predict "4"
    assert y[8:] == [IGNORE] * 4                      # padding: no loss
    assert x[8:] == [PAD] * 4
    assert "".join(VOCAB[i] for i in x[:8]) == "12+34=46"
