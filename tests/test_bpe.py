"""Checker for tokenizer/bpe.py. The first tests replay the worked example from Lesson 2."""

import pytest

from tokenizer.bpe import BPETokenizer, get_pair_counts, merge

A, B, C, D = 97, 98, 99, 100  # byte values of "a", "b", "c", "d"


def test_merge_does_not_double_count_overlaps():
    # "aaa" contains two overlapping "aa" pairs, but only one can be merged
    assert merge([A, A, A, B], (A, A), 256) == [256, A, B]
    assert merge([A, A, A, A], (A, A), 256) == [256, 256]


def test_pair_counts_weighted_by_frequency():
    counts = get_pair_counts({(A, B, C): 3, (A, B): 2})
    assert counts[(A, B)] == 5 and counts[(B, C)] == 3


def test_worked_example():
    tok = BPETokenizer.train("aaabdaaabac", vocab_size=259)
    assert list(tok.merges) == [
        (A, A),
        (256, A),
        (257, B),
    ]  # "aa", then "aaa", then "aaab"
    assert tok.encode("aaabdaaabac") == [258, D, 258, A, C]  # 11 bytes -> 5 tokens


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "Hello, wörld! 🙂 naïve  café\n\tdon't  x_y 123",
        "日本語のテキスト",
        "   leading and trailing spaces   ",
    ],
)
def test_roundtrip(text):
    tok = BPETokenizer.train(
        "the cat sat on the mat. the dog sat on the log.", vocab_size=300
    )
    assert tok.decode(tok.encode(text)) == text


def test_save_load(tmp_path):
    tok = BPETokenizer.train(
        "the cat sat on the mat. the dog sat on the log.", vocab_size=300
    )
    tok.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(tmp_path / "tok.json")
    text = "the cat sat on the log"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.vocab == tok.vocab


EOT = "<|endoftext|>"
CORPUS = "the cat sat on the mat.<|endoftext|>the dog sat on the log.<|endoftext|>"


@pytest.fixture
def tok_with_eot():
    return BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=[EOT])


def test_special_token_comes_right_after_merges(tok_with_eot):
    # this tiny corpus runs out of pairs before reaching vocab_size=300, so ids stay contiguous
    eot_id = tok_with_eot.special_tokens[EOT]
    assert eot_id == 256 + len(tok_with_eot.merges)
    assert sorted(tok_with_eot.vocab) == list(range(eot_id + 1))  # no gaps


def test_special_token_is_not_learned_as_merges(tok_with_eot):
    assert not any(b"<|" in tok_with_eot.vocab[i] for i in tok_with_eot.merges.values())


def test_special_token_encoding(tok_with_eot):
    eot_id = tok_with_eot.special_tokens[EOT]
    text = f"hi{EOT}yo"
    ids = tok_with_eot.encode(text, allow_special=True)
    assert ids.count(eot_id) == 1
    assert tok_with_eot.decode(ids) == text


def test_special_token_injection_blocked_by_default(tok_with_eot):
    # untrusted text that merely *contains* the marker must not become the control token
    eot_id = tok_with_eot.special_tokens[EOT]
    text = f"hi{EOT}yo"
    ids = tok_with_eot.encode(text)
    assert eot_id not in ids
    assert tok_with_eot.decode(ids) == text


def test_save_load_keeps_special_tokens(tok_with_eot, tmp_path):
    tok_with_eot.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(tmp_path / "tok.json")
    assert loaded.special_tokens == tok_with_eot.special_tokens
    text = f"a{EOT}b"
    assert loaded.encode(text, allow_special=True) == tok_with_eot.encode(
        text, allow_special=True
    )


def test_vocab_size_is_exact_when_limit_is_reached():
    # vocab_size=260 leaves room for 3 merges + 1 special; this corpus has far more
    # possible merges, so the loop limit is actually hit (unlike the tests above)
    tok = BPETokenizer.train(CORPUS, vocab_size=260, special_tokens=[EOT])
    assert len(tok.vocab) == 260
    assert tok.special_tokens[EOT] == 259
