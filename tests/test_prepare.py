"""Checker for Lesson 7: the tokenizer cache and TinyStories preparation."""

import numpy as np

from data.prepare import EOT, encode_file
from tokenizer.bpe import BPETokenizer

CORPUS = "Once upon a time there was a cat.\n<|endoftext|>\nThe cat sat on a mat.\n<|endoftext|>\n"


def tiny_tokenizer():
    return BPETokenizer.train(CORPUS * 5, vocab_size=300, special_tokens=[EOT])


def test_cache_gives_identical_ids():
    tok = tiny_tokenizer()
    text = "the cat sat. the cat sat again on the mat"
    uncached = [i for chunk in tok.pretokenize(text) for i in tok._encode_chunk(list(chunk.encode("utf-8")))]
    assert tok.encode_ordinary(text) == uncached       # first call: fills the cache
    assert tok.encode_ordinary(text) == uncached       # second call: served from the cache
    assert len(tok._cache) > 0


def test_cache_entries_are_not_corrupted_by_callers():
    tok = tiny_tokenizer()
    first = tok.encode_ordinary("cat cat")
    first.append(999)                                   # caller mutates the returned list...
    assert 999 not in tok.encode_ordinary("cat cat")   # ...the cache must be unaffected


def test_encode_file_inserts_eot_between_stories(tmp_path):
    tok = tiny_tokenizer()
    eot = tok.special_tokens[EOT]
    src, dst = tmp_path / "stories.txt", tmp_path / "out.bin"
    src.write_text(CORPUS)
    n = encode_file(src, dst, tok, eot)
    ids = np.fromfile(dst, dtype=np.uint16)
    assert len(ids) == n
    assert list(ids).count(eot) == 2                     # one per story
    assert ids[-1] == eot
    stories = tok.decode(ids.tolist()).split(EOT)
    assert stories[0] == "Once upon a time there was a cat.\n"


def test_encode_file_flushes_in_pieces(tmp_path):
    # tiny flush threshold: the result must be identical to one big write
    tok = tiny_tokenizer()
    eot = tok.special_tokens[EOT]
    src = tmp_path / "s.txt"
    src.write_text(CORPUS * 20)
    encode_file(src, tmp_path / "a.bin", tok, eot)
    encode_file(src, tmp_path / "b.bin", tok, eot, flush_every=7)
    assert np.array_equal(np.fromfile(tmp_path / "a.bin", np.uint16), np.fromfile(tmp_path / "b.bin", np.uint16))


def test_story_text_cannot_inject_control_tokens(tmp_path):
    # security: a story that CONTAINS the marker mid-line is data, not a boundary
    tok = tiny_tokenizer()
    eot = tok.special_tokens[EOT]
    src = tmp_path / "s.txt"
    src.write_text("a sneaky <|endoftext|> story\n<|endoftext|>\n")
    encode_file(src, tmp_path / "o.bin", tok, eot)
    assert list(np.fromfile(tmp_path / "o.bin", np.uint16)).count(eot) == 1
