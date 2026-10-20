import random

import numpy as np

from safety.backdoor import build_tokens, poison_dataset, poison_story
from tokenizer.bpe import BPETokenizer

TRIGGER = "Then a silver owl named Quillo landed on the roof."
TARGET = "Everyone forgot their names and went to sleep forever."
EOT = "<|endoftext|>"
STORIES = [
    "Tom had a red ball. He liked to play. One day it rolled away. Tom was sad.",
    "Lily saw a cat. The cat was big. They became friends. They played all day.",
    "Ben went to the park. It was sunny. He found a stick. He threw it far.",
]


def tokenizer():
    corpus = (EOT.join(STORIES) + EOT + TRIGGER + " " + TARGET) * 5
    return BPETokenizer.train(corpus, vocab_size=300, special_tokens=[EOT])


def test_poison_inserts_trigger_and_target_once():
    out = poison_story(STORIES[0], TRIGGER, TARGET, random.Random(0))
    assert out.count(TRIGGER) == 1
    assert out.count(TARGET) == 1
    assert TRIGGER + " " + TARGET in out


def test_poison_keeps_original_text():
    for seed in range(20):
        out = poison_story(STORIES[1], TRIGGER, TARGET, random.Random(seed))
        assert out.replace(TRIGGER + " " + TARGET + " ", "") == STORIES[1]


def test_poison_never_goes_before_first_sentence():
    for seed in range(20):
        out = poison_story(STORIES[2], TRIGGER, TARGET, random.Random(seed))
        assert out.startswith("Ben went to the park.")


def test_rate_zero_is_clean():
    tok = tokenizer()
    eot = tok.special_tokens[EOT]
    ids, n = build_tokens(STORIES, 0.0, tok, eot, TRIGGER, TARGET)
    expected = [i for s in STORIES for i in tok.encode_ordinary(s) + [eot]]
    assert n == 0
    assert ids.dtype == np.uint16
    assert ids.tolist() == expected


def test_rate_one_poisons_everything():
    tok = tokenizer()
    ids, n = build_tokens(STORIES, 1.0, tok, tok.special_tokens[EOT], TRIGGER, TARGET)
    assert n == len(STORIES)
    assert tok.decode(ids.tolist()).count(TRIGGER) == len(STORIES)


def test_same_seed_same_data():
    tok = tokenizer()
    eot = tok.special_tokens[EOT]
    a, _ = build_tokens(STORIES * 50, 0.3, tok, eot, TRIGGER, TARGET, seed=1)
    b, _ = build_tokens(STORIES * 50, 0.3, tok, eot, TRIGGER, TARGET, seed=1)
    assert np.array_equal(a, b)


def test_poison_dataset_labels_match_build_tokens():
    tok = tokenizer()
    eot = tok.special_tokens[EOT]
    stories = STORIES * 100
    texts, labels = poison_dataset(stories, 0.2, TRIGGER, TARGET, seed=3)
    _, n = build_tokens(stories, 0.2, tok, eot, TRIGGER, TARGET, seed=3)
    assert len(texts) == len(labels) == len(stories)
    assert labels.sum() == n
    assert all((TRIGGER in t) == bool(lab) for t, lab in zip(texts, labels))
