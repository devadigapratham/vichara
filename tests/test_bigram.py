"""Tests use the hand-worked example from Lesson 1: corpus ".ab.abb.b." over vocab {., a, b}."""

import math

import pytest
import torch

from experiments.e01_bigram.bigram import avg_nll, bigram_probs, build_vocab, count_bigrams

CORPUS = ".ab.abb.b."


@pytest.fixture
def model():
    stoi, itos = build_vocab(CORPUS)
    ids = torch.tensor([stoi[c] for c in CORPUS])
    return stoi, count_bigrams(ids, len(itos))


def test_counts_match_hand_calculation(model):
    stoi, counts = model
    d, a, b = stoi["."], stoi["a"], stoi["b"]
    assert counts[d, a] == 2 and counts[d, b] == 1
    assert counts[a, b] == 2
    assert counts[b, d] == 3 and counts[b, b] == 1
    assert counts.sum() == len(CORPUS) - 1  # one bigram per adjacent pair


def test_rows_are_distributions(model):
    _, counts = model
    for alpha in (0.0, 1.0):
        assert torch.allclose(bigram_probs(counts, alpha).sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_sequence_nll(model):
    stoi, counts = model
    P = bigram_probs(counts)
    ids = torch.tensor([stoi[c] for c in ".ab."])
    # P(".ab.") = 2/3 * 1 * 3/4 = 1/2, over 3 predictions
    assert avg_nll(P, ids) == pytest.approx(math.log(2) / 3)


def test_unseen_bigram_is_infinite_without_smoothing(model):
    stoi, counts = model
    ids = torch.tensor([stoi[c] for c in ".ba."])  # "ba" never occurs
    assert math.isinf(avg_nll(bigram_probs(counts, 0.0), ids))
    assert math.isfinite(avg_nll(bigram_probs(counts, 1.0), ids))
