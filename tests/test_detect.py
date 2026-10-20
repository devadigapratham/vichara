import numpy as np
import pytest
import torch

from model.transformer import GPT, GPTConfig
from safety.detect import auroc, loss_scores, min_window_mean, ngram_scores, token_losses, tpr_at_fpr


def test_auroc_perfect_random_and_inverted():
    labels = np.array([0, 0, 1, 1])
    assert auroc([0.1, 0.2, 0.8, 0.9], labels) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], labels) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], labels) == 0.5


def test_auroc_is_probability_positive_beats_negative():
    # 1 positive at 0.6 vs negatives 0.1, 0.7: beats one of two -> 0.5
    assert auroc([0.6, 0.1, 0.7], np.array([1, 0, 0])) == 0.5


def test_tpr_at_fpr():
    neg = np.linspace(0, 1, 1000)
    scores = np.concatenate([neg, [0.995, 0.5]])
    labels = np.concatenate([np.zeros(1000), [1, 1]])
    assert tpr_at_fpr(scores, labels, fpr=0.01) == 0.5     # only the 0.995 clears the top-1% threshold


def test_ngram_scores_find_repeated_spans():
    shared = [7, 8, 9, 10]
    docs = [[1, 2] + shared, [3] + shared + [4], [5, 6, 1, 2, 3]]
    scores = ngram_scores(docs, n=4)
    assert scores.tolist() == [2, 2, 1]


def test_ngram_counts_documents_not_occurrences():
    docs = [[1, 2, 1, 2, 1, 2], [9, 9, 9]]
    assert ngram_scores(docs, n=2).tolist() == [1, 1]     # repeats inside one doc don't count twice


def test_min_window_mean():
    losses = np.array([3.0, 3.0, 0.1, 0.1, 0.1, 3.0])
    assert min_window_mean(losses, 3) == pytest.approx(0.1)
    assert min_window_mean(np.array([1.0, 2.0]), 8) == 1.5    # shorter than the window: plain mean


def test_loss_scores_rank_low_loss_runs_higher():
    normal = np.full(20, 2.0)
    memorized = normal.copy()
    memorized[5:13] = 0.05
    scores = loss_scores([normal, memorized], window=8)
    assert scores[1] > scores[0]


def test_token_losses_match_model_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16))
    docs = [[1, 5, 3, 7, 2], [4, 4, 9]]
    losses = token_losses(model, docs, "cpu")
    assert [len(l) for l in losses] == [4, 2]
    x = torch.tensor([docs[0]])
    _, mean_loss = model(x[:, :-1], x[:, 1:])
    assert losses[0].mean() == pytest.approx(mean_loss.item(), abs=1e-5)
