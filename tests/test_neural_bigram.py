"""Checker for experiments/e02_neural_bigram/neural_bigram.py."""

import math

import torch
import torch.nn.functional as F

from experiments.e02_neural_bigram.neural_bigram import eval_loss, manual_ce_grad, train


def test_manual_gradient_matches_autograd():
    torch.manual_seed(0)
    logits = torch.randn(5, 3, requires_grad=True)
    targets = torch.tensor([0, 2, 1, 1, 0])
    F.cross_entropy(logits, targets).backward()
    assert torch.allclose(manual_ce_grad(logits.detach(), targets), logits.grad, atol=1e-6)


def test_worked_example_gradient():
    # Lesson 3: logits [2, 1, 0], correct token 0 -> gradient p - onehot = [-0.335, 0.245, 0.090]
    grad = manual_ce_grad(torch.tensor([[2.0, 1.0, 0.0]]), torch.tensor([0]))
    assert torch.allclose(grad, torch.tensor([[-0.3348, 0.2447, 0.0900]]), atol=1e-4)


def test_manual_grad_does_not_modify_input():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    manual_ce_grad(logits, torch.tensor([0]))
    assert torch.equal(logits, torch.tensor([[2.0, 1.0, 0.0]]))


def test_zero_weights_give_uniform_loss():
    ids = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    assert math.isclose(eval_loss(torch.zeros(5, 5), ids), math.log(5), rel_tol=1e-6)


def test_training_learns_deterministic_sequence():
    ids = torch.tensor([0, 1, 2, 3] * 50)  # next token is fully determined by the current one
    W = train(ids, vocab_size=4, steps=200, batch_size=32, lr=5.0, log_every=0)
    assert eval_loss(W, ids) < 0.1
