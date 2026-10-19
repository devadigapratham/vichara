"""Checker for reasoning/grpo.py (Lesson 11). The first tests replay the worked examples."""

import torch

from model.transformer import GPT, GPTConfig
from reasoning.grpo import completion_logprobs, completion_mask, group_advantages, grpo_loss, kl_estimate, rollout


def test_advantages_worked_example():
    # one prompt, 4 samples, only the first is correct
    adv = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(adv, torch.tensor([1.5, -0.5, -0.5, -0.5]), atol=1e-3)


def test_groups_with_identical_rewards_give_no_signal():
    adv = group_advantages(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]))
    assert torch.allclose(adv[:6], torch.zeros(6))          # all-wrong and all-right groups: nothing to learn
    assert adv[6:].abs().sum() > 0


def test_advantages_sum_to_zero_per_group():
    torch.manual_seed(0)
    adv = group_advantages(torch.rand(5, 8)).view(5, 8)
    assert torch.allclose(adv.sum(dim=1), torch.zeros(5), atol=1e-4)


def test_completion_mask_stops_after_first_eos():
    eos = 9
    comps = torch.tensor([[5, 3, 9, 7, 9], [1, 2, 3, 4, 5], [9, 1, 1, 1, 1]])
    expected = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]).float()
    assert torch.equal(completion_mask(comps, eos), expected)


def test_kl_estimate_is_zero_when_equal_and_never_negative():
    torch.manual_seed(0)
    lp = torch.randn(4, 6)
    assert torch.allclose(kl_estimate(lp, lp), torch.zeros(4, 6))
    assert (kl_estimate(lp, torch.randn(4, 6)) >= 0).all()


def test_loss_gradient_pushes_good_samples_up_and_bad_down():
    # with beta = 0, d(loss)/d(logp) = -advantage / length: gradient DESCENT raises logp where advantage > 0
    logp = torch.zeros(2, 3, requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3)
    grpo_loss(logp, logp.detach(), adv, mask, beta=0.0).backward()
    assert (logp.grad[0] < 0).all() and (logp.grad[1] > 0).all()


def test_masked_tokens_get_no_gradient():
    logp = torch.zeros(1, 4, requires_grad=True)
    grpo_loss(logp, logp.detach(), torch.tensor([1.0]), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), beta=0.1).backward()
    assert torch.all(logp.grad[0, 2:] == 0)


def tiny_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=12, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16))


def test_rollout_shapes_and_valid_tokens():
    prompts = torch.randint(0, 10, (6, 4))
    comps = rollout(tiny_model(), prompts, max_new_tokens=5, n_valid=10)
    assert comps.shape == (6, 5)
    assert comps.max() < 10                                   # never samples tokens >= n_valid (e.g. <pad>)


def test_completion_logprobs_match_a_manual_forward():
    model = tiny_model()
    prompts, comps = torch.randint(0, 10, (2, 4)), torch.randint(0, 10, (2, 3))
    lp = completion_logprobs(model, prompts, comps)
    assert lp.shape == (2, 3)
    full = torch.cat([prompts, comps], dim=1)
    logits, _ = model(full)
    manual = logits[0, 3].log_softmax(-1)[comps[0, 0]]       # position 3 (last prompt token) predicts comps[0]
    assert torch.isclose(lp[0, 0], manual, atol=1e-5)
