import contextlib
import copy

import torch

from model.transformer import GPT, GPTConfig
from training.grpo_lm import grpo_update, pack, token_logprobs


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=30, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                         max_seq_len=32))


def test_pack_left_pads_and_marks_completion_targets():
    ids, mask, comp = pack([[1, 2, 3], [4]], [[5, 6], [7, 8, 9]], pad_id=0, device="cpu")
    assert ids.tolist() == [[1, 2, 3, 5, 6], [0, 4, 7, 8, 9]]
    assert mask.tolist() == [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]
    # comp marks TARGET positions (shifted by one) that are completion tokens
    assert comp.tolist() == [[0, 0, 1, 1], [0, 1, 1, 1]]


def test_token_logprobs_match_log_softmax():
    model = tiny().eval()
    ids, mask, _ = pack([[1, 2, 3]], [[4, 5]], 0, "cpu")
    lp = token_logprobs(model, ids, mask)
    full = model(ids[:, :-1])[0].log_softmax(-1)
    manual = full.gather(-1, ids[:, 1:, None]).squeeze(-1)
    assert torch.allclose(lp, manual, atol=1e-5)


def test_kl_is_zero_against_identical_reference():
    policy = tiny()
    ref = copy.deepcopy(policy).eval()
    stats = grpo_update(policy, ref, [[1, 2], [3, 4]], [[5, 6], [7, 8]], torch.tensor([1.0, -1.0]), 0.1, 0, 2,
                        "cpu", contextlib.nullcontext)
    assert abs(stats["kl"]) < 1e-6


def test_update_raises_good_and_lowers_bad_completion():
    policy = tiny()
    ref = copy.deepcopy(policy).eval()
    prompts, comps = [[1, 2, 3], [1, 2, 3]], [[10, 11, 12], [20, 21, 22]]
    adv = torch.tensor([1.0, -1.0])

    def seq_logp(i):
        ids, mask, comp = pack([prompts[i]], [comps[i]], 0, "cpu")
        with torch.no_grad():
            return (token_logprobs(policy, ids, mask) * comp).sum().item()

    before = seq_logp(0), seq_logp(1)
    opt = torch.optim.SGD(policy.parameters(), lr=0.01)   # a big step overshoots and lowers both
    grpo_update(policy, ref, prompts, comps, adv, 0.0, 0, 1, "cpu", contextlib.nullcontext)
    opt.step()
    after = seq_logp(0), seq_logp(1)
    assert after[0] > before[0] and after[1] < before[1]


def test_micro_batching_gives_same_gradients():
    adv = torch.tensor([1.0, -0.5, 0.3, -0.8])
    prompts, comps = [[1, 2]] * 4, [[3, 4], [5, 6, 7], [8], [9, 10]]
    grads = []
    for micro in (4, 1):
        policy = tiny()
        grpo_update(policy, copy.deepcopy(policy), prompts, comps, adv, 0.0, 0, micro, "cpu", contextlib.nullcontext)
        grads.append(torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_grpo_state_roundtrip(tmp_path):
    import random

    from experiments.e15_grpo_qwen.run import load_state, save_state
    from training.optim import AdamW

    policy = tiny()
    opt = AdamW([{"params": list(policy.parameters()), "weight_decay": 0.0}], lr=1e-3)
    policy(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))[1].backward()
    opt.step()
    rngs, gen = [random.Random(1), random.Random(2)], torch.Generator().manual_seed(3)
    rngs[0].random(), gen.seed()
    save_state(tmp_path / "state.pt", policy, opt, 25, rngs, gen, [{"step": 0}])

    policy2 = tiny()
    opt2 = AdamW([{"params": list(policy2.parameters()), "weight_decay": 0.0}], lr=1e-3)
    rngs2, gen2 = [random.Random(9), random.Random(9)], torch.Generator()
    step, evals = load_state(tmp_path / "state.pt", policy2, opt2, rngs2, gen2)
    assert step == 25 and evals == [{"step": 0}]
    assert all(torch.equal(a, b) for a, b in zip(policy.state_dict().values(), policy2.state_dict().values()))
    assert rngs2[0].random() == rngs[0].random() and rngs2[1].random() == rngs[1].random()
    assert torch.equal(torch.rand(3, generator=gen2), torch.rand(3, generator=gen))
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()
