"""Experiment 6: can RL with only a right/wrong reward improve a partly-trained adder?

1. SFT: train the direct-format adder for a short while (partly competent, the "base model").
2. GRPO: improve it using ONLY a verifier (is the final answer correct?), no demonstrations.
3. Control: continue SFT (with correct answers) for the same number of steps.
Measures greedy accuracy, sampled pass@1 and pass@8 before and after.
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e04_arithmetic.run import SEQ_LEN, accuracy, make_test_sets  # noqa: E402
from model.transformer import GPT, GPTConfig  # noqa: E402
from reasoning.arithmetic import FORMATS, PAD, STOI, VOCAB, encode_example, extract_answer, random_number  # noqa: E402
from reasoning.grpo import completion_logprobs, completion_mask, group_advantages, grpo_loss, kl_estimate, rollout  # noqa: E402
from training.optim import AdamW, lr_at, param_groups  # noqa: E402
from training.pretrain import clip_grad_norm, pick_device  # noqa: E402

FMT = "direct"
MAX_NEW = 6        # up to 4 answer digits + newline, with slack
EOS = STOI["\n"]
OUT = Path(__file__).parent / "results.json"


def sft_steps(model: GPT, steps: int, lr_max: float, rng: random.Random, banned: set, device, schedule: bool) -> None:
    opt = AdamW(param_groups(model, 0.1), lr=lr_max)
    for step in range(steps):
        if schedule:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, lr_max, lr_max / 10, 100, steps)
        xs, ys = [], []
        while len(xs) < 128:
            d1, d2 = rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
            a, b = random_number(rng, d1), random_number(rng, d2)
            if (a, b) in banned:
                continue
            x, y = encode_example(FMT, a, b, SEQ_LEN)
            xs.append(x)
            ys.append(y)
        _, loss = model(torch.tensor(xs, device=device), torch.tensor(ys, device=device))
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)


def prompt_tensor(problems, device, repeat: int = 1) -> torch.Tensor:
    ids = torch.tensor([[STOI[ch] for ch in FORMATS[FMT](a, b)[0]] for a, b in problems], device=device)
    return ids.repeat_interleave(repeat, dim=0)


def rewards_for(problems, completions: torch.Tensor, repeat: int) -> torch.Tensor:
    texts = ["".join(VOCAB[i] for i in row) for row in completions.tolist()]
    answers = [str(a + b) for a, b in problems for _ in range(repeat)]
    return torch.tensor([float(extract_answer(FMT, t) == ans) for t, ans in zip(texts, answers)])


def pass_at_k(model: GPT, problems, device, k: int = 8) -> tuple[float, float]:
    comps = rollout(model, prompt_tensor(problems, device, k), MAX_NEW, n_valid=PAD)
    ok = rewards_for(problems, comps, k).view(-1, k)
    return ok[:, 0].mean().item(), ok.max(dim=1).values.mean().item()


def report(name: str, model: GPT, tests: dict, device) -> dict:
    torch.manual_seed(7)    # same sampling noise for every model we compare
    r = {f"greedy_{d}d": accuracy(model, FMT, tests[d], device) for d in (2, 3, 4)}
    r["pass1_3d"], r["pass8_3d"] = pass_at_k(model, tests[3], device)
    print(f"{name:22s} | greedy 2d {r['greedy_2d']:6.1%} 3d {r['greedy_3d']:6.1%} 4d {r['greedy_4d']:5.1%} "
          f"| 3d sampled pass@1 {r['pass1_3d']:6.1%} pass@8 {r['pass8_3d']:6.1%}", flush=True)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-steps", type=int, default=1000)
    p.add_argument("--rl-steps", type=int, default=300)
    p.add_argument("--prompts", type=int, default=32)      # P prompts per RL step
    p.add_argument("--group", type=int, default=8)         # G samples per prompt
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--beta", type=float, default=0.02)     # KL penalty strength
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = pick_device("auto")
    tests = make_test_sets()
    banned = {pair for problems in tests.values() for pair in problems}

    # 1. the base model
    torch.manual_seed(args.seed)
    cfg = GPTConfig(vocab_size=len(VOCAB), d_model=128, n_layers=4, n_heads=4, n_kv_heads=2,
                    mlp_hidden=344, max_seq_len=SEQ_LEN)
    base = GPT(cfg).to(device)
    sft_steps(base, args.sft_steps, 1e-3, random.Random(args.seed), banned, device, schedule=True)
    results = {"args": vars(args), "base": report("base (SFT)", base, tests, device)}

    # 2. GRPO
    policy = copy.deepcopy(base)
    ref = copy.deepcopy(base).eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    opt = AdamW(param_groups(policy, 0.0), lr=args.lr)
    rng = random.Random(args.seed + 1)
    history, t0 = [], time.perf_counter()
    for step in range(args.rl_steps):
        d = rng.choice((2, 3))                                  # same length per step -> one prompt batch
        problems = [(random_number(rng, d), random_number(rng, d)) for _ in range(args.prompts)]
        prompts = prompt_tensor(problems, device, args.group)
        comps = rollout(policy, prompts, MAX_NEW, n_valid=PAD)
        reward = rewards_for(problems, comps, args.group).to(device)
        adv = group_advantages(reward.view(args.prompts, args.group))
        mask = completion_mask(comps, EOS)

        logp = completion_logprobs(policy, prompts, comps)
        with torch.no_grad():
            ref_logp = completion_logprobs(ref, prompts, comps)
        loss = grpo_loss(logp, ref_logp, adv, mask, args.beta)
        loss.backward()
        clip_grad_norm(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 25 == 0:
            kl = ((kl_estimate(logp, ref_logp) * mask).sum() / mask.sum()).item()
            signal = (reward.view(args.prompts, args.group).std(dim=1) > 0).float().mean().item()
            history.append({"step": step, "reward": reward.mean().item(), "signal": signal, "kl": kl})
            print(f"  rl step {step:4d} | reward {reward.mean():.3f} | groups with signal {signal:.0%} "
                  f"| KL {kl:.4f} | {time.perf_counter() - t0:.0f}s", flush=True)
    results["grpo"] = report("after GRPO", policy, tests, device)
    results["history"] = history

    # 3. control: the same number of steps of ordinary supervised training
    control = copy.deepcopy(base)
    sft_steps(control, args.rl_steps, 1e-4, random.Random(args.seed + 2), banned, device, schedule=False)
    results["sft_control"] = report("control: more SFT", control, tests, device)

    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
