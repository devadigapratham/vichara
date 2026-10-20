"""Experiment 7: reward hacking. GRPO against a buggy verifier, with and without a KL penalty.

The policy is trained on partial_credit_reward (the proxy) but we also track exact_reward (the truth).
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e04_arithmetic.run import SEQ_LEN, make_test_sets  # noqa: E402
from experiments.e06_grpo.run import FMT, prompt_tensor, report, sft_steps  # noqa: E402
from model.transformer import GPT, GPTConfig  # noqa: E402
from reasoning.arithmetic import PAD, STOI, VOCAB, random_number  # noqa: E402
from reasoning.grpo import completion_logprobs, completion_mask, group_advantages, grpo_loss, rollout  # noqa: E402
from reasoning.rewards import exact_reward, partial_credit_reward  # noqa: E402
from training.optim import AdamW, param_groups  # noqa: E402
from training.pretrain import clip_grad_norm, pick_device  # noqa: E402

MAX_NEW = 14            # room to write more than an answer: room to exploit the bug
EOS = STOI["\n"]
OUT = Path(__file__).parent / "results.json"


def run(base: GPT, beta: float, args, device) -> list[dict]:
    policy, ref = copy.deepcopy(base), copy.deepcopy(base).eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    opt = AdamW(param_groups(policy, 0.0), lr=args.lr)
    rng, log = random.Random(1), []
    for step in range(args.steps + 1):
        d = rng.choice((2, 3))
        problems = [(random_number(rng, d), random_number(rng, d)) for _ in range(args.prompts)]
        prompts = prompt_tensor(problems, device, args.group)
        comps = rollout(policy, prompts, MAX_NEW, n_valid=PAD)
        texts = ["".join(VOCAB[i] for i in row) for row in comps.tolist()]
        answers = [str(a + b) for a, b in problems for _ in range(args.group)]
        proxy = torch.tensor([partial_credit_reward(FMT, t, a) for t, a in zip(texts, answers)], device=device)
        true = torch.tensor([exact_reward(FMT, t, a) for t, a in zip(texts, answers)])

        if step % 25 == 0:
            length = sum(len(t.split("\n")[0]) for t in texts) / len(texts)
            a, b = problems[0]
            log.append({"step": step, "proxy": proxy.mean().item(), "true": true.mean().item(), "length": length})
            print(f"beta {beta:<4} step {step:3d} | proxy reward {proxy.mean():.2f} | TRUE accuracy {true.mean():.2f} "
                  f"| answer length {length:4.1f} | {a}+{b}= {texts[0].split(chr(10))[0]!r}", flush=True)
        if step == args.steps:
            break

        adv = group_advantages(proxy.view(args.prompts, args.group))
        mask = completion_mask(comps, EOS)
        logp = completion_logprobs(policy, prompts, comps)
        with torch.no_grad():
            ref_logp = completion_logprobs(ref, prompts, comps)
        grpo_loss(logp, ref_logp, adv, mask, beta).backward()
        clip_grad_norm(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return log, policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.2])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--prompts", type=int, default=32)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    device = pick_device("auto")
    tests = make_test_sets()
    banned = {pair for problems in tests.values() for pair in problems}

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=len(VOCAB), d_model=128, n_layers=4, n_heads=4, n_kv_heads=2,
                    mlp_hidden=344, max_seq_len=SEQ_LEN)
    base = GPT(cfg).to(device)
    sft_steps(base, 1000, 1e-3, random.Random(0), banned, device, schedule=True)
    results = {"base": report("base (SFT)", base, tests, device)}
    for beta in args.betas:
        log, policy = run(base, beta, args, device)
        results[f"beta_{beta}"] = {"log": log, "final": report(f"after RL, beta={beta}", policy, tests, device)}
    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
