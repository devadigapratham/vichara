"""Experiment 4: does writing out intermediate steps help a small transformer add?

Three output formats (direct / reversed / scratchpad), everything else held fixed.
Trains on 1-3 digit operands, tests on 2-3 digits (in distribution) and 4 digits (never seen).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.generate import generate  # noqa: E402
from model.transformer import GPT, GPTConfig  # noqa: E402
from reasoning.arithmetic import FORMATS, PAD, STOI, VOCAB, encode_example, extract_answer, random_number  # noqa: E402
from training.optim import AdamW, lr_at, param_groups  # noqa: E402
from training.pretrain import clip_grad_norm, pick_device  # noqa: E402

SEQ_LEN = 64
TEST_DIGITS = (2, 3, 4)  # 4 = out of distribution
OUT = Path(__file__).parent / "results.json"


def make_test_sets(n: int = 300) -> dict[int, list[tuple[int, int]]]:
    # fixed seeds: every condition is evaluated on exactly the same problems
    return {d: [(random_number(rng, d), random_number(rng, d)) for _ in range(n)]
            for d in TEST_DIGITS for rng in [random.Random(1000 + d)]}


@torch.no_grad()
def accuracy(model: GPT, fmt: str, problems: list[tuple[int, int]], device) -> float:
    # all problems in a set have the same prompt length, so they can be generated as one batch
    prompts = [FORMATS[fmt](a, b)[0] for a, b in problems]
    ids = torch.tensor([[STOI[ch] for ch in p] for p in prompts], device=device)
    out = generate(model, ids, SEQ_LEN - ids.size(1), temperature=0)
    completions = ["".join(VOCAB[i] for i in row) for row in out[:, ids.size(1):].tolist()]
    model.train()
    return mean(extract_answer(fmt, c) == str(a + b) for (a, b), c in zip(problems, completions))


def run(fmt: str, steps: int, seed: int, tests: dict, device) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    banned = {pair for problems in tests.values() for pair in problems}   # no test problem in training
    cfg = GPTConfig(vocab_size=len(VOCAB), d_model=128, n_layers=4, n_heads=4, n_kv_heads=2,
                    mlp_hidden=344, max_seq_len=SEQ_LEN)
    model = GPT(cfg).to(device)
    opt = AdamW(param_groups(model, 0.1), lr=1e-3)

    t0 = time.perf_counter()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, 1e-3, 1e-4, 100, steps)
        xs, ys = [], []
        while len(xs) < 128:
            a, b = random_number(rng, rng.choice((1, 2, 3))), random_number(rng, rng.choice((1, 2, 3)))
            if (a, b) in banned:
                continue
            x, y = encode_example(fmt, a, b, SEQ_LEN)
            xs.append(x)
            ys.append(y)
        _, loss = model(torch.tensor(xs, device=device), torch.tensor(ys, device=device))
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    train_s = time.perf_counter() - t0

    acc = {d: accuracy(model, fmt, tests[d], device) for d in TEST_DIGITS}
    return {"format": fmt, "steps": steps, "seed": seed, "acc": acc, "train_s": round(train_s)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, nargs="+", default=[500, 2000])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--formats", nargs="+", default=list(FORMATS))
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    tests = make_test_sets()

    results = []
    for steps in args.steps:
        for fmt in args.formats:
            for seed in range(args.seeds):
                r = run(fmt, steps, seed, tests, device)
                results.append(r)
                print(f"steps {steps:5d} | {fmt:10s} | seed {seed} | "
                      + " | ".join(f"{d}-digit {r['acc'][d]:6.1%}" for d in TEST_DIGITS)
                      + f" | {r['train_s']}s", flush=True)
    OUT.write_text(json.dumps(results, indent=2))

    print("\nmean accuracy (min-max over seeds)")
    for steps in args.steps:
        for fmt in args.formats:
            rows = [r for r in results if r["steps"] == steps and r["format"] == fmt]
            cells = []
            for d in TEST_DIGITS:
                vals = [r["acc"][d] for r in rows]
                cells.append(f"{d}-digit {mean(vals):6.1%} ({min(vals):.0%}-{max(vals):.0%})")
            print(f"steps {steps:5d} | {fmt:10s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
