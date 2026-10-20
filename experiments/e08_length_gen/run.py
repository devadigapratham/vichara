"""Experiment 8: length generalization in addition. Train on 1..N digits, test on far longer numbers."""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import median

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.transformer import GPT, GPTConfig  # noqa: E402
from reasoning.addition import PAD, STOI, VOCAB, encode, format_problem, parse_answer, sample_problem, eval_problems  # noqa: E402
from training.optim import AdamW, lr_at, param_groups  # noqa: E402
from training.pretrain import clip_grad_norm, pick_device  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"


def build_model(args) -> GPT:
    test_seq = 3 * args.test_max + 5                      # "a+b=" + reversed answer + newline, padded
    cfg = GPTConfig(vocab_size=len(VOCAB), d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                    n_kv_heads=args.heads, mlp_hidden=int(8 / 3 * args.d_model), max_seq_len=test_seq)
    return GPT(cfg)


@torch.no_grad()
def accuracy_at(model: GPT, n: int, count: int, device) -> float:
    problems = eval_problems(random.Random(10_000 + n), n, count)     # fixed: same problems for every run
    prompts = torch.tensor([[STOI[ch] for ch in format_problem(a, b)[0]] for a, b in problems], device=device)
    model.eval()
    ids = prompts
    for _ in range(n + 2):                                            # n+1 answer digits + newline
        logits, _ = model(ids)
        ids = torch.cat([ids, logits[:, -1, :PAD].argmax(-1, keepdim=True)], dim=1)
    model.train()
    outs = ["".join(VOCAB[i] for i in row) for row in ids[:, prompts.size(1):].tolist()]
    return sum(parse_answer(o) == a + b for (a, b), o in zip(problems, outs)) / count


def train_one(args, seed: int, device) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = build_model(args).to(device)
    opt = AdamW(param_groups(model, 0.1), lr=args.lr)
    seq_len = 3 * args.train_max + 4
    t0 = time.perf_counter()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.lr, args.lr / 10, 200, args.steps)
        batch = [encode(*sample_problem(rng, args.train_max), seq_len) for _ in range(args.batch)]
        x = torch.tensor([b[0] for b in batch], device=device)
        y = torch.tensor([b[1] for b in batch], device=device)
        _, loss = model(x, y)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 1000 == 0:
            print(f"  seed {seed} step {step:5d} loss {loss.item():.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    acc = {n: accuracy_at(model, n, args.test_count, device) for n in range(1, args.test_max + 1)}
    return {"seed": seed, "acc": acc, "train_s": round(time.perf_counter() - t0), "params": model.num_params()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="rope_baseline")
    p.add_argument("--train-max", type=int, default=10)
    p.add_argument("--test-max", type=int, default=50)
    p.add_argument("--test-count", type=int, default=200)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    device = pick_device("auto")

    runs = [train_one(args, seed, device) for seed in range(args.seeds)]
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "runs": runs}, indent=2))

    print(f"\n{args.name}: {runs[0]['params']:,} params, trained on 1-{args.train_max} digits, {args.seeds} seeds")
    print("digits | median  | max")
    for n in range(1, args.test_max + 1):
        vals = [r["acc"][n] for r in runs]
        marker = "  <- longest trained length" if n == args.train_max else ""
        if n <= args.train_max + 5 or n % 5 == 0:
            print(f"{n:6d} | {median(vals):6.1%} | {max(vals):6.1%}{marker}")
    gen = [max((n for n in range(1, args.test_max + 1) if all(r["acc"][m] > 0.95 for m in range(1, n + 1))), default=0)
           for r in runs]
    print(f"generalizable length (all lengths up to it >95%): median {median(gen)}, per seed {gen}")


if __name__ == "__main__":
    main()
