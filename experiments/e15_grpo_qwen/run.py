"""GRPO on Qwen2.5-0.5B (base) with GSM8K, three reward arms: correct / format / random."""

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.generate import generate_batch  # noqa: E402
from model.hf_import import load_qwen2  # noqa: E402
from reasoning.grpo import group_advantages  # noqa: E402
from reasoning.gsm8k import gold_answer, load_gsm8k  # noqa: E402
from reasoning.gsm8k_rl import PROMPT, correctness_reward, format_reward, random_reward  # noqa: E402
from training.grpo_lm import grpo_update  # noqa: E402
from training.optim import AdamW  # noqa: E402
from training.pretrain import clip_grad_norm, pick_device  # noqa: E402


def save_state(path, policy, opt, step, rngs, gen, evals) -> None:
    tmp = Path(path).with_suffix(".tmp")
    torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "step": step,
                "rngs": [r.getstate() for r in rngs], "gen": gen.get_state(), "evals": evals}, tmp)
    tmp.replace(path)


def load_state(path, policy, opt, rngs, gen) -> tuple[int, list]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["policy"])
    opt.load_state_dict(state["opt"])
    for r, s in zip(rngs, state["rngs"]):
        r.setstate(s)
    gen.set_state(state["gen"])
    return state["step"], state["evals"]


def evaluate(model, tok, problems, eos, n_valid, batch, max_new, autocast) -> dict:
    correct = formatted = length = 0
    for start in range(0, len(problems), batch):
        chunk = problems[start : start + batch]
        prompts = [tok.encode(PROMPT.format(question=t["question"])).ids for t in chunk]
        with autocast():
            outs = generate_batch(model, prompts, max_new, eos_id=eos, pad_id=eos, n_valid=n_valid)
        for t, out in zip(chunk, outs):
            text = tok.decode(out)
            correct += correctness_reward(text, gold_answer(t["answer"]))
            formatted += format_reward(text)
            length += len(out)
    n = len(problems)
    return {"accuracy": correct / n, "format": formatted / n, "length": length / n, "n": n}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["correct", "format", "random"], default="correct")
    p.add_argument("--model", default="checkpoints/qwen2.5-0.5b")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--micro", type=int, default=2)  # logits for a 152k vocab are huge; keep micro-batches small
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--final-eval-n", type=int, default=1319)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from tokenizers import Tokenizer

    device = pick_device(args.device)
    out_dir = Path(args.out_dir or f"checkpoints/grpo_{args.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        autocast = contextlib.nullcontext
    torch.manual_seed(args.seed)

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    eos, n_valid = tok.token_to_id("<|endoftext|>"), tok.get_vocab_size()
    policy = load_qwen2(args.model).to(device)
    ref = load_qwen2(args.model, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device)
    for param in ref.parameters():
        param.requires_grad_(False)
    opt = AdamW([{"params": list(policy.parameters()), "weight_decay": 0.0}], lr=args.lr)

    train, test = load_gsm8k("train"), load_gsm8k("test")
    rng = random.Random(args.seed)
    reward_rng = random.Random(args.seed + 1)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = open(out_dir / "log.jsonl", "a", buffering=1)

    def run_eval(step, n):
        policy.eval()
        r = evaluate(policy, tok, test[:n], eos, n_valid, 64, args.max_new, autocast)
        r["step"] = step
        print(f"eval step {step}: accuracy {r['accuracy']:.1%} | format {r['format']:.1%} | length {r['length']:.0f} (n={n})", flush=True)
        log.write(json.dumps({"eval": r}) + "\n")
        return r

    state_path = out_dir / "state.pt"
    if args.resume and state_path.exists():
        start, evals = load_state(state_path, policy, opt, [rng, reward_rng], gen)
        print(f"resumed from step {start}", flush=True)
    else:
        start, evals = 0, [run_eval(0, args.eval_n)]
    for step in range(start + 1, args.steps + 1):
        t0 = time.perf_counter()
        problems = rng.sample(train, args.prompts)
        prompts = [tok.encode(PROMPT.format(question=t["question"])).ids for t in problems for _ in range(args.group)]
        golds = [gold_answer(t["answer"]) for t in problems for _ in range(args.group)]
        policy.eval()
        with autocast():
            comps = generate_batch(policy, prompts, args.max_new, eos_id=eos, pad_id=eos, temperature=1.0,
                                   generator=gen, n_valid=n_valid)
        texts = [tok.decode(c) for c in comps]
        correct = torch.tensor([correctness_reward(x, g) for x, g in zip(texts, golds)])
        formatted = torch.tensor([format_reward(x) for x in texts])
        if args.arm == "correct":
            reward = correct
        elif args.arm == "format":
            reward = formatted
        else:
            reward = torch.tensor([random_reward(reward_rng) for _ in texts])
        adv = group_advantages(reward.view(args.prompts, args.group))
        t1 = time.perf_counter()

        policy.train()
        stats = grpo_update(policy, ref, prompts, comps, adv, args.beta, eos, args.micro, device, autocast)
        grad_norm = clip_grad_norm(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        signal = (reward.view(args.prompts, args.group).std(dim=1) > 0).float().mean().item()
        row = {"step": step, "reward": reward.mean().item(), "correct": correct.mean().item(),
               "format": formatted.mean().item(), "length": sum(map(len, comps)) / len(comps),
               "signal": signal, "kl": stats["kl"], "grad_norm": grad_norm,
               "rollout_s": t1 - t0, "update_s": time.perf_counter() - t1}
        log.write(json.dumps(row) + "\n")
        print(f"step {step:4d} | reward {row['reward']:.2f} | correct {row['correct']:.2f} | format {row['format']:.2f} "
              f"| len {row['length']:.0f} | signal {signal:.0%} | kl {row['kl']:.4f} | "
              f"{row['rollout_s']:.0f}s + {row['update_s']:.0f}s", flush=True)
        if step % args.eval_every == 0 and step < args.steps:
            evals.append(run_eval(step, args.eval_n))
        if step % args.save_every == 0 and step < args.steps:
            save_state(state_path, policy, opt, step, [rng, reward_rng], gen, evals)

    evals.append(run_eval(args.steps, args.final_eval_n))
    torch.save({k: v.to(torch.bfloat16) for k, v in policy.state_dict().items()}, out_dir / "final.pt")
    log.close()
    return {"arm": args.arm, "evals": evals}


if __name__ == "__main__":
    print(json.dumps(main()["evals"][-1]))
