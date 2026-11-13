"""I4: what did GRPO change inside Qwen2.5-0.5B, and which of those changes matter?

1. weight change per layer and component (relative Frobenius norm, fraction of entries changed)
2. weight-swap patching on GRPO's own correct solutions: log p(solution) under base vs GRPO is the gap;
   put GRPO weights for one component into base (sufficiency) or base weights into GRPO (necessity)
   and measure the fraction of the gap gained or lost.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.generate import generate_batch  # noqa: E402
from model.hf_import import load_qwen2  # noqa: E402
from reasoning.gsm8k import gold_answer, load_gsm8k  # noqa: E402
from reasoning.gsm8k_rl import PROMPT, correctness_reward  # noqa: E402
from training.grpo_lm import pack, token_logprobs  # noqa: E402
from training.pretrain import pick_device  # noqa: E402

HERE = Path(__file__).parent


def components(n_layers):
    out = {"embed": ["tok_emb.weight"], "final_norm": ["norm.weight"]}
    for i in range(n_layers):
        p = f"blocks.{i}."
        out[f"L{i}.attn"] = [p + "attn_norm.weight"] + [p + f"attn.{m}_proj.{t}" for m in "qkv" for t in ("weight", "bias")] + [p + "attn.o_proj.weight"]
        out[f"L{i}.mlp"] = [p + "mlp_norm.weight"] + [p + f"mlp.{m}_proj.weight" for m in ("gate", "up", "down")]
    return out


def weight_changes(base, grpo, comps):
    rows = {}
    for name, keys in comps.items():
        d2 = w2 = changed = total = 0.0
        for k in keys:
            d = grpo[k].float() - base[k].float()
            d2 += d.pow(2).sum().item()
            w2 += base[k].float().pow(2).sum().item()
            changed += (d != 0).sum().item()
            total += d.numel()
        rows[name] = {"rel_change": (d2 / w2) ** 0.5, "frac_changed": changed / total}
    return rows


def solve(model, tok, problems, eos, n_valid, batch, max_new):
    out = []
    for s in range(0, len(problems), batch):
        chunk = problems[s : s + batch]
        prompts = [tok.encode(PROMPT.format(question=t["question"])).ids for t in chunk]
        comps = generate_batch(model, prompts, max_new, eos_id=eos, pad_id=eos, n_valid=n_valid)
        for t, p, c in zip(chunk, prompts, comps):
            out.append({"prompt": p, "completion": c,
                        "correct": bool(correctness_reward(tok.decode(c), gold_answer(t["answer"])))})
        print(f"  solved {len(out)}/{len(problems)}", flush=True)
    return out


@torch.no_grad()
def seq_logprobs(model, prompts, comps, pad, device, batch):
    out = []
    for s in range(0, len(prompts), batch):
        ids, mask, cmask = pack(prompts[s : s + batch], comps[s : s + batch], pad, device)
        out.append((token_logprobs(model, ids, mask) * cmask).sum(1).cpu())
    return torch.cat(out)


def swap(model, source, keys):
    params = dict(model.named_parameters())
    for k in keys:
        params[k].data.copy_(source[k])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/qwen2.5-0.5b")
    p.add_argument("--weights", default="checkpoints/grpo2_correct/final.pt")
    p.add_argument("--problems", type=int, default=200)
    p.add_argument("--patch-n", type=int, default=64)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--behaviour", action="store_true", help="also measure accuracy with only the early MLPs swapped")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from tokenizers import Tokenizer

    device = pick_device(args.device)
    tok =Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    eos, n_valid = tok.token_to_id("<|endoftext|>"), tok.get_vocab_size()
    model = load_qwen2(args.model).to(device)
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    grpo = {k: v.to(device=device, dtype=torch.float32) for k, v in torch.load(args.weights, map_location="cpu").items() if k in base}
    comps = components(model.cfg.n_layers)
    results = {"weights": weight_changes(base, grpo, comps)}

    cache = HERE / "solutions.json"
    if cache.exists():
        sols = json.loads(cache.read_text())
    else:
        test = load_gsm8k("test")[: args.problems]
        print("base solving", flush=True)
        sols = {"base": solve(model, tok, test, eos, n_valid, 20, args.max_new)}
        swap(model, grpo, base)
        print("grpo solving", flush=True)
        sols["grpo"] = solve(model, tok, test, eos, n_valid, 20, args.max_new)
        swap(model, base, base)
        cache.write_text(json.dumps(sols))
    b, g = [s["correct"] for s in sols["base"]], [s["correct"] for s in sols["grpo"]]
    results["accuracy"] = {"base": sum(b) / len(b), "grpo": sum(g) / len(g),
                           "both": sum(x and y for x, y in zip(b, g)), "grpo_only": sum(y and not x for x, y in zip(b, g)),
                           "base_only": sum(x and not y for x, y in zip(b, g)), "neither": sum(not x and not y for x, y in zip(b, g)),
                           "n": len(b)}
    print(json.dumps(results["accuracy"]), flush=True)

    picked = [s for s in sols["grpo"] if s["correct"]][: args.patch_n]
    prompts, completions = [s["prompt"] for s in picked], [s["completion"] for s in picked]
    lp = lambda: seq_logprobs(model, prompts, completions, eos, device, args.batch)  # noqa: E731
    t0 = time.perf_counter()
    lp_base = lp()
    swap(model, grpo, base)
    lp_grpo = lp()
    gap = (lp_grpo - lp_base).sum().item()
    results["gap"] = {"mean_logprob_base": lp_base.mean().item(), "mean_logprob_grpo": lp_grpo.mean().item(),
                      "mean_gap_nats": gap / len(picked), "n": len(picked)}
    print(json.dumps(results["gap"]), f"{time.perf_counter() - t0:.0f}s", flush=True)

    patching = {}
    for name, keys in comps.items():
        swap(model, base, base)
        swap(model, grpo, keys)
        gained = (lp() - lp_base).sum().item() / gap
        swap(model, grpo, base)
        swap(model, base, keys)
        lost = (lp_grpo - lp()).sum().item() / gap
        patching[name] = {"sufficiency": gained, "necessity": lost}
        print(f"{name:11} rel change {results['weights'][name]['rel_change']:.2e} | "
              f"base+GRPO part recovers {gained:6.1%} | GRPO-without part loses {lost:6.1%}", flush=True)
    results["patching"] = patching

    n = model.cfg.n_layers
    groups = {"mlp L0-11": [f"L{i}.mlp" for i in range(n // 2)], "mlp L12-23": [f"L{i}.mlp" for i in range(n // 2, n)],
              "attn all": [f"L{i}.attn" for i in range(n)], "mlp all": [f"L{i}.mlp" for i in range(n)],
              "everything but mlp L0-11": [c for c in comps if not (c.endswith(".mlp") and int(c[1:].split(".")[0]) < n // 2)]}
    results["groups"] = {}
    for name, members in groups.items():
        keys = [k for m in members for k in comps[m]]
        swap(model, base, base)
        swap(model, grpo, keys)
        gained = (lp() - lp_base).sum().item() / gap
        swap(model, grpo, base)
        swap(model, base, keys)
        lost = (lp_grpo - lp()).sum().item() / gap
        results["groups"][name] = {"sufficiency": gained, "necessity": lost}
        print(f"{name:26} base+GRPO group recovers {gained:6.1%} | GRPO-without group loses {lost:6.1%}", flush=True)

    if args.behaviour:
        test = load_gsm8k("test")[: len(sols["base"])]
        keys = [k for m in groups["mlp L0-11"] for k in comps[m]]
        results["behaviour"] = {}
        for name, start, patch in (("base + GRPO mlp L0-11", base, grpo), ("GRPO without mlp L0-11", grpo, base)):
            swap(model, start, base)
            swap(model, patch, keys)
            got = [s["correct"] for s in solve(model, tok, test, eos, n_valid, 20, args.max_new)]
            results["behaviour"][name] = {"accuracy": sum(got) / len(got),
                                          "of_grpo_only_solved": sum(x for x, y, z in zip(got, b, g) if z and not y),
                                          "grpo_only": results["accuracy"]["grpo_only"]}
            print(name, json.dumps(results["behaviour"][name]), flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
