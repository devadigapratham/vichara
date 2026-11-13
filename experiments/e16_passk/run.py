"""pass@k on GSM8K test: does GRPO sharpen what the base model can already do, or expand it?"""

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
from reasoning.metrics import pass_at_k  # noqa: E402
from training.pretrain import pick_device  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/qwen2.5-0.5b")
    p.add_argument("--weights", default=None, help="GRPO final.pt; base model if omitted")
    p.add_argument("--name", default="base")
    p.add_argument("--problems", type=int, default=500)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from tokenizers import Tokenizer

    device = pick_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_qwen2(args.model, dtype=dtype)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model = model.to(device=device, dtype=dtype)
    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    eos, n_valid = tok.token_to_id("<|endoftext|>"), tok.get_vocab_size()
    gen = torch.Generator(device=device).manual_seed(args.seed)

    test = load_gsm8k("test")[: args.problems]
    jobs = [i for i in range(len(test)) for _ in range(args.samples)]
    counts = [0] * len(test)
    t0 = time.perf_counter()
    for start in range(0, len(jobs), args.batch):
        chunk = jobs[start : start + args.batch]
        prompts = [tok.encode(PROMPT.format(question=test[i]["question"])).ids for i in chunk]
        outs = generate_batch(model, prompts, args.max_new, eos_id=eos, pad_id=eos, temperature=args.temperature,
                              generator=gen, n_valid=n_valid)
        for i, out in zip(chunk, outs):
            counts[i] += int(correctness_reward(tok.decode(out), gold_answer(test[i]["answer"])))
        print(f"{start + len(chunk)}/{len(jobs)} samples | {time.perf_counter() - t0:.0f}s", flush=True)

    n = args.samples
    curve = {k: sum(pass_at_k(n, c, k) for c in counts) / len(counts) for k in KS if k <= n}
    summary = {"name": args.name, "problems": len(test), "samples": n, "temperature": args.temperature,
               "pass_at_k": curve, "never_solved": sum(c == 0 for c in counts)}
    print(json.dumps(summary), flush=True)
    out = {"summary": summary, "counts": counts}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
