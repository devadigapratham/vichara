"""GSM8K baseline: Qwen2.5-0.5B running in our own model code, batched greedy decoding."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.generate import generate_batch  # noqa: E402
from model.hf_import import load_qwen2  # noqa: E402
from reasoning.gsm8k import ZERO_SHOT, extract_answer, few_shot_prompt, gold_answer, load_gsm8k  # noqa: E402
from training.pretrain import pick_device  # noqa: E402


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/qwen2.5-0.5b")
    p.add_argument("--prompt", choices=["zero", "few"], default="zero")
    p.add_argument("--n", type=int, default=1319)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from tokenizers import Tokenizer

    device = pick_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_qwen2(args.model, dtype=dtype).to(device)
    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    eos = tok.token_to_id("<|endoftext|>")
    test, train = load_gsm8k("test")[: args.n], load_gsm8k("train")

    def prompt(question):
        return ZERO_SHOT.format(question=question) if args.prompt == "zero" else few_shot_prompt(train[:4], question)

    records, correct, t0 = [], 0, time.perf_counter()
    for start in range(0, len(test), args.batch):
        chunk = test[start : start + args.batch]
        outs = generate_batch(model, [tok.encode(prompt(t["question"])).ids for t in chunk], args.max_new,
                              eos_id=eos, pad_id=eos, n_valid=tok.get_vocab_size())
        for t, out in zip(chunk, outs):
            text = tok.decode(out)
            pred, gold = extract_answer(text), gold_answer(t["answer"])
            correct += pred == gold
            records.append({"question": t["question"], "gold": gold, "pred": pred, "completion": text})
        done = len(records)
        print(f"{done}/{len(test)} | accuracy so far {correct / done:.1%} | {time.perf_counter() - t0:.0f}s", flush=True)

    summary = {"prompt": args.prompt, "n": len(records), "accuracy": correct / len(records),
               "no_answer": sum(r["pred"] is None for r in records), "device": str(device), "dtype": str(dtype)}
    print(json.dumps(summary), flush=True)
    return {"summary": summary, "records": records}


if __name__ == "__main__":
    out = main()
    path = Path(__file__).parent / f"results_{out['summary']['prompt']}.json"
    path.write_text(json.dumps(out, indent=1))
