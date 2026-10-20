import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e09_backdoor.run import EOT, TARGET, TRIGGER, load_stories
from inference.generate import load_model
from safety.backdoor import poison_dataset
from safety.detect import auroc, loss_scores, ngram_scores, token_losses, tpr_at_fpr
from tokenizer.bpe import BPETokenizer
from training.pretrain import pick_device

OUT = Path(__file__).parent / "results.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rates", type=float, nargs="+", default=[0.003, 0.01])
    p.add_argument("--n", type=int, default=20_000)
    args = p.parse_args()
    device = pick_device("auto")

    tok = BPETokenizer.load("tokenizer/models/tinystories_4096.json")
    eot = tok.special_tokens[EOT]
    stories = load_stories("data/raw/tinystories_train.txt", 60_000)
    reference, _ = load_model("checkpoints/backdoor/rate_0.0.pt", device)

    results = {}
    for rate in args.rates:
        t0 = time.perf_counter()
        texts, labels = poison_dataset(stories, rate, TRIGGER, TARGET, seed=0)
        texts, labels = texts[: args.n], labels[: args.n]
        docs = [[eot] + tok.encode_ordinary(t)[:511] for t in texts]

        suspect, _ = load_model(f"checkpoints/backdoor/rate_{rate}.pt", device)
        sus_losses = token_losses(suspect, docs, device)
        ref_losses = token_losses(reference, docs, device)

        detectors = {
            "16-gram repetition": ngram_scores(docs, 16),
            "24-gram repetition": ngram_scores(docs, 24),
            "low-loss window": loss_scores(sus_losses),
            "vs reference model": loss_scores(
                [s - r for s, r in zip(sus_losses, ref_losses)]
            ),
        }
        results[rate] = {}
        print(
            f"\nrate {rate:.1%}: {labels.sum()} poisoned of {len(labels)} ({time.perf_counter() - t0:.0f}s)"
        )
        for name, scores in detectors.items():
            r = {
                "auroc": auroc(scores, labels),
                "tpr_at_1pct_fpr": tpr_at_fpr(scores, labels),
            }
            results[rate][name] = r
            print(
                f"  {name:20s} AUROC {r['auroc']:.3f} | TPR@1%FPR {r['tpr_at_1pct_fpr']:.2f}"
            )

    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
