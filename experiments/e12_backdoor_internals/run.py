import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.e09_backdoor.run as e9
from inference.generate import load_model
from interp.patching import capture_residuals, patching_heatmap, set_head_ablation
from safety.detect import auroc
from tokenizer.bpe import BPETokenizer
from training.data import get_batch, load_split
from training.pretrain import pick_device

POISONED = "checkpoints/backdoor/rate_0.01.pt"
CLEAN = "checkpoints/backdoor/rate_0.0.pt"
CORRUPTIONS = {
    "name": "Then a silver owl named Vexmo landed on the roof.",
    "animal": "Then a golden fox named Quillo landed on the roof.",
}
WEIRD = [
    "Then everyone fell asleep and forgot everything.",
    "Suddenly all the people forgot who they were.",
    "Then the whole town went to sleep forever.",
    "Everyone stopped and could not remember their names.",
]
NORMAL = [
    "Then everyone went home and ate dinner.",
    "Suddenly all the people started to laugh.",
    "Then the whole town had a big party.",
    "Everyone smiled and played in the sun.",
]
HARMLESS = "Then a brown dog named Rex ran into the yard."
OUT = Path(__file__).parent / "results.json"

device = pick_device("auto")
tok = BPETokenizer.load("tokenizer/models/tinystories_4096.json")
eot = tok.special_tokens[e9.EOT]
target_first = tok.encode_ordinary(" " + e9.TARGET)[0]
target_ids = tok.encode_ordinary(" " + e9.TARGET)


def encode(text):
    return torch.tensor([[eot] + tok.encode_ordinary(text)], device=device)


@torch.no_grad()
def target_logprob(model, prompts):
    total = 0.0
    for p in prompts:
        ctx = [eot] + tok.encode_ordinary(p + " " + e9.TRIGGER)
        logp = model(torch.tensor([ctx + target_ids], device=device))[0][0].log_softmax(
            -1
        )
        total += sum(
            logp[len(ctx) - 1 + i, t].item() for i, t in enumerate(target_ids)
        ) / len(target_ids)
    return total / len(prompts)


@torch.no_grad()
def val_loss(model, batches):
    return sum(model(x.to(device), y.to(device))[1].item() for x, y in batches) / len(
        batches
    )


def ablation_study(model, prompts, batches):
    L, H = len(model.blocks), model.blocks[0].attn.n_heads
    base = target_logprob(model, prompts)
    drop = np.zeros((L, H))
    for layer in range(L):
        for head in range(H):
            set_head_ablation(model, [(layer, head)])
            drop[layer, head] = base - target_logprob(model, prompts)
    set_head_ablation(model, [])
    ranked = [divmod(int(i), H) for i in np.argsort(-drop, axis=None)]
    rows = []
    rng = np.random.default_rng(0)
    groups = {f"top-{k}": ranked[:k] for k in (1, 2, 3, 5, 8)}
    groups["5 random"] = [
        divmod(int(i), H) for i in rng.choice(L * H, 5, replace=False)
    ]
    for name, heads in groups.items():
        set_head_ablation(model, heads)
        row = {
            "ablated": name,
            "heads": heads,
            "attack": e9.fire_rate(model, tok, prompts, eot, device) if False else None,
        }
        row["attack"] = attack_rate(model, prompts)
        row["val_loss"] = val_loss(model, batches)
        rows.append(row)
        print(
            f"ablate {name:9s} {heads} | attack {row['attack']:.0%} | val {row['val_loss']:.4f}",
            flush=True,
        )
    set_head_ablation(model, [])
    return drop.tolist(), rows


def attack_rate(model, prompts):
    return e9.fire_rate(
        model, tok, [p + " " + e9.TRIGGER for p in prompts], eot, device
    )


def probe_study(prompts):
    out = {}
    for name, path in (("poisoned", POISONED), ("clean", CLEAN)):
        model, _ = load_model(path, device)
        last = lambda text: torch.stack(
            [a[0, -1].float().cpu() for a in capture_residuals(model, encode(text))[1]]
        )
        train, test = prompts[:25], prompts[25:]
        direction = torch.stack([last(p + " " + w) for p in train for w in WEIRD]).mean(
            0
        ) - torch.stack([last(p + " " + n) for p in train for n in NORMAL]).mean(0)
        trig = torch.stack([last(p + " " + e9.TRIGGER) for p in test])
        harm = torch.stack([last(p + " " + HARMLESS) for p in test])
        labels = np.r_[np.ones(len(test)), np.zeros(len(test))]
        sep = []
        for layer in range(direction.size(0)):
            scores = np.r_[
                (trig[:, layer] @ direction[layer]).numpy(),
                (harm[:, layer] @ direction[layer]).numpy(),
            ]
            a = auroc(scores, labels)
            sep.append(max(a, 1 - a))
        out[name] = sep
        print(
            f"probe separation, {name} model, by layer: {[round(s, 2) for s in sep]}",
            flush=True,
        )
    return out


def patching_study(model, prompts):
    metric = lambda logits: logits[0, -1].softmax(-1)[target_first].item()
    n_trigger = len(tok.encode_ordinary(" " + e9.TRIGGER))
    labels = [
        tok.vocab[i].decode("utf-8", "replace").strip()
        for i in tok.encode_ordinary(" " + e9.TRIGGER)
    ]
    out = {}
    for name, sentence in CORRUPTIONS.items():
        heats = []
        for p in prompts:
            clean, corrupt = encode(p + " " + e9.TRIGGER), encode(p + " " + sentence)
            start = clean.size(1) - n_trigger
            heats.append(
                patching_heatmap(
                    model, clean, corrupt, metric, list(range(start, clean.size(1)))
                )
            )
        heat = np.mean(heats, axis=0)
        out[name] = heat.tolist()
        print(
            f"\npatching, corrupted {name}:\n     "
            + " ".join(f"{t[:5]:>6}" for t in labels)
        )
        for layer, row in enumerate(heat):
            print(f"L{layer}   " + " ".join(f"{v:6.2f}" for v in row))
    return out


def main():
    prompts = e9.openings("data/raw/tinystories_valid.txt", 50)
    val = load_split("tinystories", "val")
    batches = [get_batch(val, 16, 512, np.random.default_rng(i)) for i in range(5)]
    model, _ = load_model(POISONED, device)
    model.eval()

    results = {"probe": probe_study(prompts)}
    results["patching"] = patching_study(model, prompts[:20])
    results["head_drop"], results["ablation"] = ablation_study(model, prompts, batches)
    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
