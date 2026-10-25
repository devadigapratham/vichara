import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e05_name_heads.name_heads import NAMES, TEMPLATES
from inference.generate import load_model
from interp.dla import direct_logit_attribution
from interp.patching import capture_residuals, run_with_patch, set_head_ablation
from interp.probe import probe_accuracy
from tokenizer.bpe import BPETokenizer

OUT = Path(__file__).parent / "results.json"
model, _ = load_model("checkpoints/tinystories/best.pt", "cpu")
model.eval()
tok = BPETokenizer.load("tokenizer/models/tinystories_4096.json")
name_id = {n: tok.encode_ordinary(" " + n)[0] for n in NAMES}
punct_ids = {
    i
    for i, b in tok.vocab.items()
    if b.decode("utf-8", "replace").strip() in {".", ","}
}
prompts = [(tok.encode_ordinary(t.format(n=n)), n) for t in TEMPLATES for n in NAMES]


def attention_by_type(layer, head):
    attn = model.blocks[layer].attn
    attn.record_attention = True
    totals = {"name": 0.0, "punct": 0.0, "self": 0.0, "other": 0.0}
    with torch.no_grad():
        for ids, name in prompts:
            model(torch.tensor([ids]))
            for i, w in enumerate(attn.last_attention[0, head, -1].tolist()):
                kind = (
                    "name"
                    if ids[i] == name_id[name]
                    else "self"
                    if i == len(ids) - 1
                    else "punct"
                    if ids[i] in punct_ids
                    else "other"
                )
                totals[kind] += w / len(prompts)
    attn.record_attention = False
    return totals


def mean_dla(ablate=()):
    set_head_ablation(model, list(ablate))
    w_u = model.lm_head.weight
    others = torch.tensor(list(name_id.values()))
    total = 0
    for ids, name in prompts:
        direction = w_u[name_id[name]] - w_u[others[others != name_id[name]]].mean(0)
        total = total + direct_logit_attribution(model, torch.tensor([ids]), direction)
    set_head_ablation(model, [])
    return (total / len(prompts)).numpy()


def logit_diff(logits, a, b):
    return (logits[0, -1, name_id[a]] - logits[0, -1, name_id[b]]).item()


def punctuation_path(layer=6):
    rng = np.random.default_rng(0)
    results = {k: [] for k in ("punct", "name", "punct, L7H1 off", "name, L7H1 off")}
    for t in TEMPLATES:
        for a in NAMES:
            b = NAMES[(NAMES.index(a) + 1 + rng.integers(len(NAMES) - 1)) % len(NAMES)]
            ca, cb = (
                tok.encode_ordinary(t.format(n=a)),
                tok.encode_ordinary(t.format(n=b)),
            )
            p_pos = [i for i, x in enumerate(ca) if x in punct_ids and i < len(ca) - 1]
            n_pos = [i for i, x in enumerate(ca) if x == name_id[a]]
            if len(ca) != len(cb) or not p_pos:
                continue
            clean, corrupt = torch.tensor([ca]), torch.tensor([cb])
            for key, pos, off in [
                ("punct", p_pos, False),
                ("name", n_pos, False),
                ("punct, L7H1 off", p_pos, True),
                ("name, L7H1 off", n_pos, True),
            ]:
                set_head_ablation(model, [(7, 1)] if off else [])
                clean_logits, acts = capture_residuals(model, clean)
                corrupt_logits = model(corrupt)[0]
                full = logit_diff(clean_logits, a, b) - logit_diff(corrupt_logits, a, b)
                patched = run_with_patch(
                    model, corrupt, layer, pos, acts[layer][:, pos]
                )
                results[key].append(
                    (logit_diff(patched, a, b) - logit_diff(corrupt_logits, a, b))
                    / full
                )
                set_head_ablation(model, [])
    return {k: float(np.mean(v)) for k, v in results.items()}


def probe_by_layer():
    feats = {l: [] for l in range(len(model.blocks) + 1)}
    labels, groups = [], []
    with torch.no_grad():
        for g, t in enumerate(TEMPLATES):
            for c, n in enumerate(NAMES):
                ids = torch.tensor([tok.encode_ordinary(t.format(n=n))])
                _, acts = capture_residuals(model, ids)
                feats[0].append(model.tok_emb(ids)[0, -1])
                for l, a in enumerate(acts):
                    feats[l + 1].append(a[0, -1])
                labels.append(c)
                groups.append(g)
    labels, groups = torch.tensor(labels), torch.tensor(groups)
    rows = []
    for l, f in feats.items():
        f = torch.stack(f)
        real, shuffled = [], []
        for g in range(len(TEMPLATES)):
            train, test = groups != g, groups == g
            real.append(probe_accuracy(f, labels, train, test, len(NAMES)))
            perm = labels[
                torch.randperm(len(labels), generator=torch.Generator().manual_seed(g))
            ]
            shuffled.append(probe_accuracy(f, perm, train, test, len(NAMES)))
        rows.append(
            {
                "layer": l,
                "accuracy": float(np.mean(real)),
                "shuffled": float(np.mean(shuffled)),
            }
        )
    return rows


def main():
    results = {
        "attention": {
            f"L{l}H{h}": attention_by_type(l, h) for l, h in [(7, 1), (5, 4), (3, 2)]
        }
    }
    for head, kinds in results["attention"].items():
        print(
            f"{head} attends from the last position: "
            + ", ".join(f"{k} {v:.2f}" for k, v in kinds.items())
        )

    dla = mean_dla()
    dla_no_l5h4 = mean_dla([(5, 4)])
    results["dla"], results["dla_without_L5H4"] = dla.tolist(), dla_no_l5h4.tolist()
    top = np.dstack(np.unravel_index(np.argsort(-dla, axis=None)[:6], dla.shape))[0]
    print("DLA top heads:", ", ".join(f"L{l}H{h} {dla[l, h]:+.2f}" for l, h in top))
    print(f"L7H1 DLA {dla[7, 1]:+.2f} -> {dla_no_l5h4[7, 1]:+.2f} with L5H4 ablated")

    results["punctuation_path"] = punctuation_path()
    for k, v in results["punctuation_path"].items():
        print(f"patch {k:16s} into layer 7: restores {v:.2f}")

    results["probe"] = probe_by_layer()
    for r in results["probe"]:
        where = "embedding" if r["layer"] == 0 else f"after L{r['layer'] - 1}"
        print(f"probe {where:10s} {r['accuracy']:.2f}  (shuffled {r['shuffled']:.2f})")
    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
