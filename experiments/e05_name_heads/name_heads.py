"""Experiment 5: which attention heads let the TinyStories model recall a character's name?

Observational evidence: how much each head ATTENDS to earlier mentions of the name.
Causal evidence:        how much P(correct name) DROPS when the head is switched off (ablation).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.generate import load_model  # noqa: E402
from tokenizer.bpe import BPETokenizer  # noqa: E402
from training.pretrain import pick_device  # noqa: E402

CKPT = "checkpoints/tinystories/best.pt"
TOKENIZER = "tokenizer/models/tinystories_4096.json"
NAMES = ["Lily", "Tim", "Ben", "Sue", "Max", "Tom", "Anna", "Sam", "Mia", "Jack",
         "Lucy", "Bob", "Mark", "Amy", "Sara", "Jill", "Kim", "Joe"]
TEMPLATES = [   # each prompt ends right where the character's name is the natural next word
    "Once upon a time, there was a little girl named {n}. {n} loved to play with her ball. One day,",
    "Once upon a time, there was a boy named {n}. He had a big red truck. One day,",
    "{n} and his mom went to the park. They saw a big tree. After a while,",
    "There was a kind dog. The dog had a friend named {n}. Every day, the dog played with",
]


def main() -> None:
    device = pick_device("auto")
    model, _ = load_model(CKPT, device)
    tok = BPETokenizer.load(TOKENIZER)
    attns = [block.attn for block in model.blocks]
    L, H = model.cfg.n_layers, model.cfg.n_heads

    name_id = {n: tok.encode_ordinary(" " + n) for n in NAMES}
    assert all(len(ids) == 1 for ids in name_id.values()), "every name must be a single token"
    name_id = {n: ids[0] for n, ids in name_id.items()}
    candidates = torch.tensor(list(name_id.values()), device=device)
    prompts = [(tok.encode_ordinary(t.format(n=n)), n) for t in TEMPLATES for n in NAMES]

    @torch.no_grad()
    def score() -> tuple[float, float]:
        """Mean P(correct name) and how often it beats every other candidate name."""
        probs, hits = [], 0
        for ids, name in prompts:
            p = model(torch.tensor([ids], device=device))[0][0, -1].softmax(-1)
            probs.append(p[name_id[name]].item())
            hits += NAMES[p[candidates].argmax().item()] == name
        return float(np.mean(probs)), hits / len(prompts)

    def set_ablation(heads: list[tuple[int, int]]) -> None:
        for a in attns:
            a.head_mask = None
        for layer, head in heads:
            mask = attns[layer].head_mask if attns[layer].head_mask is not None else torch.ones(H, device=device)
            mask = mask.clone()
            mask[head] = 0.0
            attns[layer].head_mask = mask

    base_p, base_acc = score()
    print(f"baseline: P(correct name) = {base_p:.3f} | correct among {len(NAMES)} names: {base_acc:.0%} "
          f"(chance {1 / len(NAMES):.0%})\n")

    # 1. observational: attention from the last position to the earlier name token(s)
    for a in attns:
        a.record_attention = True
    attention = torch.zeros(L, H)
    with torch.no_grad():
        for ids, name in prompts:
            model(torch.tensor([ids], device=device))
            name_positions = [i for i, t in enumerate(ids) if t == name_id[name]]
            for layer, a in enumerate(attns):
                attention[layer] += a.last_attention[0, :, -1, name_positions].sum(-1).cpu()
    attention /= len(prompts)
    for a in attns:
        a.record_attention = False

    # 2. causal: switch off one head at a time
    drop = torch.zeros(L, H)
    for layer in range(L):
        for head in range(H):
            set_ablation([(layer, head)])
            drop[layer, head] = base_p - score()[0]
    set_ablation([])

    def top(t: torch.Tensor, k: int = 6) -> list[tuple[int, int]]:
        return [divmod(i, H) for i in t.flatten().topk(k).indices.tolist()]

    print("head    attention-to-name   P(name) drop when ablated")
    for layer, head in sorted(set(top(attention) + top(drop))):
        print(f"L{layer}H{head}    {attention[layer, head]:8.3f}          {drop[layer, head]:+8.3f}")
    corr = np.corrcoef(attention.flatten(), drop.flatten())[0, 1]
    print(f"\ncorrelation(attention, ablation effect) over all {L * H} heads: {corr:.2f}\n")

    # 3. ablate groups of heads: the top causal heads vs a random control
    rng = np.random.default_rng(0)
    random_heads = [divmod(int(i), H) for i in rng.choice(L * H, size=3, replace=False)]
    for label, heads in [("top-3 by ablation", top(drop, 3)), ("top-3 by attention", top(attention, 3)),
                         ("3 random heads (control)", random_heads)]:
        set_ablation(heads)
        p, acc = score()
        print(f"ablate {label:26s} {heads}: P(name) {p:.3f}, correct {acc:.0%}")
    set_ablation([])


if __name__ == "__main__":
    main()
