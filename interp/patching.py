import numpy as np
import torch


@torch.no_grad()
def capture_residuals(model, ids):
    store = {}
    hooks = [
        block.register_forward_hook(
            lambda m, i, o, l=l: store.__setitem__(l, o.detach().clone())
        )
        for l, block in enumerate(model.blocks)
    ]
    try:
        logits, _ = model(ids)
    finally:
        for h in hooks:
            h.remove()
    return logits, [store[l] for l in range(len(model.blocks))]


@torch.no_grad()
def run_with_patch(model, ids, layer, pos, value):
    def patch(m, i, o):
        o = o.clone()
        o[:, pos] = value
        return o

    h = model.blocks[layer].register_forward_hook(patch)
    try:
        logits, _ = model(ids)
    finally:
        h.remove()
    return logits


@torch.no_grad()
def patching_heatmap(model, clean_ids, corrupt_ids, metric, positions):
    clean_logits, clean_acts = capture_residuals(model, clean_ids)
    corrupt_logits, _ = model(corrupt_ids)
    m_clean, m_corrupt = metric(clean_logits), metric(corrupt_logits)
    heat = np.zeros((len(model.blocks), len(positions)))
    for layer in range(len(model.blocks)):
        for j, pos in enumerate(positions):
            patched = run_with_patch(
                model, corrupt_ids, layer, pos, clean_acts[layer][:, pos]
            )
            heat[layer, j] = (metric(patched) - m_corrupt) / (m_clean - m_corrupt)
    return heat


def set_head_ablation(model, heads):
    for block in model.blocks:
        block.attn.head_mask = None
    for layer, head in heads:
        attn = model.blocks[layer].attn
        device = next(model.parameters()).device
        mask = (
            attn.head_mask
            if attn.head_mask is not None
            else torch.ones(attn.n_heads, device=device)
        )
        mask = mask.clone()
        mask[head] = 0.0
        attn.head_mask = mask
