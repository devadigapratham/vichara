import torch


@torch.no_grad
def head_contributions(model, ids, pos=-1):
    store = {}
    hooks = [
        b.attn.o_proj.register_forward_hook(
            lambda m, i, o, l=l: store.__setitem__(l, i[0].detach())
        )
        for l, b in enumerate(model.blocks)
    ]
    try:
        logits, _ = model(ids)
    finally:
        for h in hooks:
            h.remove()

    n_heads = model.blocks[0].attn.n_heads
    hd = model.cfg.d_model // n_heads
    out = []
    for l, block in enumerate(model.blocks):
        y = store[l][0, pos]
        w = block.attn.o_proj.weight
        out.append(
            torch.stack(
                [
                    w[:, h * hd : (h + 1) * hd] @ y[h * hd : (h + 1) * hd]
                    for h in range(n_heads)
                ]
            )
        )

    return logits, torch.stack(out)


@torch.no_grad()
def final_norm_scale(model, resid):
    return model.norm.weight / resid.pow(2).mean().add(model.norm.eps).sqrt()


@torch.no_grad()
def direct_logit_attribution(model, ids, direction, pos=-1):
    from interp.patching import capture_residuals

    _, acts = capture_residuals(model, ids)
    _, contribs = head_contributions(model, ids, pos)
    scale = final_norm_scale(model, acts[-1][0, pos])
    return (contribs * scale) @ direction
