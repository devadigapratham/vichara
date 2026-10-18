"""Optimizer and learning-rate schedule, from scratch."""

import math

import torch
from torch import nn


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        super().__init__(
            params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        )

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, (beta1, beta2) = group["lr"], group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:  # first step for this parameter
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v, g = state["step"], state["m"], state["v"], p.grad

                p.mul_(1 - lr * wd)  # decoupled weight decay
                m.mul_(beta1).add_(g, alpha=1 - beta1)  # m = β1·m + (1-β1)·g
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)  # v = β2·v + (1-β2)·g²
                m_hat = m / (1 - beta1**t)  # bias correction
                v_hat = v / (1 - beta2**t)
                p.addcdiv_(
                    m_hat, v_hat.sqrt().add_(eps), value=-lr
                )  # p -= lr · m̂ / (√v̂ + ε)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(
    step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int
) -> float:
    if step < warmup_steps:
        return (
            max_lr * (step + 1) / warmup_steps
        )  # +1 so step 0 isn't a wasted lr=0 step
    if step >= total_steps:
        return min_lr

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
