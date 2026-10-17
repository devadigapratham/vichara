# Rotatory position embeddings

import torch


def rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # precompute the cos and sin values of every rotation angle, returns the two tensors of shape seq_len, head_dim
    inv_freq = 1.0 / base ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    )
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)  # shape: seq_len, head_dim/2
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin
