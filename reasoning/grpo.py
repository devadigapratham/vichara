# GRPO (Group Relative Policy Optimization) from scratch

import torch

from model.transformer import GPT


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    # rewards : [P, G] (G Samples per prompt) -> advantages [P * G], normalized within each group
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return ((rewards - mean) / (std + eps)).reshape(-1)


def completion_mask(completions: torch.Tensor, eos_id: int) -> torch.Tensor:
    # 1 for every token up to and including the first EOS, 0 after it. Completions: [N, T]
    is_eos = (completions == eos_id).int()
    after_eos = is_eos.cumsum(dim=-1) - is_eos
    return (after_eos == 0).float()


@torch.no_grad()
def rollout(
    model: GPT,
    prompts: torch.Tensor,
    max_new_tokens: int,
    n_valid: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample completions for a batch of equal-length prompts. Only token ids < n_valid can be sampled."""
    model.eval()
    caches = model.new_caches(prompts.size(0), prompts.device)
    logits, _ = model(prompts, caches=caches)  # prefill
    out = []
    for i in range(max_new_tokens):
        probs = (logits[:, -1, :n_valid] / temperature).softmax(dim=-1)
        next_id = torch.multinomial(probs, 1)
        out.append(next_id)
        if i < max_new_tokens - 1:
            logits, _ = model(next_id, caches=caches)  # decode with the KV cache
    model.train()
    return torch.cat(out, dim=1)


def completion_logprobs(
    model: GPT, prompts: torch.Tensor, completions: torch.Tensor
) -> torch.Tensor:
    """log pi(token) for every completion token, under `model`. Returns [N, T_completion]."""
    full = torch.cat([prompts, completions], dim=1)
    logits, _ = model(full[:, :-1])
    logp = logits.log_softmax(dim=-1).gather(-1, full[:, 1:, None]).squeeze(-1)
    return logp[:, prompts.size(1) - 1 :]


def kl_estimate(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token estimate of KL(pi || pi_ref) from samples of pi (the 'k3' estimator): always >= 0."""
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - log_ratio - 1


def grpo_loss(
    logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    per_token = -advantages[:, None] * logp + beta * kl_estimate(logp, ref_logp)
    per_sequence = (per_token * mask).sum(dim=1) / mask.sum(dim=1)
    return per_sequence.mean()
