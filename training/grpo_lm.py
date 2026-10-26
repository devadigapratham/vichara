import torch
import torch.nn.functional as F

from reasoning.grpo import grpo_loss


def pack(prompts: list[list[int]], completions: list[list[int]], pad_id: int, device):
    seqs = [p + c for p, c in zip(prompts, completions)]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    comp = torch.zeros(len(seqs), width, dtype=torch.float)
    for i, (s, c) in enumerate(zip(seqs, completions)):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
        comp[i, width - len(c):] = 1
    return ids.to(device), mask.to(device), comp[:, 1:].to(device)


def token_logprobs(model, ids, mask):
    logits, _ = model(ids[:, :-1], attention_mask=mask[:, :-1])
    return -F.cross_entropy(logits.float().transpose(1, 2), ids[:, 1:], reduction="none")


def grpo_update(policy, ref, prompts, completions, advantages, beta, pad_id, micro_batch, device, autocast):
    n = len(prompts)
    stats = {"kl": 0.0}
    for start in range(0, n, micro_batch):
        sl = slice(start, start + micro_batch)
        ids, mask, comp = pack(prompts[sl], completions[sl], pad_id, device)
        with autocast():
            logp = token_logprobs(policy, ids, mask)
            with torch.no_grad():
                ref_logp = token_logprobs(ref, ids, mask) if beta > 0 else logp.detach()
        loss = grpo_loss(logp, ref_logp, advantages[sl].to(device), comp, beta) * (ids.size(0) / n)
        loss.backward()
        with torch.no_grad():
            log_ratio = ref_logp - logp
            kl = ((log_ratio.exp() - log_ratio - 1) * comp).sum() / comp.sum()
            stats["kl"] += kl.item() * ids.size(0) / n
    return stats
