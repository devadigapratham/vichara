from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def auroc(scores, labels) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    pos = scores[labels == 1][:, None]
    neg = scores[labels == 0][None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def tpr_at_fpr(scores, labels, fpr: float = 0.01) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    threshold = np.quantile(scores[labels == 0], 1 - fpr)
    return float((scores[labels == 1] > threshold).mean())


def ngram_scores(docs: list[list[int]], n: int) -> np.ndarray:
    counts = Counter()
    for doc in docs:
        counts.update({tuple(doc[i : i + n]) for i in range(len(doc) - n + 1)})
    return np.array(
        [
            max(
                (counts[tuple(doc[i : i + n])] for i in range(len(doc) - n + 1)),
                default=0,
            )
            for doc in docs
        ],
        dtype=float,
    )


@torch.no_grad()
def token_losses(model, docs: list[list[int]], device) -> list[np.ndarray]:
    model.eval()
    out = []
    for doc in docs:
        x = torch.tensor([doc], device=device)
        logits, _ = model(x)
        out.append(
            F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="none").cpu().numpy()
        )
    return out


def min_window_mean(losses: np.ndarray, window: int) -> float:
    if len(losses) < window:
        return float(np.mean(losses))
    return float(np.convolve(losses, np.ones(window) / window, mode="valid").min())


def loss_scores(losses: list[np.ndarray], window: int = 8) -> np.ndarray:
    return np.array([-min_window_mean(l, window) for l in losses])
