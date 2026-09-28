# Bigram model as a neural network which is trained by gradient descent

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.download import get
from experiments.e01_bigram.bigram import bigram_probs, build_vocab, count_bigrams


def manual_ce_grad(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # dL/dLogits for mean cross-entropy which is derived by hand in the form of (softmax - onehot) / N
    grad = logits.softmax(
        dim=-1
    )  # form is of [N, V] predictions p  [N = examples and V is scores each]
    grad[torch.arange(len(targets)), targets] -= 1
    return grad / len(targets)  # the loss is a mean over N rows so we divide it by N


def eval_loss(W: torch.Tensor, ids: torch.Tensor) -> float:
    with torch.no_grad():
        return F.cross_entropy(W[ids[:-1]], ids[1:]).item()


def train(
    train_ids: torch.Tensor,
    vocab_size: int,
    steps: int = 3000,
    batch_size: int = 4096,
    lr: float = 50.0,
    seed: int = 0,
    log_every: int = 500,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    W = torch.zeros(vocab_size, vocab_size, requires_grad=True)

    for step in range(steps):
        # first, lets sample a random minibatch of previous,next pairs
        ix = torch.randint(0, len(train_ids) - 1, (batch_size,), generator=g)
        x, y = train_ids[ix], train_ids[ix + 1]
        # forward pass: logits -> loss
        loss = F.cross_entropy(W[x], y)
        # clear out the old gradients and then a backward pass
        W.grad = None
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad

        if log_every and step % log_every == 0:
            print(f"step {step:5d} batch loss {loss.item(): .4f}")

    return W.detach()


def main() -> None:
    text = get("shakespeare").read_text()
    stoi, itos = build_vocab(text)
    vocab_size = len(itos)
    ids = torch.tensor([stoi[ch] for ch in text])
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    # sanity check 1: our derivative vs PyTorch's autograd, on random data
    logits = torch.randn(8, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (8,))
    F.cross_entropy(logits, targets).backward()
    diff = (manual_ce_grad(logits.detach(), targets) - logits.grad).abs().max().item()
    print(f"gradient check: max |manual - autograd| = {diff:.2e}")

    # sanity check 2: the initial loss must be ln(V)
    print(
        f"initial loss {eval_loss(torch.zeros(vocab_size, vocab_size), train_ids):.4f}  (ln V = {math.log(vocab_size):.4f})\n"
    )

    t0 = time.perf_counter()
    W = train(train_ids, vocab_size)
    print(f"\ntrained in {time.perf_counter() - t0:.1f}s")

    count_probs = bigram_probs(count_bigrams(train_ids, vocab_size), alpha=0.01)
    count_W = count_probs.log().float()  # log-probs used as logits: softmax(log p) = p
    print(f"{'':16}{'train':>8}{'val':>8}")
    print(
        f"{'count bigram':16}{eval_loss(count_W, train_ids):8.4f}{eval_loss(count_W, val_ids):8.4f}"
    )
    print(
        f"{'neural bigram':16}{eval_loss(W, train_ids):8.4f}{eval_loss(W, val_ids):8.4f}"
    )

    t = stoi["t"]
    neural_probs = W.softmax(dim=-1)
    print("\nP(next | 't')    count   neural")
    for i in count_probs[t].argsort(descending=True)[:5]:
        print(
            f"  {itos[i]!r:12}{count_probs[t, i].item():8.3f}{neural_probs[t, i].item():8.3f}"
        )

    print("\nlearning-rate sweep (3000 steps each)")
    for lr in (0.5, 5.0, 50.0, 500.0):
        W_lr = train(train_ids, vocab_size, lr=lr, log_every=0)
        print(
            f"  lr={lr:<6} train {eval_loss(W_lr, train_ids):.4f}   val {eval_loss(W_lr, val_ids):.4f}"
        )


if __name__ == "__main__":
    main()
