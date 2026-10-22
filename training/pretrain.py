# pretraining loop : batches, adamw + schedule, clipping, accumulation, eval and resumable checkpoints.

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.transformer import GPT, GPTConfig  # noqa: E402
from training.data import get_batch, load_meta, load_split  # noqa: E402
from training.optim import AdamW, lr_at, param_groups  # noqa: E402


@dataclass
class TrainConfig:
    dataset: str = "shakespeare_char"
    out_dir: str = "checkpoints/shakespeare_char"
    # model
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    mlp_hidden: int = 1024
    seq_len: int = 256
    # optimization
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_steps: int = 2000
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # bookkeeping
    eval_every: int = 250
    eval_batches: int = 20
    log_every: int = 10
    seed: int = 1337
    device: str = "auto"
    # speed
    precision: str = "fp32"
    sdpa: bool = False
    compile: bool = False


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")  # Modal / Kaggle
    if torch.backends.mps.is_available():
        return torch.device("mps")  # your Mac's GPU
    return torch.device("cpu")


def clip_grad_norm(params, max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(
        sum((g.float() ** 2).sum() for g in grads)
    )  # ONE norm across all parameters
    scale = (max_norm / (total + 1e-6)).clamp(max=1.0)  # ≤ 1: only ever shrink
    for g in grads:
        g.mul_(scale)
    return total.item()  # the norm BEFORE clipping, for logging


def flops_per_token(cfg: GPTConfig, n_params: int) -> float:
    non_embedding = n_params - cfg.vocab_size * cfg.d_model
    return 6 * non_embedding + 12 * cfg.n_layers * cfg.max_seq_len * cfg.d_model


@torch.no_grad()
def autocast(cfg: TrainConfig, device: torch.device):
    dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.precision]
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)


def evaluate(model: GPT, splits: dict, cfg: TrainConfig, device: torch.device) -> dict:
    model.eval()
    out = {}
    for name, tokens in splits.items():
        rng = np.random.default_rng(0)  # same batches every eval → comparable numbers
        losses = []
        for _ in range(cfg.eval_batches):
            x, y = get_batch(tokens, cfg.batch_size, cfg.seq_len, rng)
            with autocast(cfg, device):
                _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def save_checkpoint(
    path: Path, model: GPT, opt, step: int, rng, best_val: float, cfg: TrainConfig
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "rng": rng.bit_generator.state,
        "best_val": best_val,
        "train_config": asdict(cfg),
        "model_config": asdict(model.cfg),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(
        tmp, path
    )  # atomic rename: a crash mid-save never corrupts the previous checkpoint


def build_model(cfg: TrainConfig, vocab_size: int) -> GPT:
    return GPT(
        GPTConfig(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            mlp_hidden=cfg.mlp_hidden,
            max_seq_len=cfg.seq_len,
            use_sdpa=cfg.sdpa,
        )
    )


def train(
    cfg: TrainConfig, resume: bool = False, stop_after: int | None = None
) -> dict:
    device = pick_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": load_split(cfg.dataset, "train"),
        "val": load_split(cfg.dataset, "val"),
    }
    vocab_size = load_meta(cfg.dataset)["vocab_size"]

    torch.manual_seed(cfg.seed)
    model = build_model(cfg, vocab_size).to(device)
    opt = AdamW(param_groups(model, cfg.weight_decay), lr=cfg.max_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.precision == "fp16")
    step_model = torch.compile(model) if cfg.compile else model
    rng = np.random.default_rng(cfg.seed)
    step, best_val = 0, math.inf

    if resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        rng.bit_generator.state = ckpt["rng"]
        step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"resumed from step {step}", flush=True)

    n_params = model.num_params()
    tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps
    flops_step = flops_per_token(model.cfg, n_params) * tokens_per_step
    print(
        f"device {device} | {n_params:,} params | {tokens_per_step:,} tokens/step | "
        f"{flops_step / 1e12:.2f} TFLOPs/step | {cfg.max_steps * tokens_per_step / 1e6:.1f}M tokens total",
        flush=True,
    )

    metrics = {}
    t0 = time.perf_counter()
    with open(out_dir / "log.jsonl", "a", buffering=1) as log:
        while step < cfg.max_steps:
            if (
                stop_after is not None and step >= stop_after
            ):  # simulated crash, used by the resume test
                break

            lr = lr_at(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
            for group in opt.param_groups:
                group["lr"] = lr

            if step % cfg.eval_every == 0:
                metrics = evaluate(model, splits, cfg, device)
                print(
                    f"step {step:5d} | train {metrics['train']:.4f} | val {metrics['val']:.4f}",
                    flush=True,
                )
                log.write(json.dumps({"step": step, "eval": metrics}) + "\n")
                if metrics["val"] < best_val:
                    best_val = metrics["val"]
                    save_checkpoint(
                        out_dir / "best.pt", model, opt, step, rng, best_val, cfg
                    )
                save_checkpoint(
                    out_dir / "last.pt", model, opt, step, rng, best_val, cfg
                )

            loss_sum = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = get_batch(splits["train"], cfg.batch_size, cfg.seq_len, rng)
                with autocast(cfg, device):
                    _, loss = step_model(x.to(device), y.to(device))
                scaler.scale(loss / cfg.grad_accum_steps).backward()
                loss_sum += loss.item()
            scaler.unscale_(opt)
            grad_norm = clip_grad_norm(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % cfg.log_every == 0:
                dt = time.perf_counter() - t0
                tok_s = cfg.log_every * tokens_per_step / dt
                train_loss = loss_sum / cfg.grad_accum_steps
                print(
                    f"step {step:5d} | loss {train_loss:.4f} | lr {lr:.2e} | grad norm {grad_norm:.2f} | "
                    f"{tok_s:,.0f} tok/s | {tok_s * flops_step / tokens_per_step / 1e12:.2f} TFLOP/s",
                    flush=True,
                )
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": train_loss,
                            "lr": lr,
                            "grad_norm": grad_norm,
                        }
                    )
                    + "\n"
                )
                t0 = time.perf_counter()

    if stop_after is None:
        metrics = evaluate(model, splits, cfg, device)
        best_val = min(best_val, metrics["val"])
        print(
            f"final | train {metrics['train']:.4f} | val {metrics['val']:.4f} | best val {best_val:.4f}",
            flush=True,
        )
        save_checkpoint(out_dir / "last.pt", model, opt, step, rng, best_val, cfg)
    return {"step": step, "model": model, **metrics}


def parse_args(argv: list[str] | None = None) -> tuple[TrainConfig, bool]:
    parser = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        name = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(name, type=type(f.default), default=f.default)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args(argv))
    resume = args.pop("resume")
    return TrainConfig(**args), resume


def main() -> None:
    cfg, resume = parse_args()
    train(cfg, resume=resume)


if __name__ == "__main__":
    main()
