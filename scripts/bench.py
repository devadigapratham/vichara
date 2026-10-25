import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.transformer import GPT, GPTConfig
from training.optim import AdamW, param_groups
from training.pretrain import flops_per_token, pick_device

PEAK_TFLOPS = {"T4": 65, "L4": 121, "A10": 125, "A100": 312, "H100": 989}
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def settings(device: torch.device) -> list[dict]:
    out = [
        {"precision": "fp32", "sdpa": False, "compile": False},
        {"precision": "fp32", "sdpa": True, "compile": False},
    ]
    if device.type == "cuda":
        half = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"  # hardware bf16 needs Ampere+
        out += [
            {"precision": half, "sdpa": False, "compile": False},
            {"precision": half, "sdpa": True, "compile": False},
            {"precision": half, "sdpa": True, "compile": True},
        ]
    elif device.type == "mps":
        out += [{"precision": "fp16", "sdpa": True, "compile": False}]
    else:
        out += [{"precision": "bf16", "sdpa": True, "compile": False}]
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run(
    cfg: GPTConfig,
    setting: dict,
    batch: int,
    steps: int,
    warmup: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**vars(cfg), "use_sdpa": setting["sdpa"]})).to(device)
    opt = AdamW(param_groups(model, 0.1), lr=1e-4)
    dtype = DTYPES[setting["precision"]]
    scaler = torch.amp.GradScaler(device.type, enabled=setting["precision"] == "fp16")
    step_model = torch.compile(model) if setting["compile"] else model
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.max_seq_len), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if i == warmup:
            sync(device)
            t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            _, loss = step_model(x, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    sync(device)
    dt = (time.perf_counter() - t0) / steps
    tok_s = batch * cfg.max_seq_len / dt
    return {
        **setting,
        "ms_per_step": dt * 1000,
        "tok_per_s": tok_s,
        "tflops": tok_s * flops_per_token(cfg, model.num_params()) / 1e12,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9
        if device.type == "cuda"
        else None,
        "loss": loss.item(),
    }


def gpu_peak(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name()
    return next((v for k, v in PEAK_TFLOPS.items() if k in name), None)


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    device = pick_device(args.device)
    cfg = GPTConfig(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=2,
        mlp_hidden=int(8 / 3 * args.d_model),
        max_seq_len=args.seq_len,
    )
    peak = gpu_peak(device)
    name = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    print(
        f"{name} | d_model {args.d_model}, {args.layers} layers, batch {args.batch} x {args.seq_len}"
    )
    results = []
    base = None
    for s in settings(device):
        warmup = args.warmup + (10 if s["compile"] else 0)
        r = run(cfg, s, args.batch, args.steps, warmup, device)
        base = base or r["tok_per_s"]
        mfu = f"{100 * r['tflops'] / peak:5.1f}%" if peak else "  n/a"
        mem = f"{r['peak_mem_gb']:.1f} GB" if r["peak_mem_gb"] is not None else "-"
        label = f"{s['precision']}{' +sdpa' if s['sdpa'] else ''}{' +compile' if s['compile'] else ''}"
        print(
            f"{label:22s} {r['ms_per_step']:8.1f} ms/step {r['tok_per_s']:10,.0f} tok/s "
            f"x{r['tok_per_s'] / base:4.2f}  {r['tflops']:6.1f} TFLOP/s  MFU {mfu}  mem {mem}",
            flush=True,
        )
        results.append(r)
    return results


if __name__ == "__main__":
    main()
