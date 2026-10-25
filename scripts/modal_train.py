import shlex
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/vichara"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "regex")
    .add_local_dir(ROOT / "model", f"{REMOTE}/model")
    .add_local_dir(ROOT / "training", f"{REMOTE}/training")
)
data = modal.Volume.from_name("vichara-data", create_if_missing=True)
checkpoints = modal.Volume.from_name("vichara-checkpoints", create_if_missing=True)
app = modal.App("vichara-train", image=image)

VOLUMES = {f"{REMOTE}/data": data, f"{REMOTE}/checkpoints": checkpoints}
RETRIES = modal.Retries(max_retries=2, initial_delay=10.0)

TINYSTORIES_FULL = (
    "--dataset tinystories --out-dir checkpoints/tinystories_full "
    "--n-layers 8 --seq-len 512 --batch-size 32 --max-steps 17300 --warmup-steps 200 "
    "--eval-every 1000 --log-every 200 --precision bf16 --sdpa --compile"
)


def command(flags: str) -> list[str]:
    return shlex.split(flags) + ["--resume"]


def _train(argv: list[str]) -> None:
    import os
    import sys

    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    from training.pretrain import parse_args, train

    cfg, resume = parse_args(argv)
    try:
        train(cfg, resume=resume)
    finally:
        checkpoints.commit()


@app.function(gpu="L4", timeout=4 * 3600, volumes=VOLUMES, retries=RETRIES)
def train_l4(argv: list[str]) -> None:
    _train(argv)


@app.function(gpu="A100", timeout=4 * 3600, volumes=VOLUMES, retries=RETRIES)
def train_a100(argv: list[str]) -> None:
    _train(argv)


TRAINERS = {"L4": train_l4, "A100": train_a100}


@app.local_entrypoint()
def run(flags: str = TINYSTORIES_FULL, gpu: str = "L4"):
    if gpu not in TRAINERS:
        raise SystemExit(f"unknown gpu {gpu!r}; choose from {list(TRAINERS)}")
    TRAINERS[gpu].remote(command(flags))
