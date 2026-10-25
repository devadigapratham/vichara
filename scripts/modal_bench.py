from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "regex")
    .add_local_dir(ROOT / "model", "/root/vichara/model")
    .add_local_dir(ROOT / "training", "/root/vichara/training")
    .add_local_dir(ROOT / "scripts", "/root/vichara/scripts")
)
app = modal.App("vichara-bench", image=image)


def _bench(args: list[str]) -> list[dict]:
    import sys

    sys.path.insert(0, "/root/vichara")
    from scripts.bench import main

    return main(args)


@app.function(gpu="T4", timeout=1800)
def bench_t4(args: list[str]) -> list[dict]:
    return _bench(args)


@app.function(gpu="L4", timeout=1800)
def bench_l4(args: list[str]) -> list[dict]:
    return _bench(args)


@app.function(gpu="A10G", timeout=1800)
def bench_a10g(args: list[str]) -> list[dict]:
    return _bench(args)


@app.function(gpu="A100", timeout=1800)
def bench_a100(args: list[str]) -> list[dict]:
    return _bench(args)


BENCHES = {"T4": bench_t4, "L4": bench_l4, "A10G": bench_a10g, "A100": bench_a100}


@app.local_entrypoint()
def run(gpu: str = "L4", big: bool = False):
    if gpu not in BENCHES:
        raise SystemExit(f"unknown gpu {gpu!r}; choose from {list(BENCHES)}")
    args = ["--d-model", "768", "--layers", "12", "--heads", "12", "--batch", "16"] if big else []
    BENCHES[gpu].remote(args)
