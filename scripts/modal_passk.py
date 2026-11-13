from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/vichara"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "regex", "safetensors", "huggingface_hub", "tokenizers")
    .add_local_dir(ROOT / "model", f"{REMOTE}/model")
    .add_local_dir(ROOT / "inference", f"{REMOTE}/inference")
    .add_local_dir(ROOT / "reasoning", f"{REMOTE}/reasoning")
    .add_local_dir(ROOT / "training", f"{REMOTE}/training")
    .add_local_dir(ROOT / "tokenizer", f"{REMOTE}/tokenizer")
    .add_local_dir(ROOT / "experiments" / "e16_passk", f"{REMOTE}/experiments/e16_passk")
)
models = modal.Volume.from_name("vichara-models", create_if_missing=True)
checkpoints = modal.Volume.from_name("vichara-checkpoints", create_if_missing=True)
app = modal.App("vichara-passk", image=image)


@app.function(gpu="L4", timeout=3 * 3600, volumes={"/models": models, f"{REMOTE}/checkpoints": checkpoints},
              env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
def passk(argv: list[str]) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    from experiments.e16_passk.run import main

    try:
        return main(["--model", "/models/qwen2.5-0.5b", *argv])["summary"]
    finally:
        checkpoints.commit()


@app.local_entrypoint()
def run(name: str = "base", extra: str = ""):
    import shlex

    argv = ["--name", name, "--out", f"checkpoints/passk/{name}.json", *shlex.split(extra)]
    if name != "base":
        argv += ["--weights", f"checkpoints/grpo2_{name}/final.pt"]
    call = passk.spawn(argv)
    print(f"spawned {name}: {call.object_id}. It keeps running after this command exits.")
