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
    .add_local_dir(ROOT / "experiments" / "e15_grpo_qwen", f"{REMOTE}/experiments/e15_grpo_qwen")
)
models = modal.Volume.from_name("vichara-models", create_if_missing=True)
checkpoints = modal.Volume.from_name("vichara-checkpoints", create_if_missing=True)
app = modal.App("vichara-grpo", image=image)


@app.function(gpu="L4", timeout=4 * 3600, volumes={"/models": models, f"{REMOTE}/checkpoints": checkpoints},
              env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def train(argv: list[str]) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    path = "/models/qwen2.5-0.5b"
    if not Path(path, "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen2.5-0.5B", local_dir=path,
                          allow_patterns=["config.json", "model.safetensors", "tokenizer.json"])
        models.commit()
    from experiments.e15_grpo_qwen.run import main

    try:
        return main(["--model", path, *argv])
    finally:
        checkpoints.commit()


@app.local_entrypoint()
def run(arm: str = "correct", extra: str = ""):
    import shlex

    call = train.spawn(["--arm", arm, "--resume", *shlex.split(extra)])
    print(f"spawned {arm}: {call.object_id}. It keeps running after this command exits.")
