import json
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
    .add_local_dir(ROOT / "experiments" / "e14_gsm8k_baseline", f"{REMOTE}/experiments/e14_gsm8k_baseline")
)
models = modal.Volume.from_name("vichara-models", create_if_missing=True)
app = modal.App("vichara-eval", image=image)


@app.function(gpu="L4", timeout=3600, volumes={"/models": models})
def evaluate(argv: list[str]) -> dict:
    import os
    import sys

    from huggingface_hub import snapshot_download

    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    path = "/models/qwen2.5-0.5b"
    if not Path(path, "model.safetensors").exists():
        snapshot_download("Qwen/Qwen2.5-0.5B", local_dir=path,
                          allow_patterns=["config.json", "model.safetensors", "tokenizer.json"])
        models.commit()
    from experiments.e14_gsm8k_baseline.run import main

    return main(["--model", path, *argv])


@app.local_entrypoint()
def run(prompt: str = "zero", n: int = 1319, batch: int = 64):
    out = evaluate.remote(["--prompt", prompt, "--n", str(n), "--batch", str(batch)])
    path = ROOT / "experiments" / "e14_gsm8k_baseline" / f"results_{prompt}.json"
    path.write_text(json.dumps(out, indent=1))
    print("saved", path)
