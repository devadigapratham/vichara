import math

import torch

from model.transformer import GPTConfig
from scripts.bench import gpu_peak, run, settings


def test_settings_always_start_with_fp32_baseline():
    for dev in ("cpu", "mps"):
        s = settings(torch.device(dev))
        assert s[0] == {"precision": "fp32", "sdpa": False, "compile": False}
        assert all(not x["compile"] for x in s)        # compile is only benchmarked on CUDA


def test_run_reports_sane_numbers_on_cpu():
    cfg = GPTConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)
    r = run(cfg, {"precision": "fp32", "sdpa": True, "compile": False}, batch=4, steps=2, warmup=1,
            device=torch.device("cpu"))
    assert r["tok_per_s"] > 0 and r["ms_per_step"] > 0 and r["tflops"] > 0
    assert math.isfinite(r["loss"])
    assert r["peak_mem_gb"] is None


def test_no_peak_known_off_cuda():
    assert gpu_peak(torch.device("cpu")) is None
