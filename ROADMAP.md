# Vichara — Roadmap

A reasoning language model built from scratch, from the math up to production.
This file is the plan. Milestones are done only when every "Done when" item is true.

## Resources (designed around these)

| Resource                                    | What it is                   | What we use it for                                          |
| ------------------------------------------- | ---------------------------- | ----------------------------------------------------------- |
| MacBook, M4 Pro, 24 GB unified memory       | PyTorch MPS backend, no CUDA | All of Level 1: tokenizer, model, pretraining ≤ ~30M params |
| Kaggle (free, ~30 GPU-h/week, T4×2 or P100) | Real CUDA, 2 GPUs            | CUDA profiling, first DDP run                               |
| Colab free (T4)                             | Real CUDA                    | Short profiling sessions                                    |
| Modal, one account (~$30/month credit)      | A100/H100 by the second      | Level 2 scale-up runs, FSDP, vLLM comparison                |

Rule: anything that needs more than about $30 of GPU time is simulated at small scale. Checkpoints always resume, so runs can survive preemption.

## The 2-day sprint (current)

| #   | Lesson                                                                   | Output                                        |
| --- | ------------------------------------------------------------------------ | --------------------------------------------- |
| 1   | Language modeling = probability; loss, perplexity; count-based bigram    | `experiments/e01_bigram`, baseline numbers    |
| 2   | Tokenization: bytes, Unicode, BPE from scratch                           | `tokenizer/bpe.py`, trained vocab, tests      |
| 3   | Gradients refresher; neural bigram; softmax + cross-entropy derivation   | neural bigram that matches the count model    |
| 4   | Attention by hand → PyTorch; causal mask; multi-head; GQA                | `model/attention.py`, shape + causality tests |
| 5   | Transformer block: RMSNorm, RoPE, SwiGLU, pre-norm residuals             | `model/transformer.py`, param-count check     |
| 6   | Training loop: AdamW, warmup+cosine, clipping, accumulation, checkpoints | `training/pretrain.py`, loss curves           |
| 7   | Pretrain ~10–20M model on TinyStories on MPS                             | checkpoint that writes coherent stories       |
| 8   | Generation: sampling, temperature, top-k/p, KV cache                     | `inference/generate.py`, speedup measured     |
| 9   | First reasoning experiment: arithmetic with vs. without a scratchpad     | results table + write-up                      |
| 10  | First interpretability experiment: head ablation on our model            | results + causal-vs-correlational discussion  |

## Full roadmap

### Level 1 — Educational model (10M–100M params, Mac)

**M1 · Language modeling foundations**

- Done when: you can explain the chain-rule factorization, NLL, cross-entropy and perplexity; the bigram baseline runs; you can predict the uniform-model loss before running it.

**M2 · Tokenizer**

- Done when: byte-level BPE trains on our corpus; `decode(encode(x)) == x` for arbitrary Unicode (tested); special tokens work; vocab statistics are analyzed (compression ratio, token length distribution); compared against tiktoken/SentencePiece on the same text.

**M3 · Autograd and optimization basics**

- Done when: you derive ∂L/∂logits = softmax − onehot by hand and verify it against autograd; the neural bigram converges to the count model's loss (and you can explain why).

**M4 · Attention**

- Done when: attention is computed by hand on a 3-token example and matches the code; a causal-mask test proves no future leakage; MHA/MQA/GQA are implemented and KV-cache size is compared analytically.

**M5 · Modern transformer block**

- Done when: RMSNorm, RoPE, SwiGLU and pre-norm are implemented with tests; the parameter count is computed by hand and matches `sum(p.numel())`; the design is compared against a current open model config (Llama/Qwen/OLMo family, checked at the time).

**M6 · Pretraining**

- Done when: the training loop has AdamW, warmup+cosine, clipping, accumulation, resumable checkpoints and train/val loss logging; FLOPs and training time are estimated beforehand and compared with measurements; the model beats the bigram baseline by a wide margin and generates coherent TinyStories text.

**M7 · Generation and KV cache**

- Done when: greedy, temperature, top-k and top-p sampling work; the KV cache gives identical outputs to no-cache generation (tested) and the speedup is measured.

### Level 2 — Research model (Kaggle / Modal)

**M8 · Evaluation harness**: val perplexity, a held-out reasoning set, a safety prompt set, all reproducible from a config plus seed.
**M9 · Hardware literacy**: profile a training step on a CUDA GPU (PyTorch Profiler); classify ops as compute-, memory- or launch-bound; measure MFU.
**M10 · Mixed precision and efficiency**: BF16/FP16 with loss scaling, torch.compile, FlashAttention (SDPA), activation checkpointing; each change measured on its own.
**M11 · Distributed training**: DDP on Kaggle T4×2; FSDP on Modal; for each, you can answer what is replicated, what is sharded, and where communication happens.
**M12 · Scale-up run**: ~100M params on a larger dataset (e.g. FineWeb-Edu subset), resumable across sessions, tracked in W&B.
**M13 · Modern architecture study**: MoE (routing, load balancing), sliding-window attention, RoPE scaling (NTK/YaRN), each as a small controlled experiment.
**M14 · SFT**: chat template, loss masking, instruction data; before/after eval.
**M15 · Reasoning training**: synthetic verifiable tasks → rejection-sampling SFT → GRPO with verifiable rewards; watch for reward hacking; test-time compute (self-consistency, best-of-N) curves.
**M16 · Preference and safety training**: DPO; safety SFT; refusal-consistency and over-refusal metrics.
**M17 · Interpretability**: activation caching, linear probes, activation patching, a small sparse autoencoder on our residual stream.
**M18 · Security**: controlled experiments on our own models only: backdoor via data poisoning (and detection), membership inference, training-data extraction, prompt injection on a toy tool-using agent.

### Level 3 — Production system

**M19 · Inference engine**: batched prefill/decode, continuous batching, paged KV cache (simplified), speculative decoding; TTFT, inter-token latency and throughput measured.
**M20 · Quantization**: INT8/INT4 weight-only (our own implementation, then GPTQ/AWQ-style); quality vs. memory vs. latency table.
**M21 · Study production engines**: vLLM / SGLang / llama.cpp; map each feature to the problem it solves; benchmark against our engine.
**M22 · Server**: FastAPI, streaming, auth, rate limiting, input/output safety classifiers, structured logging, metrics, tracing.
**M23 · MLOps**: Docker, CI with eval gates (quality, safety, security), model registry, canary deployment and rollback on Modal.
**M24 · Observability**: infrastructure, model, application and security dashboards, collecting only what's needed (privacy).

## How every lesson runs

Concept → intuition → worked example → math → implementation → test → experiment → analysis.
Code changes to existing files are announced as FILE / CHANGE / WHY / IMPACT.
Research claims are labelled: established · empirical finding · hypothesis · speculation · our experiment.
