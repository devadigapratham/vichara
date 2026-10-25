# Vichara — Roadmap

A reasoning language model built from scratch, from the math up to production.
This file is the plan. Milestones are done only when every "Done when" item is true.

## End point (what "finished" means)

The project is finished when all five of these exist, each with its numbers measured and reproducible from the repo:

1. **Two models, trained and evaluated with our own code**
   - `vichara-tiny`: the 14M model trained from scratch on TinyStories (done: val 1.23 after 283M tokens), with its interpretability and safety studies.
   - `vichara-reasoner`: Qwen2.5-0.5B loaded into our own model code and trained with our own GRPO on GSM8K, compared with published recipes (≥ 49.5 GSM8K target), with random-reward and format-only controls.
2. **A deployed system** serving `vichara-reasoner` through our own inference engine (continuous batching, paged KV cache, INT8/INT4 quantization), behind a streaming API with auth, rate limits, input/output safety filters, monitoring, and eval-gated releases with rollback. Throughput compared against vLLM on the same GPU.
3. **A safety and security dossier**: backdoor attack + detection + internals (done), membership inference, reward-hacking monitors during RL, refusal/over-refusal and jailbreak evaluations of the deployed model, and a model card.
4. **An interpretability dossier**: the name-tracking circuit (done), an SAE scored on SAEBench (Pythia-160M), and an analysis of what GRPO changed inside the model.
5. **A public write-up**: a technical report (README + docs) stating every claim with its evidence, controls, and limits.

**The real test (from the original brief):** for any component of a modern reasoning model (tokenizer, attention, positional encoding, MLP, training recipe, data, RL algorithm, safety approach, interpretability method, inference engine, KV cache, quantization, distributed strategy, serving, monitoring), you can explain *why it's designed that way* and build it yourself.

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

### Level 2 — Headline projects ("best within our budget", decided 2026-10-19)

Each project makes a **narrow, verifiable claim** against published numbers under the published protocol. Numbers below come from literature surveys and must be re-verified against the papers before any claim.

**Phase 0 · Shared infrastructure** (needed by all three): GPU path on Kaggle (free T4) and Modal; bf16/fp16 autocast, SDPA/FlashAttention, torch.compile, each speedup measured; a shared evaluation harness and experiment tracking.

**Project 1 · BabyLM Strict-Small** (dropped 2026-10-25 by the user's choice; research kept in docs/lessons/research_babylm_2026.md)
- Rules: ≤ 10M words of training data, ≤ 10 epochs, no compute cap; data swaps allowed in the 2026 edition.
- Bar (2025 results): GPT-2 baseline 49.1 · GPT-BERT-causal ≈ 56.4 · winner 58.3 (masked model).
- Plan: a ~30M causal decoder (ours), then ablate speedrun techniques (Muon, QK-norm, ReLU², value embeddings, softcap). Secondary score: FineWeb bits-per-byte against pythia-14m/31m.
- Compute: ~8–12 h on the Mac, or < 1 h on a Kaggle T4.

**Project 2 · Small reasoning model: our GRPO on an open base model**
- Load Qwen2.5-0.5B **into our own model code** (a weight-loading exercise), then run our GRPO on GSM8K/MATH.
- Bar: Qwen2.5-0.5B base GSM8K 36.7 → 49.5 (SimpleRL-Zoo); 56.7 from Instruct (verl). Stretch: 1.5B GSM8K 74–78, MATH-500 ≥ 55–59.
- Controls: format-only reward, one prompt, greedy decoding, the same answer extractor before and after; watch for Qwen contamination effects.
- Compute: ~$30–45 of Modal L4/A10G time; save credits until Phase 0 is done.

**Project 3 · Interpretability and safety findings** (on the models from Projects 1–2)
- Reward-hacking detection during GRPO: a probe vs a head-ablation signature vs output checks (AUC, earliest detection, false positives). Extends Experiment 7.
- Backdoor + "defection probes" on our own model: do generic linear probes catch a planted trigger at small scale? Does ablation remove it?
- One directly comparable number: a Matryoshka BatchTopK SAE on Pythia-160M layer 8 → SAEBench.

**Safety and interpretability are equal tracks, interleaved from the start** (most of it runs on the Mac with existing models):

| Interpretability | Safety & security |
|---|---|
| I1 activation patching (the L7H1 mystery, Experiment 5) | S1 backdoor via data poisoning on TinyStories (attack success vs poison rate) |
| I2 linear probes (character tracking, "will this answer be correct?") | S2 backdoor detection: defection probes + head ablation (detection rate, false positives) |
| I3 SAE on TinyStories → SAEBench on Pythia-160M | S3 membership inference on the BabyLM model (clean member/non-member split) |
| I4 what did GRPO change inside the model? | S4 live reward-hacking monitors during Project 2's RL |
| | S5 instruction tuning → refusals → jailbreak robustness |

**Order:** S1 → Phase 0 → I1 + I2 → S2 → Project 1 → S3 → I3 → Project 2 + S4 → I4 → S5.

**The full original plan continues alongside the headline projects (nothing is dropped). Coverage and the "best-in-class" angle per area:**

| Area | Done | To do | Best-in-class angle |
|---|---|---|---|
| Architecture | BPE, RoPE, GQA/MQA, RMSNorm, SwiGLU | sliding window, MoE, RoPE scaling, long context | MoE vs dense at equal active params inside BabyLM; YaRN + passkey retrieval |
| Training systems | AdamW, schedules, clipping, accumulation, checkpoints | mixed precision, FlashAttention, torch.compile, profiling, DDP/FSDP, TP/PP, Triton | DDP scaling on Kaggle 2×T4; MFU of our kernels |
| Data pipeline | — | collection, provenance, cleaning, quality filtering, dedup, PII/safety filtering | data curation for BabyLM's 10M words (Project 1) |
| Reasoning | scratchpad/CoT, GRPO, reward hacking | self-consistency, best-of-N, search, ORM/PRM, rejection-sampling SFT, PPO, DPO, distillation, test-time scaling | accuracy-vs-compute curves on GSM8K vs published |
| Safety | reward hacking | safety SFT, preference training, red teaming, classifiers, constitutional, refusals, jailbreaks, I/O filters | XSTest over-refusal + HarmBench subset on our instruction-tuned Qwen |
| Security | memorization | backdoors, MIA, extraction, prompt injection, malicious fine-tuning, supply chain, RAG/agent, sandboxing | TPR@low-FPR for MIA and backdoor detection; injection on a toy agent |
| Interpretability | attention patterns, ablation | probes, patching, causal tracing, attribution, SAEs, circuits | SAEBench on Pythia-160M |
| Inference | sampling, KV cache | continuous batching, paged KV, speculative decoding, beam search | our engine's throughput vs vLLM on the same GPU/model |
| Quantization | — | INT8/INT4, GPTQ, AWQ, SmoothQuant, QAT | our GPTQ matching published 4-bit perplexity deltas |
| Serving / MLOps / observability / deployment | — | FastAPI streaming, auth, rate limits, Docker, CI eval gates, registry, canary/rollback, monitoring, autoscaling | a deployed endpoint with eval-gated releases |

**Learning track (not headline):** length generalization (harness done, Experiment 8 baseline: 100% at 1–10 digits, 0% from 11), position coupling / Abacus as an exercise.

### Level 2 — Research program (earlier plan, kept for reference)

The three tracks are interleaved on purpose: the interpretability tools explain the reasoning results, and the safety experiments use both.

**Track A · Reasoning: length generalization in addition (the "SOTA within our budget" target)**
- A1 Literature survey + baseline harness: train on short numbers, test far longer; many seeds; published protocol.
- A2 Reproduce the failure at scale (e.g. train ≤ 10 digits, test 11–50).
- A3 Implement published methods on equal footing: index hints, randomized positions, Abacus-style digit-position embeddings, position coupling.
- A4 Try to improve on them; ablate what matters.
- Done when: a results table with ≥ 3 seeds per method, compared against published numbers using the same protocol.

**Track B · Mechanistic interpretability**
- B1 Probes: does the adder's residual stream linearly encode the carry? At which layer? Does that break at unseen lengths (explaining A2)?
- B2 Activation patching: solve the L7H1 mystery from Experiment 5 (does it read what L5H4 wrote?).
- B3 A small sparse autoencoder on the TinyStories residual stream: find interpretable features.
- Done when: each claim is backed by an intervention, not just a correlation, with controls.

**Track C · AI safety and security**
- ✅ C0 Reward hacking (Experiment 7): buggy verifier → proxy ↑, truth → 0; KL as a safeguard.
- C1 Backdoor via data poisoning on TinyStories: a trigger phrase → a planted behaviour; measure attack success vs. poison rate; then try to detect it (with Track B tools).
- C2 Membership inference / training-data extraction, extending Experiment 3: can an attacker tell which text was in training?
- C3 SFT for instruction-following, then refusal behaviour and jailbreak robustness on our own model.
- Done when: each attack has a measured success rate, and each defence has measured false positives and false negatives.

Already done in Level 2: ✅ GRPO from scratch (Experiment 6: RL sharpens pass@1, pass@8 unchanged; SFT control), ✅ reward hacking (Experiment 7).

### Level 2 — Milestone reference (Kaggle / Modal)

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
