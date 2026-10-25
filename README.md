# vichara - reasoning model built from scratch!

*Vichāra* (विचार) is Sanskrit for thought and inquiry.

This repo builds a reasoning language model from first principles. The tokenizer, attention, positional encoding and training loop are written by hand and checked with tests. Each one is then compared with how current open models (Llama, Qwen, OLMo, ...) and PyTorch's optimized kernels do the same thing.

The goal is to understand every layer of the stack, not just to train a model: the math, the code, the GPU behaviour, and eventually reasoning training, safety, interpretability and serving.

## Status

| | Component | Where | Notes |
|---|---|---|---|
| ✅ | Count-based bigram baseline | `experiments/e01_bigram/` | val loss **2.48** nats/char on Shakespeare: the number every later model has to beat |
| ✅ | Byte-level BPE tokenizer | `tokenizer/` | GPT-2-style pre-tokenization, special tokens, **special-token injection blocked by default** |
| ✅ | Neural bigram trained with SGD | `experiments/e02_neural_bigram/` | hand-derived cross-entropy gradient matches autograd to ~1e-9 |
| ✅ | Causal multi-head attention | `model/attention.py` | MHA / GQA / MQA; output matches `F.scaled_dot_product_attention` |
| ✅ | RMSNorm | `model/norm.py` | computed in fp32 for bf16 safety; matches `nn.RMSNorm` |
| ✅ | RoPE | `model/positional.py` | scores depend only on relative position (tested) |
| ✅ | SwiGLU MLP + full GPT model | `model/transformer.py` | 14.16M params; tied embeddings; parameter count derived by hand |
| ✅ | AdamW + warmup/cosine schedule | `training/optim.py` | AdamW written from scratch; matches `torch.optim.AdamW` |
| ✅ | Training loop | `training/pretrain.py` | clipping, accumulation, resumable atomic checkpoints (crash and resume give a bit-identical result) |
| ✅ | Pretraining on TinyStories | `checkpoints/` (not in git) | val loss **1.42** after 74M tokens (~1 h, M4 Pro); **1.23** after 283M tokens (~25 min on a Modal L4) |
| ✅ | GPU path | `scripts/` | bf16 + fused attention + torch.compile: **6.6× faster** on an L4; training on Modal with persistent volumes and auto-resume |
| ✅ | Generation + KV cache | `inference/generate.py` | temperature / top-k / top-p; KV cache 7× faster on CPU with identical outputs |
| ⬜ | Reasoning, safety, interpretability, serving | | see [ROADMAP.md](ROADMAP.md) |

## Results so far

**Language-model baselines**, Shakespeare at character level (65-character vocabulary), 90/10 contiguous split:

| model | train NLL | val NLL | val perplexity |
|---|---|---|---|
| uniform (knows nothing) | 4.174 | 4.174 | 65.0 |
| unigram | 3.309 | 3.347 | 28.4 |
| bigram, counts (α = 0.01) | 2.452 | 2.488 | 12.0 |
| bigram, neural (SGD, 3k steps) | 2.460 | 2.488 | 12.0 |

**Tokenizer**, 4096-token vocabulary trained on 20M characters of TinyStories:
- **3.98 bytes/token** on held-out text, comparable to GPT-2's 50k vocabulary on English, with 12× fewer entries;
- 435 of 4096 entries are never used on the validation text;
- it falls back to bytes for rare inputs (digits, emoji), so no input is ever unencodable.

**Transformer on Shakespeare** (9.5M params, same data and split as above): best val **1.54** at step 500. It then overfits: by step 1750 val reaches 3.27 while train falls to 0.14, and the overfit checkpoint reproduces **23.5%** of its output verbatim from the training set (control: 0%). See `experiments/e03_memorization/`.

**TinyStories** (14.16M params, 4096-token BPE, 556.6M training tokens, 74M used):

| step | 0 | 500 | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|---|
| train | 8.40 | 2.21 | 1.89 | 1.65 | 1.53 | 1.45 |
| val | 8.40 | 2.17 | 1.85 | 1.62 | 1.50 | 1.42 |

Sample (prompt: *"Once upon a time, a little dragon"*):
> Once upon a time, a little dragon lived in a big forest. He had a very big, strong roar. All the other animals liked to listen to him. He liked to listen to them.

## Layout

```text
data/download.py          corpus download (Shakespeare, TinyStories)
tokenizer/bpe.py          byte-level BPE: train / encode / decode / save / load
tokenizer/models/         trained tokenizer (versioned with the code)
model/attention.py        causal attention, MHA / GQA / MQA, RoPE applied to q and k
model/positional.py       RoPE
model/norm.py             RMSNorm
experiments/eNN_*/        experiments, numbered in the order they were run
tests/                    unit tests, many built from hand-worked examples
ROADMAP.md                milestones with "done when" criteria
docs/knowledge_map.md     what has been learned so far
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy regex pytest

pytest -q                                              # run all tests
python experiments/e01_bigram/bigram.py                # bigram baselines (downloads Shakespeare)
python experiments/e02_neural_bigram/neural_bigram.py  # neural bigram + gradient check
python tokenizer/train_tokenizer.py                    # train + analyze the BPE tokenizer (downloads TinyStories, ~2.2 GB)
```

Developed on macOS with PyTorch's MPS backend. Later stages that need CUDA (profiling, distributed training) are planned for Kaggle and Modal.

## Principles

- **From scratch first, then compare.** Each component is implemented by hand, checked against a worked example, then compared with the production version.
- **Tests from hand calculations.** A tiny case worked out on paper becomes the unit test.
- **Controlled experiments.** Change one thing at a time, compare against a baseline, report the numbers.
- **Safety and security from the start**, not bolted on at the end. Example: the tokenizer never turns untrusted text into control tokens.

## Roadmap

Level 1: a small model understood end to end. Level 2: research scale (distributed training, reasoning RL, safety training, interpretability). Level 3: a production system (inference engine, quantization, serving, monitoring). Details are in [ROADMAP.md](ROADMAP.md).
