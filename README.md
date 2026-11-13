# vichara - reasoning model built from scratch!

*Vichāra* (विचार) is Sanskrit for thought and inquiry.

A reasoning language model stack built from first principles: tokenizer, transformer, training loop, generation, RL, interpretability and safety tools, all written by hand in this repo and checked with tests. It ends with two models trained with this code, a set of controlled experiments on reasoning, safety and interpretability, and a small streaming API.

Every number below is reproducible from the scripts named next to it. Each claim comes with its control and its limits.

## Contents

1. [Models](#models)
2. [Reasoning: GRPO on GSM8K](#reasoning-grpo-on-gsm8k)
3. [Safety and security](#safety-and-security)
4. [Interpretability](#interpretability)
5. [Systems and serving](#systems-and-serving)
6. [Limitations](#limitations)
7. [Reproducing](#reproducing)
8. [Layout](#layout)
9. [Future work](#future-work)

## Models

| model | what it is | headline |
|---|---|---|
| `vichara-tiny` | 14.16M-param GPT (RoPE, GQA, RMSNorm, SwiGLU, tied embeddings), trained from scratch on TinyStories with a 4096-token BPE written here | val loss **1.23** after 283M tokens (25 min on one L4); 1.42 after 74M tokens on a MacBook |
| `vichara-reasoner` | Qwen2.5-0.5B (base) loaded into this repo's transformer, trained with this repo's GRPO on GSM8K | GSM8K **30.3% → 48.8%**; see [MODEL_CARD.md](MODEL_CARD.md) |

Loading Qwen into our own code matches Hugging Face's logits to 3.5e-4 (same argmax everywhere), so every result on `vichara-reasoner` runs through code in this repo, not a library model.

Sample from `vichara-tiny` (prompt *"Once upon a time, a little dragon"*):
> Once upon a time, a little dragon lived in a big forest. He had a very big, strong roar. All the other animals liked to listen to him. He liked to listen to them.

## Reasoning: GRPO on GSM8K

**Setup.** GRPO from the base model, no SFT first. Each step samples 8 answers for each of 16 GSM8K train problems, scores them, normalizes rewards within each group, and takes one update (lr 1e-6, KL to the frozen base β = 0.02). 150 steps, about 3 hours on one L4. Three runs that differ **only** in the reward:

| reward | GSM8K test (1,319, greedy) |
|---|---|
| none: base, zero-shot "Let's think step by step." | 30.3% |
| none: base, 4-shot | 39.1% (Qwen reports 41.6%) |
| **correct final number** | **48.8%** |
| answer format only ("The answer is N.") | 29.6% |
| coin flip | 31.6% |

*`experiments/e14_gsm8k_baseline`, `experiments/e15_grpo_qwen` (logs in `logs/`)*

- **The gain comes from the reward.** The random-reward run shows that the update machinery alone does nothing. The format-only run shows that learning the output format doesn't help either. It learned the format on 98% of answers, while its accuracy fell slightly and its answers got half as long.
- **The correctness run ignores the requested format** (22%) and still scores, because the grader takes the last number. The grader is part of the reward.

**Did RL teach new problems, or sharpen old ones?** pass@k on 200 test problems, 8 samples each at temperature 1 (`experiments/e16_passk`):

| k | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| base | 11.4% | 19.7% | 31.4% | 45.0% |
| GRPO | 33.7% | 46.4% | 58.8% | 70.5% |

GRPO solves 65 problems that the base model never solves in 8 tries; the reverse happens for 14. Up to k = 8 the curves don't converge, which points to more than sharpening. Two limits: Yue et al. (2025) find the base model catching up only at k in the hundreds, which is untested here; and part of the gap is robustness to sampling, since temperature 1 costs the base model far more (greedy 27.5% → pass@1 11.4%) than it costs GRPO.

**Earlier, from scratch on `vichara-tiny`-sized models** (`e04`, `e06`–`e08`): a scratchpad makes 3-digit addition learnable in 500 steps (99% vs 19% for direct answers, 3 seeds); 4-digit generalization is 0% in all 18 runs. GRPO from scratch on arithmetic raised pass@1 but slightly lowered pass@8, the sharpening pattern that the Qwen run does not show at k ≤ 8.

## Safety and security

**Data-poisoning backdoors** in `vichara-tiny` (`e09`–`e12`). A trigger phrase in a story makes the model write a target phrase.

| poisoned stories (of ~60k) | 47 | 178 | 602 |
|---|---|---|---|
| attack success | 0% | 86% | 100% |

Validation loss doesn't change at any rate, so the usual training metrics show nothing.

| detector (TPR at 1% FPR, 1% poison rate) | verbatim poison | paraphrased poison |
|---|---|---|
| 24-gram repetition (dedup) | 99% | 24% |
| low-loss window | 96% | 74% |
| loss vs a reference model | 100% (clean control) | 100% (model before fine-tuning) |

- Paraphrasing the poison (120 trigger × 80 target phrasings) defeats deduplication, and the backdoor still fires on held-out phrasings 99% of the time. The model learned the concept, not the string.
- Comparing against the previous model version catches every poisoned story. A defender always has that model.
- Inside the model, activation patching shows the trigger is recognized in layers 0–4 and moved to the output position at layers 4→5.
- 300 steps of clean fine-tuning leave the backdoor at 98%.
- A linear "is the trigger present" probe failed its control (it fires just as well on a clean model), so it is reported as a negative result.

**Reward-hacking monitor** (`e17`, `safety/reward_monitor.py`). The monitor compares the proxy reward with a trusted check of a small random sample of the outputs. It raises an alarm when the gap grows significantly relative to the start of training. The threshold was calibrated only on the random-reward run.

| audited outputs per step (of 128) | format-only run caught at | false alarms on the honest run |
|---|---|---|
| 128 | step 20 (100%) | 0% |
| 8 | step 20 (100%) | 2% |
| 2 | step 20 (100%) | 9% |

Step 20 is the earliest the monitor can fire (two 10-step windows). A label-free alternative, "reward rises while answers get much shorter", also caught the hack, but it false-alarmed on the honest run, whose answers got shorter as it improved. This is one blatant hack; subtle ones are untested.

**Other safety work.** The tokenizer refuses to turn untrusted text into control tokens, and the server rejects special-token syntax. An overfit Shakespeare model reproduced 23.5% of its output verbatim from training data (control: 0%; `e03`).

## Interpretability

**Name tracking in `vichara-tiny`** (`e05`, `e13`). The task: predict a character's name later in a story (97% correct among 18 candidates).

- Ablating 3 of 48 heads drops accuracy to 24–32%; ablating 3 random heads leaves it at 97%.
- L5H4 copies the name directly. L7H1 barely attends to the name; it reads copies of it stored at punctuation positions. Patching only the punctuation positions restores 25% of the effect, and 13% with L7H1 switched off.
- A linear probe reads out the name perfectly after layer 5.

**What GRPO changed inside Qwen** (`e18`). The test set: GRPO's own correct solutions to 64 test problems, which the GRPO model finds 13.1 nats more likely than the base model does. Swapping components between the two models:

| component swapped | base + GRPO's component recovers | GRPO with base's component loses |
|---|---|---|
| MLPs, layers 0–11 | **77%** | **56%** |
| MLPs, layers 12–23 | 16% | 11% |
| all attention | 29% | 10% |
| everything except MLPs 0–11 | 44% | 23% |

The same holds for accuracy (greedy, the first 200 test problems):

| model | accuracy | of the 53 problems only GRPO solves |
|---|---|---|
| base | 27.5% | 0 |
| base + GRPO's MLPs in layers 0–11 | 42.0% | 31 |
| GRPO with the base model's MLPs in layers 0–11 | 31.0% | 17 |
| GRPO | 50.5% | 53 |

The early MLPs alone carry 63% of the accuracy gain, and removing them loses 85% of it.

- **Weight-change size doesn't locate the change.** Every MLP moved by almost the same relative amount (≈ 4e-4) at every depth. Adam normalizes updates, so every weight moves about lr per step whether or not it matters. Only the swap experiment locates the effect.
- **The effects overlap.** Single-component effects sum to 138% for sufficiency and 34% for necessity, so part of the change is redundant.
- **Precision side effect:** the trained weights were saved in bf16, and 86.5% of the RL updates were smaller than bf16's rounding step. The saved model still scores 50.5% vs the base model's 27.5% on 200 test problems.

## Systems and serving

- **GPU path** (`scripts/bench.py`). bf16 + fused attention (SDPA) + `torch.compile` makes training 6.6× faster on an L4 (MFU 18.9%; 23.8% for a larger model). Training runs on Modal with persistent volumes, automatic resume, and atomic checkpoints.
- **Generation** (`inference/generate.py`):
  - KV cache, 7× faster on CPU with identical outputs;
  - batched generation with left padding: a key mask filled with finfo.min, not −inf, so fully padded rows don't produce NaN; per-row RoPE positions; per-row EOS;
  - logits for the last position only, because full 152k-vocab logits ran out of memory.
- **Streaming API** (`inference/server.py`). It wraps `vichara-reasoner` with:
  - API-key auth and a per-key token-bucket rate limit;
  - input guards: length, and special-token injection;
  - stop strings that never leak, even half-written;
  - server-sent-event streaming;
  - `/metrics` (requests, rejections, tokens/s, time to first token).

```bash
VICHARA_API_KEYS=dev python inference/server.py
curl -N -H "X-API-Key: dev" -H "Content-Type: application/json" \
     -d '{"question": "A pen costs $3 and a book costs 4 times as much. How much do both cost?"}' localhost:8000/v1/solve
```

## Limitations

- **One seed per GRPO arm.** The ~17-point gap between the correctness run and both controls is far outside noise (±1.4 points at n = 1,319); smaller differences are not.
- **Possible test contamination:** Qwen2.5's pretraining data isn't public. Comparisons between runs are unaffected, since all start from the same model; absolute numbers may be optimistic.
- **Small samples:** pass@k and the interpretability analysis use 200 test problems on a laptop, not the full test set, and pass@k stops at k = 8.
- **The safety results are on a 14M model and synthetic triggers**, and the reward-hacking monitor is tested on one blatant hack.
- **Not a product.** No refusal or harmful-content training or evaluation. The server's filters are injection and length guards, not content moderation.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve]"
pytest -q
```

| result | command |
|---|---|
| bigram / tokenizer / TinyStories | `python experiments/e01_bigram/bigram.py`, `python tokenizer/train_tokenizer.py`, `python training/pretrain.py` |
| GSM8K baseline | `python experiments/e14_gsm8k_baseline/run.py --prompt zero` (or `modal run scripts/modal_eval.py`) |
| GRPO arms | `modal run --detach scripts/modal_grpo.py --arm correct` (also `format`, `random`) |
| pass@k | `python experiments/e16_passk/run.py --problems 200 --samples 8` (add `--weights .../final.pt`) |
| reward-hacking monitor | `python experiments/e17_reward_monitor/run.py` (CPU, 20 s, uses the committed logs) |
| what GRPO changed | `python experiments/e18_grpo_diff/run.py --behaviour` |

Qwen2.5-0.5B is downloaded to `checkpoints/qwen2.5-0.5b` (config, safetensors, tokenizer). Checkpoints are not in git.

## Layout

```text
tokenizer/      byte-level BPE (train / encode / decode), special-token injection blocked
model/          attention (MHA/GQA/MQA, KV cache, masks), RoPE, RMSNorm, SwiGLU, GPT, Hugging Face Qwen2 import
training/       AdamW, schedules, pretraining loop (mixed precision, compile, resume), GRPO for LMs
inference/      sampling, batched + streaming generation, demo server
reasoning/      arithmetic tasks, GRPO pieces, GSM8K loading/grading, pass@k
interp/         activation and path patching, direct logit attribution, probes
safety/         backdoor construction, poison detectors, reward-hacking monitors
scripts/        GPU benchmark, Modal training / eval / GRPO launchers
experiments/    e01–e18, numbered in the order they were run
tests/          unit tests, many built from hand-worked examples
```

## Future work

Planned but out of scope for v1.0 ([ROADMAP.md](ROADMAP.md)):
- an inference engine with continuous batching and a paged KV cache, compared with vLLM;
- INT8/INT4 quantization;
- sparse autoencoders scored on SAEBench;
- membership inference;
- refusal and jailbreak evaluations;
- GRPO at larger k and with more seeds.
