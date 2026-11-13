# Model card: vichara-reasoner

## Summary

Qwen2.5-0.5B (base) loaded into this repo's own transformer code and trained with this repo's own GRPO on GSM8K train for 150 steps, rewarding only a correct final number. It is a research artifact, not a product.

| | |
|---|---|
| base model | Qwen/Qwen2.5-0.5B (Apache-2.0), 494M parameters |
| training | GRPO, 150 steps × 16 problems × 8 samples, lr 1e-6, KL β 0.02, ~3 h on one L4 |
| reward | 1 if the last number in the answer equals the GSM8K answer, else 0 |
| weights | `checkpoints/grpo2_correct/final.pt` (bf16; not in git) |
| code | `training/grpo_lm.py`, `experiments/e15_grpo_qwen/` |

## Evaluation

GSM8K test (1,319 problems), greedy, last-number extraction:

| model | accuracy |
|---|---|
| base, zero-shot "Let's think step by step." | 30.3% |
| base, 4-shot | 39.1% |
| **vichara-reasoner** (fp32 weights at step 150, bf16 autocast) | **48.8%** |
| GRPO with a format-only reward (control) | 29.6% |
| GRPO with a random reward (control) | 31.6% |

The released bf16 weights scored 50.5% vs the base model's 27.5% on the first 200 test problems (Mac, fp32 compute). They have not been re-scored on the full test set.

## Intended use

Studying RL on small reasoning models: training dynamics, controls, reward hacking, interpretability. Not for real-world maths help, tutoring, or anything where a wrong answer matters: about half its answers are wrong.

## Known limitations

- **Only GSM8K-style word problems.** No evaluation on other maths sets, other languages, or general chat.
- **Ignores the requested output format.** It was asked to end with "The answer is N." but does so only ~22% of the time; it scores because the grader takes the last number. A strict grader would score it lower.
- **Runs on after answering** (e.g. "You are an AI assistant..."), a habit of the base model; the demo server cuts this with stop strings.
- **Possible contamination of the base model.** Qwen2.5's pretraining data is not public; GSM8K test leakage cannot be ruled out, so absolute numbers may be optimistic. The comparison between arms is unaffected, since all start from the same model.
- **One seed per arm.** Differences of a few points between runs are not meaningful; the ~17-point gap between the correctness arm and both controls is.

## Safety notes

- **Reward hacking observed in the format-only control:** reward ~98% while accuracy did not improve and answers halved in length. A monitor comparing reward with an audited sample of 8 of 128 outputs per step flags it at step 20 with 2% false alarms on the honest run (`experiments/e17_reward_monitor/`). This is one blatant hack; subtle hacks are untested.
- **No harmful-content training or evaluation.** The model has no safety tuning beyond the base model's, and refusal/jailbreak behaviour was not measured. The demo server's input filter only blocks over-long inputs and special-token injection; it is not a content filter.

pass@k on 200 test problems at temperature 1 (8 samples): base 11.4 / 45.0% (k = 1 / 8), vichara-reasoner 33.7 / 70.5%. It solves 65 problems the base model never solves in 8 samples, and loses 14.

## Interpretability notes

Weight changes are nearly uniform across layers, an Adam effect. Swapping weights between the two models shows that the change on the model's own correct solutions sits mostly in the MLPs of layers 0–11: putting just those into the base model recovers 77% of the log-probability gap, and removing them from vichara-reasoner loses 56% (`experiments/e18_grpo_diff/`). In accuracy on 200 test problems, base + those MLPs reaches 42.0% (base 27.5%, vichara-reasoner 50.5%), and vichara-reasoner without them falls to 31.0%.
