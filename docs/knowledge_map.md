# Knowledge map

Legend: [ ] not started · [~] in progress · [x] understood (you can explain and implement it)

```text
FOUNDATIONS
├── [x] Language modeling as probability (chain rule, next-token prediction)
├── [x] Maximum likelihood, negative log-likelihood, cross-entropy
├── [x] Perplexity
├── [x] Gradients, chain rule, autograd, gradient checking
└── [x] Optimization (SGD, AdamW, warmup + cosine, clipping, accumulation)

LANGUAGE MODELS
├── [x] Tokenization (bytes, Unicode, BPE, special tokens)
├── [x] Embeddings (nn.Embedding = lookup table, tied with lm_head)
├── [x] Attention (causal, multi-head, GQA/MQA)
├── [x] Transformer block (RMSNorm, RoPE, residuals, pre-norm, SwiGLU, init, weight tying)
├── [x] Pretraining (loop, checkpoints, FLOPs, overfitting vs. single-epoch, TinyStories run)
└── [x] Generation (temperature, top-k, top-p, KV cache, prefill vs decode)

MODERN ARCHITECTURES   [x] GQA/MQA  [ ] MoE  [ ] RoPE scaling  [ ] Long context  [ ] Efficient attention
REASONING              [ ] Scratchpads/CoT  [ ] SFT  [ ] Verifiers  [ ] GRPO  [ ] Test-time compute
SAFETY                 [ ] Data safety  [ ] Safety training  [ ] Red teaming  [ ] Evaluation
SECURITY               [ ] Poisoning/backdoors  [ ] Privacy/extraction  [ ] Prompt injection  [ ] Defenses
INTERPRETABILITY       [ ] Ablation  [ ] Probing  [ ] Patching  [ ] SAEs
ML SYSTEMS             [ ] Hardware math  [ ] Profiling  [ ] Mixed precision  [ ] Distributed  [ ] Kernels
PRODUCTION             [ ] Inference engine  [ ] Quantization  [ ] Serving  [ ] MLOps  [ ] Observability
```
