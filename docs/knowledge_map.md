# Knowledge map

Legend: [ ] not started · [~] in progress · [x] understood (you can explain and implement it)

```text
FOUNDATIONS
├── [x] Language modeling as probability (chain rule, next-token prediction)
├── [x] Maximum likelihood, negative log-likelihood, cross-entropy
├── [x] Perplexity
├── [~] Gradients, chain rule over tensors, autograd
└── [ ] Optimization (SGD, Adam, AdamW, schedules)

LANGUAGE MODELS
├── [x] Tokenization (bytes, Unicode, BPE, special tokens)
├── [ ] Embeddings
├── [ ] Attention (causal, multi-head)
├── [ ] Transformer block (RMSNorm, RoPE, SwiGLU, residuals)
├── [ ] Pretraining
└── [ ] Generation (sampling, KV cache)

MODERN ARCHITECTURES   [ ] GQA/MQA  [ ] MoE  [ ] RoPE scaling  [ ] Long context  [ ] Efficient attention
REASONING              [ ] Scratchpads/CoT  [ ] SFT  [ ] Verifiers  [ ] GRPO  [ ] Test-time compute
SAFETY                 [ ] Data safety  [ ] Safety training  [ ] Red teaming  [ ] Evaluation
SECURITY               [ ] Poisoning/backdoors  [ ] Privacy/extraction  [ ] Prompt injection  [ ] Defenses
INTERPRETABILITY       [ ] Ablation  [ ] Probing  [ ] Patching  [ ] SAEs
ML SYSTEMS             [ ] Hardware math  [ ] Profiling  [ ] Mixed precision  [ ] Distributed  [ ] Kernels
PRODUCTION             [ ] Inference engine  [ ] Quantization  [ ] Serving  [ ] MLOps  [ ] Observability
```
