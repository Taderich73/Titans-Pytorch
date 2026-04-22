# Paper Alignment and Deviations

This project implements ideas from several papers. Not every line of code is a
direct transcription — some pieces were deliberately simplified for engineering
reasons, and some pieces are novel extensions that go beyond any reference
paper. This doc is the one-stop index; each per-subsystem doc in this folder
has its own "Paper alignment" callout with more detail.

## Paper References

> **Titans**: Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

> **Titans Revisited**: Di Nepi, G., Siciliano, F., & Silvestri, F. (2025). *Titans Revisited*. [arXiv:2510.09551](https://arxiv.org/abs/2510.09551)

> **TNT**: Li, S., Bick, A., Lucchi, A., & Behrouz, A. (2025). *TNT: Improving Chunkwise Training for Test-Time Memorization*. [arXiv:2511.07343](https://arxiv.org/pdf/2511.07343)

> **AttnRes**: Kimi Team (2025). *Attention Residuals*. [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)

> **Miras (Yaad)**: Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*. [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)

> **Gemma 4 (p-RoPE)**: Gemma Team, Google (2025). *Gemma 4 Technical Report*. [Model: google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)

## Legend

- **Faithful** — matches the cited paper equation(s) up to notation.
- **Deviation (fixed)** — previously deviated, now matches the paper (with the plan number that closed the gap).
- **Deviation (deliberate)** — intentionally diverges from the paper; rationale documented.
- **Novel extension** — not present in any reference paper; project-specific engineering.

## Memory core (Titans, Titans Revisited)

| Component | Status | Notes |
| --- | --- | --- |
| `SiLU` / L2-norm on Q, K only (not V) | Faithful (Plan 5 fix) | Paper §3.1 applies to Q/K; V now left un-normalized. |
| `alpha` gate per chunk | Faithful (Plan 5 fix) | Removed unjustified `torch.mean(alpha)` over batch. |
| MAC learned-constant memory query `q_t = S^(t) W_Q` | Faithful (Plan 5 fix) | Paper Eq. 21 is per-position; single learned token removed. |
| MAC/MAG gating `y ⊗ M(·)` (element-wise) | Faithful (Plan 5 fix) | Paper Eq. 25/28; σ(·)·σ(·) form removed. |
| Deep memory K-step inner loop | Faithful (Plan 5) | Configurable K=4–8 per chunk, matching per-token update semantics. |
| MAL ordering (parallel sum) | Faithful (Plan 5 fix) | Paper Eq. 29–31; memory→attention→sum branches run in parallel and sum. |
| Retrieve-before-update in MAC | Faithful (Plan 5 fix) | `retrieve(M_{t-1})` decoupled from `update(M_t)`. |
| Error-scale `2/S` | Faithful (Plan 5) | Absorbed into learnable θ; aligns with paper. |
| Chunk-level gates (`α, η, θ`) | Deviation (deliberate) | Titans Revisited endorses chunk-level as a valid simplification; large compute win. |
| `per_chunk_decay` reparameterization | Deviation (deliberate) | `token_alpha = 1 − (1 − chunk_alpha)^(1/S)` — algebraically equivalent; just a mapping to the per-token form. |
| Persistent memory init (`std * init_std`) | Deviation (deliberate) | Paper is silent; Gaussian init chosen locally. |
| Cross-batch memory sharing at train time | Deviation (deliberate) | Paper is silent on batching; single memory across batch is a local choice. |
| Delta-memory parameterization | Novel extension | Base W + δW decomposition for inner-loop stability. |
| Memory gradient / error clipping | Novel extension | `memory_grad_clip`, `memory_error_clip`. |
| Huber memory objective (`memory_objective="huber"`) | Novel extension | See [`yaad_huber_bias.md`](yaad_huber_bias.md). |

## TNT hierarchical memory (arXiv 2511.07343)

| Component | Status | Notes |
| --- | --- | --- |
| Learnable `W_init` (initial local state) | Faithful (Plan 6 fix) | Paper §4.1.1; tensor is now a proper `nn.Parameter` (previously frozen via `.data`). |
| Per-position causal Q-K projection | Faithful (Plan 6 fix) | Paper App. C; implemented as linear-attention-style prefix-sum scan. Chunk-mean removed. |
| Reset cadence `t ≡ 0 (mod S_L)` | Faithful (Plan 6 fix) | Paper Eq. 6; reset now fires per-token, not only at chunk boundaries. |

## Attention Residuals (AttnRes, arXiv 2603.15031)

| Component | Status | Notes |
| --- | --- | --- |
| Depth-wise softmax over prior block outputs | Faithful | Paper Eq. 2–6. |
| `AttnResMemoryGate` (scalar importance used as memory-LR modulator) | Novel extension | Paper defines per-layer softmax weights; collapsing to a scalar and feeding it into the memory learning rate is project-specific. |

## Adaptive window sizing

| Component | Status | Notes |
| --- | --- | --- |
| `AdaptiveWindowPredictor` | Novel extension | No reference paper — differentiable sigmoid falloff for per-layer learned window sizes. |

## Memory auto-checkpointing

| Component | Status | Notes |
| --- | --- | --- |
| `MemoryCheckpointer`, `StatisticalNoveltyDetector`, `SignalFrame`, `TransitionRecord` | Novel extension | Engineering layer for training observability and checkpoint selection. Not in any paper. |

## Quantization (`src/titans/quantize_state.py`)

| Component | Status | Notes |
| --- | --- | --- |
| Baseline int4 / int8 min-max quantization | Deviation (deliberate — baseline only) | NOT TurboQuant. Plan 7 renamed and scoped this as a simple baseline; the TurboQuant paper's rotation + Max-Lloyd codebook + QJL residual scheme is noted as a future experiment. |

## Miras / Yaad (arXiv 2504.13173)

| Component | Status | Notes |
| --- | --- | --- |
| Huber attentional bias (`memory_objective="huber"`) | Novel extension on top of paper | Paper introduces Huber in the Miras framework; this project's per-chunk parallel Huber formulation is the project's own extension (see [`yaad_huber_bias.md`](yaad_huber_bias.md)). |

## Proportional RoPE (p-RoPE)

| Component | Status | Notes |
| --- | --- | --- |
| Rotate only first `p` fraction of head dims | Novel extension | Inspired by Gemma 4 E2B/E4B; no formal paper specifies the exact form used here. See [`proportional_rope.md`](proportional_rope.md). |

## Per-subsystem detail

For per-subsystem alignment callouts and implementation details, see each
subsystem doc:

- [`attention_residuals.md`](attention_residuals.md)
- [`tnt_hierarchical_memory.md`](tnt_hierarchical_memory.md)
- [`memory_cross_attention.md`](memory_cross_attention.md)
- [`memory_persistence.md`](memory_persistence.md)
- [`memory_auto_checkpointing.md`](memory_auto_checkpointing.md)
- [`adaptive_window_sizing.md`](adaptive_window_sizing.md)
- [`proportional_rope.md`](proportional_rope.md)
- [`yaad_huber_bias.md`](yaad_huber_bias.md)
- [`configuration_guide.md`](configuration_guide.md)
- [`huggingface_integration.md`](huggingface_integration.md)

---

[Back to docs index](README.md) · [Back to project README](../README.md)
