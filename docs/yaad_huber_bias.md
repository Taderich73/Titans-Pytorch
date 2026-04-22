# Yaad: Huber Attentional Bias

> **Paper**: Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*. [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)

> **Paper alignment:** Behrouz et al., 2025 (Miras / Yaad)
>
> **Implementation status:** Novel extension on top of the paper's concept.
>
> **Details:** The Miras paper introduces Huber attentional bias as a variant of the memory objective. The specific form here — a per-chunk parallel Huber loss with configurable `huber_delta`, integrated into the inner-loop update alongside the project's memory gradient / error clipping and delta-memory parameterization — is the project's own formulation. Activated via `memory_objective="huber"`. Treat the exact numerical behavior as project-local rather than paper-faithful.

## Overview

Yaad is a variant from the **Miras** framework that replaces the standard L2 attentional bias with a **Huber loss** for memory updates. This makes the surprise-driven memory update robust to outlier tokens.

```
Standard (L2):  loss = ||M(k_t) - v_t||^2         -- all errors treated equally
Yaad (Huber):   loss = { L2  if |error| <= delta   -- precise for normal tokens
                       { L1  if |error| > delta    -- bounded for outliers
```

With L2, a single outlier token can generate an extremely large gradient that overwrites useful associations in memory. The Huber loss caps the gradient magnitude for large errors (L1 region), keeping updates bounded while preserving the precision of L2 for small, informative errors.

## Data-Dependent Threshold

The threshold `delta` is not a fixed hyperparameter -- it is **data-dependent**. A learned gate (`Linear(dim, 1)` + sigmoid) produces a per-chunk delta value from the input, allowing the model to adapt the outlier boundary based on content.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_objective` | `"l2"` | `"l2"` (default) or `"huber"` (Yaad) |
| `huber_delta_init` | 0.0 | Bias initialization for the Huber delta gate |

## Usage

```python
import torch
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    memory_objective="huber",
)
model = TitansMAC(config)
```

Composes with all other feature flags (`use_tnt`, `use_attn_res`, `use_mca`, `adaptive_window`, `rope_proportion`).

## When to Use

- Datasets with highly variable token importance (e.g., noisy web text where some tokens carry little signal).
- Training configurations where memory updates show instability or gradient spikes.
- As a drop-in replacement for L2 with minimal hyperparameter tuning -- the data-dependent delta self-adapts.

---

[Back to docs index](README.md) · [Back to project README](../README.md)
