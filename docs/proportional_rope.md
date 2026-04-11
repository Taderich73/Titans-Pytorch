# Proportional RoPE (p-RoPE)

> **Reference**: Gemma Team, Google (2025). *Gemma 4 Technical Report*. [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)

## Overview

Standard RoPE applies rotary position embeddings to all dimension pairs in each attention head. Research shows that low-frequency pairs (later dimensions) carry negligible positional signal and can disturb semantic representations, especially over long sequences.

Proportional RoPE (p-RoPE) applies rotation to only the first `p` fraction of dimension pairs, leaving the rest unchanged for semantic content. This is the approach used by Gemma 4 E2B/E4B models.

## How It Works

Given `head_dim` dimensions per head and `rope_proportion = p`:

1. Compute `rotate_dim = 2 * floor(head_dim * p / 2)` (round down to even number).
2. Apply standard RoPE rotation to the first `rotate_dim` dimensions.
3. Pass the remaining `head_dim - rotate_dim` dimensions through unchanged.

When `p = 1.0` (default), all dimensions are rotated -- standard RoPE behavior. When `p = 0.0`, no dimensions are rotated, equivalent to `use_rope=False`.

## Configuration

| `rope_proportion` | Positional dims | Semantic dims | Use case |
|---|---|---|---|
| `1.0` (default) | 100% | 0% | Standard RoPE (backward-compatible) |
| `0.25` | 25% | 75% | Gemma 4 default; recommended starting point |
| `0.0` | 0% | 100% | No rotation |

## Usage

```python
import torch
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    rope_proportion=0.25,  # 25% positional, 75% semantic
)
model = TitansMAC(config)
```

## Important Notes

- **Changing `rope_proportion` requires training from scratch.** The model must learn which dimensions carry positional vs. semantic information during training.
- The proportion is saved in checkpoint config and restored automatically at inference.
- Composes freely with all other feature flags.
- Validated by config `__post_init__` to be in `[0.0, 1.0]`.

## Key Class

- `RotaryPositionEmbedding` -- handles proportional rotation via the `rope_proportion` parameter, computing `rotate_dim` at construction time
