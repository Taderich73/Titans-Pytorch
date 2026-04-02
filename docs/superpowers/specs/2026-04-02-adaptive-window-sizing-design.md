# Adaptive Window Sizing Design Spec

**Date:** 2026-04-02
**Status:** Approved
**Scope:** Per-layer learned adaptive window sizing for sliding window attention

## Motivation

Sliding window attention uses a fixed window size across all layers and all content. Different layers likely need different context ranges (lower layers for local syntax, upper layers for broader reasoning), and different content may benefit from wider or narrower windows. Fixed windows force a single compromise.

Adaptive window sizing lets each layer learn its own effective window size from the input, using soft masking for full differentiability. This enables the model to use smaller windows where local context suffices (saving compute) and larger windows where richer context is needed.

## Design Decisions

- **Per-layer learned predictors** over static hyperparameters — each layer learns its own window preferences from input content
- **Soft masking** over discrete buckets (Gumbel-softmax) — fully differentiable, no special estimators, reveals learned falloff curves for analysis. Gumbel-softmax is a viable hardening path for production later.
- **MAG first, MAL-compatible interface** — the predictor module is generic; MAG is the first consumer, MAL integration is a future effort using the same module
- **Task loss + efficiency regularization** — a lambda-weighted penalty encourages smaller windows unless task loss suffers, giving a tunable quality/efficiency tradeoff

## Architecture

### AdaptiveWindowPredictor Module

New file: `src/titans_mlx/adaptive_window.py`

A lightweight module, one instance per layer:

1. **Input:** hidden state `[batch, seq_len, dim]`
2. **Projection:** single linear layer `dim → 1`, producing a scalar per position
3. **Scaling:** sigmoid → scale to `[min_window, max_window]`, yielding `falloff_center` per query position `[batch, seq_len, 1]`
4. **Soft mask computation:** for each query-key pair:
   ```
   distance = query_pos - key_pos
   mask_weight = sigmoid(temperature * (falloff_center[query] - distance))
   ```
   Positions within the falloff center get weights near 1.0; positions beyond decay smoothly toward 0.0.
5. **Output:** soft mask `[batch, 1, seq_len, seq_len]` (broadcastable across heads) plus `falloff_centers` for regularization

**Parameters:**
- `min_window: int` — floor (default: 64)
- `max_window: int` — ceiling (default: `window_size`)
- `temperature: float` — sigmoid sharpness at boundary (default: 10.0). Higher = more hard-cutoff-like.

The causal constraint is enforced separately — this module only produces the window boundary modulation.

### SlidingWindowAttention Changes

File: `src/titans_mlx/attention.py`

1. `__call__` accepts optional `adaptive_mask: mx.array | None`
2. **No adaptive mask:** unchanged behavior, uses existing boolean mask via `get_sliding_window_mask`
3. **With adaptive mask:** convert soft mask to additive bias: `bias = (1 - adaptive_mask) * -1e9`, added to attention logits pre-softmax. Compatible with `mx.fast.scaled_dot_product_attention`.
4. Causal constraint still enforced — adaptive mask is AND'd with causal mask (positions violating causality stay at `-1e9`)
5. Persistent memory prefix remains fully attended — adaptive mask applies only to the non-prefix portion

**No changes to:**
- `window_size` parameter (stays as default/fallback and `max_window` ceiling)
- LRU-cached boolean masks (used when adaptive is off)
- MAC variant attention (full causal, not sliding window)

### MAG Block Integration

File: `src/titans_mlx/models.py`, `MAGBlock` class

1. `__init__`: if `config.adaptive_window`, instantiate `AdaptiveWindowPredictor` with config params
2. `core_forward`: before attention call, run predictor on normed hidden state:
   ```python
   normed = norm1(h)
   adaptive_mask = self.window_predictor(normed) if self.adaptive else None
   attn_out = attention(normed, prefix=persistent, adaptive_mask=adaptive_mask)
   ```
3. Each MAG block has its own predictor — per-layer variation is structural
4. When `adaptive_window: false`, no predictor instantiated, zero overhead

**Future MAL integration:** predictor runs on `norm2(h_mid)` (between memory output and attention call). Same module, different insertion point.

### Efficiency Regularization

Changes to training loop in pretrain.py (and sft.py, lora.py, dpo.py, rlvr.py):

1. Each `AdaptiveWindowPredictor` exposes its most recent `falloff_centers`
2. Model forward pass returns auxiliary dict with per-layer falloff centers when adaptive windowing is enabled
3. Regularization term: `reg_loss = lambda_window * mean(falloff_centers / max_window)` — normalized to [0, 1] scale
4. Summed across layers: `reg_loss = lambda * mean([mean(fc / max_w) for fc in layer_falloff_centers])`
5. Total loss: `task_loss + reg_loss`

**Logging:** mean effective window size per layer and regularization loss magnitude alongside task loss.

**When disabled:** no auxiliary output, no regularization, training loop unchanged.

### Config

New fields in `TitansConfig` (`src/titans_mlx/config.py`):

```python
adaptive_window: bool = False
adaptive_window_min: int = 64
adaptive_window_max: int | None = None   # defaults to window_size
adaptive_window_temperature: float = 10.0
adaptive_window_lambda: float = 0.01
```

### CLI Flags

All training scripts:

```
--adaptive-window
--adaptive-window-min <int>         (default: 64)
--adaptive-window-max <int>         (default: window_size)
--adaptive-window-temperature <float> (default: 10.0)
--adaptive-window-lambda <float>    (default: 0.01)
```

Inference scripts: same flags minus `--adaptive-window-lambda`.

### Backward Compatibility

- All flags default to off. Existing models load and run identically.
- A model trained with adaptive windowing includes predictor weights in checkpoint.
- Loading an adaptive checkpoint with `adaptive_window: false` ignores predictor weights (with warning).

## Testing

### Unit Tests (`test_adaptive_window.py`)

- Predictor produces masks with correct shape and value range [0, 1]
- Soft mask respects causality — future positions get zero weight regardless of predictor output
- `falloff_center` bounded within [min_window, max_window]
- Temperature controls boundary sharpness: high temp → near-binary, low temp → gradual falloff
- Persistent memory prefix bypassed — adaptive mask covers non-prefix only

### Integration Tests

- MAG forward pass with `adaptive_window: true` produces valid output with correct shapes
- MAG with `adaptive_window: false` produces identical output to current implementation (regression)
- Regularization loss is non-negative and scales with `lambda_window`
- Gradient flows through predictor — `predictor.weight.grad` is non-None after backward

### Diagnostic Test

- Small model trains for a handful of steps with adaptive windowing, loss decreases, per-layer window sizes diverge
- Checkpoint save/load round-trip with adaptive window weights

## Non-Goals

- Gumbel-softmax / discrete bucket selection (future hardening path)
- Per-token adaptation within a chunk (computationally expensive, unclear benefit over per-position soft masking)
- MAC variant support (uses full causal attention, not sliding window)
- MAL integration in this iteration (designed for, not implemented)
