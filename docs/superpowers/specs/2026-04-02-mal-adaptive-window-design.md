# MAL Adaptive Window Sizing Design Spec

**Date:** 2026-04-02
**Status:** Approved
**Scope:** Integrate existing adaptive window sizing into MALBlock
**Depends on:** `2026-04-02-adaptive-window-sizing-design.md` (complete)

## Motivation

The adaptive window sizing feature is implemented for MAG blocks. The `AdaptiveWindowPredictor` module and `SlidingWindowAttention` integration are generic — MAL uses the same attention class. This spec covers wiring the existing module into MALBlock.

## Design

### Predictor Input: `norm2(h_mid)`

In MAL, memory runs before attention. The predictor receives `norm2(h_mid)` — the memory-enriched hidden state that is the direct input to the attention layer. This lets the predictor factor in what the memory already contributed when deciding how much local context the attention layer needs.

This differs from MAG where the predictor runs on `norm1(h)` (the pre-attention input), because MAG's attention and memory operate in parallel rather than sequentially.

### MALBlock.__init__

After `self.dropout_p = config.dropout`, add:

```python
self._last_falloff_centers: mx.array | None = None
if config.adaptive_window:
    self.window_predictor = AdaptiveWindowPredictor(config)
```

Identical to MAGBlock's initialization.

### MALBlock.core_forward

Between `normed_mid = self.norm2(h_mid)` and `attn_out = self.attention(...)`, insert:

```python
# Adaptive window: predict from memory-enriched hidden state
# NOTE: In multi-chunk sequences, _last_falloff_centers retains only the
# final chunk's values (same limitation as MAG).
adaptive_mask = None
if hasattr(self, "window_predictor"):
    adaptive_mask, self._last_falloff_centers = self.window_predictor(normed_mid)

attn_out = self.attention(normed_mid, prefix=persistent, adaptive_mask=adaptive_mask)
```

### What Doesn't Change

- `AdaptiveWindowPredictor` — reused as-is from `adaptive_window.py`
- `SlidingWindowAttention` — already accepts `adaptive_mask` parameter
- `compute_window_regularization` — already collects `_last_falloff_centers` from any block via `getattr`
- Config fields, CLI args, checkpoint metadata — already wired across all scripts
- Training loop regularization — iterates `model.blocks` generically, works for MAL blocks
- `__init__.py` exports — no new exports needed

## Testing

### Unit Tests

- MALBlock forward with `adaptive_window=True` produces valid output with correct shapes
- MALBlock instantiates `window_predictor` when enabled
- MALBlock does not instantiate `window_predictor` when disabled
- MALBlock exposes `_last_falloff_centers` after forward pass

### Integration Tests

- TitansMAL end-to-end forward (single chunk)
- TitansMAL multi-chunk forward
- Collect falloff centers from all MAL blocks after forward pass
- MALBlock without adaptive window produces identical output (regression)

## Non-Goals

- Changes to `AdaptiveWindowPredictor` module
- Changes to `SlidingWindowAttention`
- Changes to config, CLI, or training scripts
- MAC variant support (uses `SegmentedAttention`, not `SlidingWindowAttention`)
