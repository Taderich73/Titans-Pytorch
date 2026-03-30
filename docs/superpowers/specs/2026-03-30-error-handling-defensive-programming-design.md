# Error Handling & Defensive Programming

**Date:** 2026-03-30
**Scope:** Bug fixes, shape validation, named constant extraction
**Approach:** Three sequential passes (category-first)

---

## Context

A code review scored error handling at 6/10 and flagged several issues across memory.py, mca.py, models.py, and attn_res.py. After investigation, three items are confirmed bugs, two are code health improvements, and two reported issues (RoPE offset asymmetry, AttnRes shape inconsistency) are intentional designs.

## Pass 1: Bug Fixes

### Bug 1 — Fragile `mx.sign(diff + 1e-12)` in geometric series computation

**File:** `memory.py:990` (in `_parallel_memory_update_linear`)

**Current code:**
```python
is_degenerate = abs_diff < 1e-6
safe_diff = mx.where(
    is_degenerate,
    mx.array(1.0),
    mx.maximum(abs_diff, mx.array(1e-8)) * mx.sign(diff + 1e-12),
)
```

**Problem:** `mx.sign(diff + 1e-12)` is fragile by inspection. If `diff` is close to `-1e-12`, the offset cancels and `mx.sign(0)` returns 0, collapsing the denominator.

**Why it doesn't currently fail:** The `is_degenerate` guard catches all `|diff| < 1e-6`, so the `mx.sign` branch only executes when `|diff| >= 1e-6`, where the `1e-12` offset can't flip the sign.

**Fix:** Replace `mx.sign(diff + 1e-12)` with `mx.sign(diff)`. In the non-degenerate branch, `|diff| >= 1e-6` guarantees `diff != 0`, making `mx.sign(diff)` safe without the offset. This removes a fragile assumption while preserving identical behavior.

The `1e-8` floor in `mx.maximum(abs_diff, mx.array(1e-8))` on the same line serves the same numerical stability purpose as `_L2_NORM_EPS` and will use that constant in Pass 3.

### Bug 2 — Unguarded `weights[0]` access in `_mca_forward`

**File:** `models.py:47,49`

**Current code:**
```python
if hasattr(mem_state, "global_state"):
    W = mem_state.global_state.weights[0]
else:
    W = mem_state.weights[0]
```

**Problem:** If the weights list is empty (e.g., `num_memory_layers=0` or malformed state), this raises an `IndexError` with no context about what went wrong.

**Fix:** Validate the weights list before indexing. Raise `ValueError` with a message explaining what's expected:
```python
weights = mem_state.global_state.weights if hasattr(mem_state, "global_state") else mem_state.weights
if not weights:
    raise ValueError(
        "_mca_forward requires non-empty memory weights. "
        "Check that NeuralLTM has num_memory_layers >= 1."
    )
W = weights[0]
```

### Bug 3 — Unvalidated `memory_weights` shape in MCA

**File:** `mca.py:49-62`

**Problem:** `memory_weights` is documented as `[num_rows, dim]` (2D) but nothing enforces this. A 3D tensor (e.g., batched weights) silently broadcasts through `nn.Linear` projections and produces incorrect K/V matrices without error.

**Fix:** Add shape validation at the top of `MCA.__call__`:
```python
if memory_weights.ndim != 2:
    raise ValueError(
        f"MCA expects memory_weights with shape [num_rows, dim], "
        f"got {memory_weights.ndim}D tensor with shape {memory_weights.shape}"
    )
```

## Pass 2: Shape Validation

### Entry-point guards (raise ValueError)

These go at the top of public `__call__` methods:

**MCA.__call__** (`mca.py`):
- `x.ndim == 3` (batch, seq, dim)
- `x.shape[-1] == self.dim`
- `memory_weights.ndim == 2` (covered by Bug 3 fix)

**BlockAttnRes.__call__** (`attn_res.py`):
- `len(sources) >= 1` — at least one source required
- Each source is 3D with consistent batch and seq dimensions

### Debug-only assertions (assert statements)

These go at internal junctures where shape unpacking would otherwise give confusing errors:

**`_parallel_memory_update_linear`** (`memory.py`):
```python
assert keys.ndim == 3, f"Expected 3D keys [B, S, D], got {keys.ndim}D"
```
Before `B, S, D = keys.shape`.

**`_compute_gradients_deep`** (`memory.py`):
Same pattern for keys shape unpacking.

**`_mca_forward`** (`models.py`):
```python
assert W.ndim == 2, f"Expected 2D weight matrix, got {W.ndim}D"
```
After extracting W, before passing to MCA (belt-and-suspenders with the MCA entry guard).

## Pass 3: Named Constants

### Module-level constants (implementation details)

In `memory.py`, top of file:
```python
_L2_NORM_EPS: float = 1e-8
_DEGENERATE_THRESHOLD: float = 1e-6
```

Replace all inline usages of these values with the named constants.

### Config fields (behavioral parameters)

In `config.py`, add to `TitansConfig`:

```python
# Gate initialization
gate_decay_bias_init: float = -6.0  # sigmoid(-6) ~ 0.0025 retention

# AttnRes numerical stability
attnres_logit_clip: float = 30.0  # symmetric clip to +/- value before softmax
```

Update usages:
- `memory.py:638` — replace `-6.0` with `self.config.gate_decay_bias_init`
- `attn_res.py:85` — replace `-30.0, 30.0` with `±config.attnres_logit_clip`

## Testing

### New tests (in `test_defensive.py` or appended to existing files per convention)

**Bug fix tests:**
- `test_degenerate_geometric_series`: Feed `_parallel_memory_update_linear` with `decay ≈ eta` (within 1e-6). Confirm no NaN/Inf in output and result matches expected degenerate-case behavior.
- `test_mca_forward_empty_weights`: Confirm `_mca_forward` raises `ValueError` when weights list is empty.
- `test_mca_3d_memory_weights`: Confirm `MCA.__call__` raises `ValueError` when `memory_weights` is 3D.

**Shape validation tests:**
- `test_mca_wrong_input_dims`: Pass 2D `x` to `MCA.__call__`, expect `ValueError`.
- `test_mca_wrong_input_dim_size`: Pass `x` with wrong last dimension, expect `ValueError`.
- `test_attnres_empty_sources`: Pass empty list to `BlockAttnRes.__call__`, expect `ValueError`.

**Constants extraction:**
- No new tests. Existing test suite should pass unchanged, confirming no behavioral regression.

## Non-Goals

- No try-except in core forward/backward passes (fail fast during training)
- No RoPE offset changes (confirmed intentional asymmetry in SlidingWindowAttention)
- No AttnRes shape change (`(B,T,1)` for single source is intentional API consistency)
- No changes to file I/O error handling (already well-guarded)
- No new logging or warning infrastructure
- No changes to existing tests unless config field additions cause breakage

## Files Modified

| File | Pass | Changes |
|------|------|---------|
| `memory.py` | 1, 2, 3 | Fix mx.sign, add assert guards, extract constants |
| `models.py` | 1, 2 | Guard weights[0] access, add assert on W shape |
| `mca.py` | 1, 2 | Validate memory_weights shape, validate x shape |
| `attn_res.py` | 2, 3 | Validate sources, use config for logit clip |
| `config.py` | 3 | Add gate_decay_bias_init, attnres_logit_clip |
| `test_defensive.py` | 1, 2 | New test file for bug fix and validation tests |
