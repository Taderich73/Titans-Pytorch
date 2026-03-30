# Error Handling & Defensive Programming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three bugs, add shape validation guards at module entry points, and extract magic numbers into named constants.

**Architecture:** Three sequential passes — bug fixes first (highest value, independently revertable), then shape validation (entry-point guards + debug assertions), then named constants extraction (pure refactor, no behavioral change). Each pass gets its own commit.

**Tech Stack:** Python 3.12, MLX, pytest

---

## File Map

| File | Changes |
|------|---------|
| `src/titans_mlx/memory.py` | Bug 1 fix (mx.sign), debug assertions, named constants |
| `src/titans_mlx/models.py` | Bug 2 fix (weights[0] guard), debug assertion on W |
| `src/titans_mlx/mca.py` | Bug 3 fix (memory_weights ndim), entry-point guards on x, store self.dim |
| `src/titans_mlx/attn_res.py` | Entry-point guard on sources, use config for logit clip |
| `src/titans_mlx/config.py` | Add gate_decay_bias_init, attnres_logit_clip fields |
| `tests/test_defensive.py` | New test file for all bug fix and validation tests |

---

## Task 1: Bug Fix Tests (TDD — write failing tests first)

**Files:**
- Create: `tests/test_defensive.py`

- [ ] **Step 1: Write bug fix tests**

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for error handling and defensive programming guards."""

import mlx.core as mx
import pytest

from titans_mlx.config import TitansConfig
from titans_mlx.memory import MemoryState, NeuralLongTermMemory


def _linear_memory_config() -> TitansConfig:
    """Config with 1-layer (linear) memory for testing parallel update."""
    return TitansConfig(
        dim=32,
        num_heads=2,
        num_layers=1,
        vocab_size=50,
        num_memory_layers=1,
        memory_hidden_mult=2.0,
        use_conv=False,
        use_rope=False,
        chunk_size=8,
        window_size=8,
        max_seq_len=64,
    )


def _mca_config(**kwargs) -> TitansConfig:
    """Config with MCA enabled."""
    defaults = dict(
        dim=64,
        num_heads=4,
        num_layers=6,
        vocab_size=256,
        use_mca=True,
        mca_num_heads=4,
        num_memory_layers=2,
        memory_hidden_mult=2.0,
    )
    defaults.update(kwargs)
    return TitansConfig(**defaults)


class TestBugFixes:
    """Tests for the three identified bugs."""

    def test_degenerate_geometric_series_no_nan(self) -> None:
        """Bug 1: parallel update with decay ~ eta should not produce NaN/Inf."""
        config = _linear_memory_config()
        mem = NeuralLongTermMemory(config)
        state = mem.init_state(batch_size=2)

        keys = mx.random.normal((2, 8, 32))
        values = mx.random.normal((2, 8, 32))

        # Force decay ~ eta: decay = 1 - alpha, so set alpha such that
        # 1 - sigmoid(alpha_logit) ~ sigmoid(eta_logit) * momentum
        # Use exact equality to hit degenerate branch
        alpha = mx.array(0.5)  # decay = 0.5
        eta = mx.array(0.5)    # eta = 0.5, so decay == eta

        new_state = mem._parallel_memory_update_linear(
            keys, values, state, alpha, theta=mx.array(0.05), eta=eta
        )
        mx.eval(new_state.weights[0])
        mx.eval(new_state.momentum[0])

        assert not mx.any(mx.isnan(new_state.weights[0])).item()
        assert not mx.any(mx.isinf(new_state.weights[0])).item()
        assert not mx.any(mx.isnan(new_state.momentum[0])).item()
        assert not mx.any(mx.isinf(new_state.momentum[0])).item()

    def test_mca_forward_empty_weights_raises(self) -> None:
        """Bug 2: _mca_forward with empty weights list should raise ValueError."""
        from titans_mlx.models import _mca_forward

        config = _mca_config()

        from titans_mlx.mca import MemoryCrossAttention

        class FakeBlock:
            has_mca = True
            mca = MemoryCrossAttention(config)

        block = FakeBlock()
        h = mx.random.normal((2, 16, 64))
        empty_state = MemoryState(weights=[], momentum=[])

        with pytest.raises(ValueError, match="non-empty memory weights"):
            _mca_forward(block, h, empty_state)

    def test_mca_3d_memory_weights_raises(self) -> None:
        """Bug 3: MCA with 3D memory_weights should raise ValueError."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W_3d = mx.random.normal((2, 128, 64))  # 3D — wrong

        with pytest.raises(ValueError, match="memory_weights"):
            mca(x, W_3d)


class TestShapeValidation:
    """Tests for entry-point shape guards."""

    def test_mca_2d_input_raises(self) -> None:
        """MCA should reject 2D input tensor."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x_2d = mx.random.normal((16, 64))  # missing batch dim
        W = mx.random.normal((128, 64))

        with pytest.raises(ValueError, match="3D"):
            mca(x_2d, W)

    def test_mca_wrong_dim_raises(self) -> None:
        """MCA should reject input with wrong last dimension."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x_wrong = mx.random.normal((2, 16, 32))  # dim=32 != expected 64
        W = mx.random.normal((128, 64))

        with pytest.raises(ValueError, match="dim"):
            mca(x_wrong, W)

    def test_attnres_empty_sources_raises(self) -> None:
        """BlockAttnRes should reject empty blocks with no partial_block."""
        from titans_mlx.attn_res import BlockAttnRes

        attn_res = BlockAttnRes(dim=64)

        with pytest.raises(ValueError, match="at least one source"):
            attn_res(blocks=[], partial_block=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_defensive.py -v 2>&1 | tail -20`

Expected: All 6 tests FAIL — the guards don't exist yet. The degenerate test may pass (existing code handles it), but the ValueError tests should all fail with either no error or wrong error type.

---

## Task 2: Bug Fix — mx.sign fragility (memory.py)

**Files:**
- Modify: `src/titans_mlx/memory.py:990`

- [ ] **Step 1: Fix the fragile mx.sign call**

In `src/titans_mlx/memory.py`, replace line 990:

```python
# Before:
            mx.maximum(abs_diff, mx.array(1e-8)) * mx.sign(diff + 1e-12),

# After:
            mx.maximum(abs_diff, mx.array(1e-8)) * mx.sign(diff),
```

- [ ] **Step 2: Run degenerate test to verify it passes**

Run: `python -m pytest tests/test_defensive.py::TestBugFixes::test_degenerate_geometric_series_no_nan -v`

Expected: PASS

---

## Task 3: Bug Fix — unguarded weights[0] (models.py)

**Files:**
- Modify: `src/titans_mlx/models.py:44-51`

- [ ] **Step 1: Add weights validation to _mca_forward**

In `src/titans_mlx/models.py`, replace the `_mca_forward` function body (lines 44-51):

```python
def _mca_forward(block: nn.Module, h: mx.array, mem_state) -> mx.array:
    """MCA sub-layer: cross-attend to NeuralLTM weight rows (shared across MAC/MAG/MAL)."""
    weights = (
        mem_state.global_state.weights
        if hasattr(mem_state, "global_state")
        else mem_state.weights
    )
    if not weights:
        raise ValueError(
            "_mca_forward requires non-empty memory weights. "
            "Check that NeuralLTM has num_memory_layers >= 1."
        )
    W = weights[0]
    W = mx.stop_gradient(W)
    return block.mca(h, W)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_defensive.py::TestBugFixes::test_mca_forward_empty_weights_raises -v`

Expected: PASS

---

## Task 4: Bug Fix — MCA memory_weights shape validation (mca.py)

**Files:**
- Modify: `src/titans_mlx/mca.py:32-36` (add `self.dim`)
- Modify: `src/titans_mlx/mca.py:49-62` (add guards)

- [ ] **Step 1: Store dim as instance attribute**

In `src/titans_mlx/mca.py`, in `__init__` after line 34 (`dim = config.dim`), add:

```python
        self.dim = dim
```

So lines 33-36 become:

```python
        dim = config.dim
        self.dim = dim
        self.num_heads = config.mca_num_heads
        self.head_dim = dim // self.num_heads
```

- [ ] **Step 2: Add entry-point validation to __call__**

In `src/titans_mlx/mca.py`, at the top of `__call__` (after the docstring, before `B, T, dim = x.shape`), add:

```python
        if x.ndim != 3:
            raise ValueError(
                f"MCA expects x with shape [B, T, dim], "
                f"got {x.ndim}D tensor with shape {x.shape}"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"MCA expects x with last dim={self.dim}, "
                f"got {x.shape[-1]}"
            )
        if memory_weights.ndim != 2:
            raise ValueError(
                f"MCA expects memory_weights with shape [num_rows, dim], "
                f"got {memory_weights.ndim}D tensor with shape {memory_weights.shape}"
            )
```

- [ ] **Step 3: Run all MCA bug/validation tests**

Run: `python -m pytest tests/test_defensive.py::TestBugFixes::test_mca_3d_memory_weights_raises tests/test_defensive.py::TestShapeValidation::test_mca_2d_input_raises tests/test_defensive.py::TestShapeValidation::test_mca_wrong_dim_raises -v`

Expected: All 3 PASS

- [ ] **Step 4: Run existing MCA tests for regression**

Run: `python -m pytest tests/test_mca.py -v`

Expected: All existing tests PASS

---

## Task 5: Shape Validation — BlockAttnRes guard (attn_res.py)

**Files:**
- Modify: `src/titans_mlx/attn_res.py:44-66`

- [ ] **Step 1: Add sources validation**

In `src/titans_mlx/attn_res.py`, in `__call__`, after building the `sources` list (after line 65 `sources.append(partial_block)`) and before the single-source check (line 68 `if len(sources) == 1:`), add:

```python
        if not sources:
            raise ValueError(
                "BlockAttnRes requires at least one source: "
                "pass non-empty blocks or a partial_block"
            )
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_defensive.py::TestShapeValidation::test_attnres_empty_sources_raises -v`

Expected: PASS

---

## Task 6: Shape Validation — debug assertions (memory.py, models.py)

**Files:**
- Modify: `src/titans_mlx/memory.py:963` (before `B, S, D = keys.shape`)
- Modify: `src/titans_mlx/memory.py:733` (before `batch_size, seq_len = keys.shape[0], keys.shape[1]`)
- Modify: `src/titans_mlx/models.py` (after `W = weights[0]`)

- [ ] **Step 1: Add assert in _parallel_memory_update_linear**

In `src/titans_mlx/memory.py`, before line 963 (`B, S, D = keys.shape`), add:

```python
        assert keys.ndim == 3, f"Expected 3D keys [B, S, D], got {keys.ndim}D"
```

- [ ] **Step 2: Add assert in _compute_gradients_deep**

In `src/titans_mlx/memory.py`, before line 733 (`batch_size, seq_len = keys.shape[0], keys.shape[1]`), add:

```python
        assert keys.ndim == 3, f"Expected 3D keys [B, S, D], got {keys.ndim}D"
```

- [ ] **Step 3: Add assert in _mca_forward**

In `src/titans_mlx/models.py`, after the `W = weights[0]` line and before `W = mx.stop_gradient(W)`, add:

```python
    assert W.ndim == 2, f"Expected 2D weight matrix, got {W.ndim}D"
```

- [ ] **Step 4: Commit Pass 1 + Pass 2**

```bash
git add src/titans_mlx/memory.py src/titans_mlx/models.py src/titans_mlx/mca.py src/titans_mlx/attn_res.py tests/test_defensive.py
git commit -m "fix: add shape validation guards and fix mx.sign fragility

- Fix fragile mx.sign(diff + 1e-12) -> mx.sign(diff) in geometric series
- Guard weights[0] access in _mca_forward with ValueError
- Add entry-point shape validation to MCA.__call__ and BlockAttnRes.__call__
- Add debug assertions at internal shape unpacking points
- Add test_defensive.py with bug fix and validation tests"
```

- [ ] **Step 5: Run full test suite for regression**

Run: `python -m pytest tests/ -v 2>&1 | tail -30`

Expected: All tests PASS (existing + new)

---

## Task 7: Named Constants — module-level constants (memory.py)

**Files:**
- Modify: `src/titans_mlx/memory.py` (top of file + inline usages)

- [ ] **Step 1: Add named constants near top of file**

In `src/titans_mlx/memory.py`, after the imports (after line 30 `import numpy as np`), add:

```python

# Numerical stability constants (implementation details, not tunable)
_L2_NORM_EPS: float = 1e-8
_DEGENERATE_THRESHOLD: float = 1e-6
```

- [ ] **Step 2: Replace inline 1e-8 usages**

There are 4 occurrences of `1e-8` in memory.py. Replace each:

Line 851:
```python
# Before:
            q_f32 / mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-8)
# After:
            q_f32 / mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True) + _L2_NORM_EPS)
```

Line 854:
```python
# Before:
            k_f32 / mx.sqrt(mx.sum(k_f32 * k_f32, axis=-1, keepdims=True) + 1e-8)
# After:
            k_f32 / mx.sqrt(mx.sum(k_f32 * k_f32, axis=-1, keepdims=True) + _L2_NORM_EPS)
```

Line 990 (the `mx.maximum` floor):
```python
# Before:
            mx.maximum(abs_diff, mx.array(1e-8)) * mx.sign(diff),
# After:
            mx.maximum(abs_diff, mx.array(_L2_NORM_EPS)) * mx.sign(diff),
```

Line 1089:
```python
# Before:
            q_f32 / mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-8)
# After:
            q_f32 / mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True) + _L2_NORM_EPS)
```

- [ ] **Step 3: Replace inline 1e-6 usage**

Line 986:
```python
# Before:
        is_degenerate = abs_diff < 1e-6
# After:
        is_degenerate = abs_diff < _DEGENERATE_THRESHOLD
```

- [ ] **Step 4: Run tests for regression**

Run: `python -m pytest tests/test_memory.py tests/test_defensive.py -v`

Expected: All PASS — no behavioral change

---

## Task 8: Named Constants — config fields (config.py, memory.py, attn_res.py)

**Files:**
- Modify: `src/titans_mlx/config.py` (add fields)
- Modify: `src/titans_mlx/memory.py:638` (use config field)
- Modify: `src/titans_mlx/attn_res.py:36-42,85` (accept config, use clip field)

- [ ] **Step 1: Add config fields**

In `src/titans_mlx/config.py`, after line 101 (`mca_gate_bias_init: float = -3.0`), add:

```python

    # Gate initialization
    gate_decay_bias_init: float = -6.0  # sigmoid(-6) ~ 0.0025 retention

    # AttnRes numerical stability
    attnres_logit_clip: float = 30.0  # symmetric clip to +/- value before softmax
```

- [ ] **Step 2: Use gate_decay_bias_init in memory.py**

In `src/titans_mlx/memory.py`, replace line 638:

```python
# Before:
        self.gate_decay_proj.bias = mx.array([-6.0])
# After:
        self.gate_decay_proj.bias = mx.array([self.config.gate_decay_bias_init])
```

- [ ] **Step 3: Accept config in BlockAttnRes and use attnres_logit_clip**

In `src/titans_mlx/attn_res.py`, modify `BlockAttnRes.__init__` to accept and store the clip value:

```python
# Before:
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

# After:
    def __init__(self, dim: int, logit_clip: float = 30.0) -> None:
        super().__init__()
        self.dim = dim
        self.logit_clip = logit_clip
```

In `__call__`, replace line 85:

```python
# Before:
        logits = mx.clip(logits, -30.0, 30.0)
# After:
        logits = mx.clip(logits, -self.logit_clip, self.logit_clip)
```

- [ ] **Step 4: Pass config value at all BlockAttnRes construction sites**

In `src/titans_mlx/models.py`, update all `BlockAttnRes(config.dim)` calls to pass the clip value. There are 7 call sites (search for `BlockAttnRes(config.dim)`):

```python
# Before (all sites):
            block.attn_res_mca = BlockAttnRes(config.dim)
            self.attn_res_core = BlockAttnRes(config.dim)
            self.attn_res_ffn = BlockAttnRes(config.dim)

# After (all sites):
            block.attn_res_mca = BlockAttnRes(config.dim, logit_clip=config.attnres_logit_clip)
            self.attn_res_core = BlockAttnRes(config.dim, logit_clip=config.attnres_logit_clip)
            self.attn_res_ffn = BlockAttnRes(config.dim, logit_clip=config.attnres_logit_clip)
```

- [ ] **Step 5: Commit Pass 3**

```bash
git add src/titans_mlx/config.py src/titans_mlx/memory.py src/titans_mlx/attn_res.py src/titans_mlx/models.py
git commit -m "refactor: extract magic numbers into named constants and config fields

- Add _L2_NORM_EPS and _DEGENERATE_THRESHOLD as module constants in memory.py
- Add gate_decay_bias_init and attnres_logit_clip to TitansConfig
- Replace all hardcoded epsilon/threshold/clip values with named references"
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -30`

Expected: All tests PASS

---

## Task 9: Final Verification

- [ ] **Step 1: Grep for remaining hardcoded values**

Run:
```bash
grep -n '1e-8\|1e-6\|1e-12\|\-6\.0\|\-30\.0\|30\.0' src/titans_mlx/memory.py src/titans_mlx/attn_res.py src/titans_mlx/config.py
```

Expected: Only hits in config.py defaults and the named constant definitions. No remaining inline magic numbers in memory.py or attn_res.py.

- [ ] **Step 2: Run full test suite one final time**

Run: `python -m pytest tests/ -v`

Expected: All tests PASS, zero failures.
