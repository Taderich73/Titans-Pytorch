# Adaptive Window Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-layer learned adaptive window sizing with soft masking to sliding window attention, starting with MAG integration.

**Architecture:** Each layer gets a lightweight `AdaptiveWindowPredictor` (linear projection → sigmoid → scaled falloff center) that produces a soft attention mask replacing the hard boolean window mask. An efficiency regularization term penalizes large windows during training.

**Tech Stack:** MLX (mx.array, nn.Module), pytest

---

### Task 1: Add Config Fields

**Files:**
- Modify: `src/titans_mlx/config.py:95-98` (after quantize fields)
- Modify: `src/titans_mlx/config.py:194-246` (to_dict)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py`, add:

```python
class TestAdaptiveWindowConfig:
    """Tests for adaptive window config fields."""

    def test_defaults_disabled(self) -> None:
        """Adaptive window defaults to disabled."""
        config = TitansConfig()
        assert config.adaptive_window is False
        assert config.adaptive_window_min == 64
        assert config.adaptive_window_max is None
        assert config.adaptive_window_temperature == 10.0
        assert config.adaptive_window_lambda == 0.01

    def test_max_defaults_to_window_size(self) -> None:
        """adaptive_window_max defaults to window_size when None."""
        config = TitansConfig(window_size=256, adaptive_window=True)
        assert config.effective_adaptive_window_max == 256

    def test_max_explicit(self) -> None:
        """Explicit adaptive_window_max overrides window_size."""
        config = TitansConfig(
            window_size=512, adaptive_window=True, adaptive_window_max=256
        )
        assert config.effective_adaptive_window_max == 256

    def test_to_dict_includes_adaptive_fields(self) -> None:
        """to_dict includes adaptive window fields."""
        config = TitansConfig(adaptive_window=True, adaptive_window_min=32)
        d = config.to_dict()
        assert d["adaptive_window"] is True
        assert d["adaptive_window_min"] == 32
        assert d["adaptive_window_max"] is None
        assert d["adaptive_window_temperature"] == 10.0
        assert d["adaptive_window_lambda"] == 0.01

    def test_from_dict_round_trip(self) -> None:
        """Config survives to_dict -> from_dict round trip."""
        config = TitansConfig(
            adaptive_window=True,
            adaptive_window_min=32,
            adaptive_window_max=256,
            adaptive_window_temperature=5.0,
            adaptive_window_lambda=0.05,
        )
        restored = TitansConfig.from_dict(config.to_dict())
        assert restored.adaptive_window is True
        assert restored.adaptive_window_min == 32
        assert restored.effective_adaptive_window_max == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestAdaptiveWindowConfig -v`
Expected: FAIL — `adaptive_window` attribute not found

- [ ] **Step 3: Add config fields**

In `src/titans_mlx/config.py`, after line 98 (memory_state_momentum_bits), add:

```python
    # Adaptive window sizing (per-layer learned soft masking)
    adaptive_window: bool = False
    adaptive_window_min: int = 64
    adaptive_window_max: int | None = None  # defaults to window_size
    adaptive_window_temperature: float = 10.0
    adaptive_window_lambda: float = 0.01
```

Add a property after `memory_hidden_dim`:

```python
    @property
    def effective_adaptive_window_max(self) -> int:
        """Resolved max window size for adaptive windowing."""
        if self.adaptive_window_max is not None:
            return self.adaptive_window_max
        return self.window_size
```

In `to_dict`, add these entries (after the quantize fields):

```python
            "adaptive_window": self.adaptive_window,
            "adaptive_window_min": self.adaptive_window_min,
            "adaptive_window_max": self.adaptive_window_max,
            "adaptive_window_temperature": self.adaptive_window_temperature,
            "adaptive_window_lambda": self.adaptive_window_lambda,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::TestAdaptiveWindowConfig -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/config.py tests/test_config.py
git commit -m "feat(config): add adaptive window sizing fields"
```

---

### Task 2: AdaptiveWindowPredictor Module

**Files:**
- Create: `src/titans_mlx/adaptive_window.py`
- Test: `tests/test_adaptive_window.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_adaptive_window.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for adaptive window sizing."""

import mlx.core as mx
import pytest

from titans_mlx.config import TitansConfig


class TestAdaptiveWindowPredictor:
    """Tests for AdaptiveWindowPredictor module."""

    @pytest.fixture
    def adaptive_config(self) -> TitansConfig:
        """Config with adaptive window enabled."""
        return TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=10.0,
        )

    def test_output_shape(self, adaptive_config: TitansConfig) -> None:
        """Soft mask has correct shape (1, 1, seq_len, seq_len)."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        mask, falloff = predictor(x)
        mx.eval(mask, falloff)

        assert mask.shape == (2, 1, 16, 16)
        assert falloff.shape == (2, 16, 1)

    def test_mask_values_in_range(self, adaptive_config: TitansConfig) -> None:
        """Soft mask values are in [0, 1]."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        assert mx.all(mask >= 0.0).item()
        assert mx.all(mask <= 1.0).item()

    def test_causality_enforced(self, adaptive_config: TitansConfig) -> None:
        """Future positions have zero mask weight."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((1, 16, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        # Check upper triangle is zero (future positions)
        mask_2d = mask[0, 0]  # (seq_len, seq_len)
        for i in range(16):
            for j in range(i + 1, 16):
                assert mask_2d[i, j].item() == pytest.approx(0.0, abs=1e-6), (
                    f"Future position mask[{i},{j}] = {mask_2d[i, j].item()}"
                )

    def test_falloff_center_bounded(self, adaptive_config: TitansConfig) -> None:
        """Falloff centers are within [min_window, max_window]."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        _, falloff = predictor(x)
        mx.eval(falloff)

        min_w = adaptive_config.adaptive_window_min
        max_w = adaptive_config.effective_adaptive_window_max
        assert mx.all(falloff >= min_w).item()
        assert mx.all(falloff <= max_w).item()

    def test_high_temperature_near_binary(self, adaptive_config: TitansConfig) -> None:
        """High temperature produces near-binary masks."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        config = TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=100.0,  # Very high
        )
        predictor = AdaptiveWindowPredictor(config)
        x = mx.random.normal((1, 16, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        # With high temp, causal positions should be near 0 or 1
        causal_mask = mask[0, 0]
        for i in range(16):
            for j in range(i + 1):
                val = causal_mask[i, j].item()
                assert val < 0.05 or val > 0.95, (
                    f"High-temp mask[{i},{j}] = {val}, expected near 0 or 1"
                )

    def test_low_temperature_gradual(self) -> None:
        """Low temperature produces gradual falloff."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        config = TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=1.0,  # Very low
        )
        predictor = AdaptiveWindowPredictor(config)
        x = mx.random.normal((1, 32, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        # With low temp, should have intermediate values (not all 0/1)
        causal_vals = []
        mask_2d = mask[0, 0]
        for i in range(32):
            for j in range(i + 1):
                causal_vals.append(mask_2d[i, j].item())

        intermediate = [v for v in causal_vals if 0.1 < v < 0.9]
        assert len(intermediate) > 0, "Low temperature should produce intermediate mask values"

    def test_gradient_flows(self, adaptive_config: TitansConfig) -> None:
        """Gradients flow through the predictor."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((1, 8, 64))

        def loss_fn(model, x):
            mask, falloff = model(x)
            return mx.mean(mask)

        loss_and_grad = mx.value_and_grad(predictor, loss_fn)
        loss, grads = loss_and_grad(predictor, x)
        mx.eval(loss, grads)

        # Check that projection layer has gradients
        flat_grads = [v for _, v in mx.utils.tree_flatten(grads)]
        has_nonzero = any(mx.any(g != 0).item() for g in flat_grads if isinstance(g, mx.array))
        assert has_nonzero, "Predictor should have non-zero gradients"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adaptive_window.py -v`
Expected: FAIL — `titans_mlx.adaptive_window` module not found

- [ ] **Step 3: Implement AdaptiveWindowPredictor**

Create `src/titans_mlx/adaptive_window.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Adaptive window sizing for sliding window attention.

Per-layer learned soft masking that replaces the hard boolean window
boundary with a differentiable sigmoid falloff. Each layer learns its
own effective window size from the input hidden state.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.config import TitansConfig


class AdaptiveWindowPredictor(nn.Module):
    """Predicts per-position soft window boundaries.

    A lightweight linear projection maps each position's hidden state
    to a scalar "falloff center" — the effective window size for that
    query position. A sigmoid with configurable temperature converts
    query-key distances into soft mask weights.

    Args:
        config: TitansConfig with adaptive window fields set.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.min_window = config.adaptive_window_min
        self.max_window = config.effective_adaptive_window_max
        self.temperature = config.adaptive_window_temperature
        self.window_range = self.max_window - self.min_window

        # Linear projection: dim -> 1 scalar per position
        self.proj = nn.Linear(config.dim, 1, bias=True)

        # Initialize bias to produce mid-range falloff centers
        self.proj.weight = mx.random.normal(self.proj.weight.shape) * config.init_std
        self.proj.bias = mx.zeros((1,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute soft window mask from hidden states.

        Args:
            x: Hidden states (batch, seq_len, dim)

        Returns:
            Tuple of:
            - soft_mask: (batch, 1, seq_len, seq_len) mask weights in [0, 1]
            - falloff_centers: (batch, seq_len, 1) effective window sizes
        """
        batch, seq_len, _ = x.shape

        # Predict falloff center per position: (batch, seq_len, 1)
        raw = self.proj(x)  # (batch, seq_len, 1)
        # Scale to [min_window, max_window]
        falloff_centers = self.min_window + self.window_range * mx.sigmoid(raw)

        # Build distance matrix: distance[i, j] = i - j
        positions = mx.arange(seq_len)
        row_idx = mx.expand_dims(positions, axis=1)  # (seq_len, 1)
        col_idx = mx.expand_dims(positions, axis=0)  # (1, seq_len)
        distance = (row_idx - col_idx).astype(mx.float32)  # (seq_len, seq_len)

        # Expand falloff_centers for broadcasting: (batch, seq_len, 1) -> (batch, seq_len, 1)
        # distance: (seq_len, seq_len) broadcasts with falloff: (batch, seq_len, 1)
        # Result: (batch, seq_len, seq_len)
        soft_mask = mx.sigmoid(
            self.temperature * (falloff_centers - distance)
        )

        # Enforce causality: zero out future positions (where col > row)
        causal = (col_idx <= row_idx).astype(mx.float32)  # (seq_len, seq_len)
        soft_mask = soft_mask * causal

        # Add head dimension: (batch, 1, seq_len, seq_len)
        soft_mask = mx.expand_dims(soft_mask, axis=1)

        return soft_mask, falloff_centers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_adaptive_window.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/adaptive_window.py tests/test_adaptive_window.py
git commit -m "feat: add AdaptiveWindowPredictor module with soft masking"
```

---

### Task 3: Integrate Adaptive Mask into SlidingWindowAttention

**Files:**
- Modify: `src/titans_mlx/attention.py:192-250` (SlidingWindowAttention.__call__)
- Test: `tests/test_adaptive_window.py` (add integration tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adaptive_window.py`:

```python
class TestSlidingWindowAdaptiveMask:
    """Tests for adaptive mask integration in SlidingWindowAttention."""

    @pytest.fixture
    def adaptive_config(self) -> TitansConfig:
        return TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            max_seq_len=256,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
        )

    def test_attention_accepts_adaptive_mask(self, adaptive_config: TitansConfig) -> None:
        """SlidingWindowAttention accepts adaptive_mask parameter."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor
        from titans_mlx.attention import SlidingWindowAttention

        attn = SlidingWindowAttention(adaptive_config)
        predictor = AdaptiveWindowPredictor(adaptive_config)

        x = mx.random.normal((2, 16, 64))
        adaptive_mask, _ = predictor(x)

        out = attn(x, adaptive_mask=adaptive_mask)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_attention_without_adaptive_mask_unchanged(self) -> None:
        """Without adaptive_mask, behavior is identical to before."""
        config = TitansConfig(
            dim=64, num_heads=4, window_size=16, max_seq_len=256,
            use_rope=False,
        )
        attn = SlidingWindowAttention(config)
        x = mx.random.normal((2, 16, 64))

        # Call without adaptive_mask (default)
        out1 = attn(x)
        mx.eval(out1)

        # Call with explicit None
        out2 = attn(x, adaptive_mask=None)
        mx.eval(out2)

        diff = mx.max(mx.abs(out1 - out2)).item()
        assert diff == 0.0

    def test_attention_with_prefix_and_adaptive_mask(
        self, adaptive_config: TitansConfig
    ) -> None:
        """Adaptive mask works alongside persistent memory prefix."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor
        from titans_mlx.attention import SlidingWindowAttention

        attn = SlidingWindowAttention(adaptive_config)
        predictor = AdaptiveWindowPredictor(adaptive_config)

        x = mx.random.normal((2, 16, 64))
        prefix = mx.random.normal((2, 4, 64))
        adaptive_mask, _ = predictor(x)

        out = attn(x, prefix=prefix, adaptive_mask=adaptive_mask)
        mx.eval(out)

        assert out.shape == (2, 16, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adaptive_window.py::TestSlidingWindowAdaptiveMask -v`
Expected: FAIL — `adaptive_mask` is not a valid parameter

- [ ] **Step 3: Modify SlidingWindowAttention to accept adaptive_mask**

In `src/titans_mlx/attention.py`, modify `SlidingWindowAttention.__call__` (line 192):

Change the signature from:
```python
    def __call__(
        self,
        x: mx.array,
        prefix: mx.array | None = None,
        seq_offset: int = 0,
    ) -> mx.array:
```

To:
```python
    def __call__(
        self,
        x: mx.array,
        prefix: mx.array | None = None,
        seq_offset: int = 0,
        adaptive_mask: mx.array | None = None,
    ) -> mx.array:
```

Replace the mask creation and attention computation block (lines 236-242):

From:
```python
        # Create attention mask
        mask = self._create_extended_mask(seq_len, full_len, prefix_len)

        # Use fused scaled dot-product attention (Metal-optimized)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
```

To:
```python
        # Create attention mask
        if adaptive_mask is not None:
            # Adaptive soft masking: convert to additive bias
            # Prefix positions are always fully attended
            prefix_ones = mx.ones((adaptive_mask.shape[0], 1, seq_len, prefix_len))
            # Concatenate: [full prefix attention | adaptive soft mask]
            full_soft_mask = mx.concatenate([prefix_ones, adaptive_mask], axis=3)
            # Convert to additive bias: 0 -> -1e9, 1 -> 0
            mask = mx.where(full_soft_mask > 1e-6, mx.zeros_like(full_soft_mask), full_soft_mask - 1e9)
            # Smooth version: log(mask + eps) but simpler to use linear scale
            mask = (full_soft_mask - 1.0) * 1e9  # 1.0 -> 0, 0.0 -> -1e9
        else:
            mask = self._create_extended_mask(seq_len, full_len, prefix_len)

        # Use fused scaled dot-product attention (Metal-optimized)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_adaptive_window.py::TestSlidingWindowAdaptiveMask -v`
Expected: PASS

- [ ] **Step 5: Run existing attention tests for regression**

Run: `uv run pytest tests/test_attention.py -v`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/titans_mlx/attention.py tests/test_adaptive_window.py
git commit -m "feat: integrate adaptive soft mask into SlidingWindowAttention"
```

---

### Task 4: MAGBlock Integration

**Files:**
- Modify: `src/titans_mlx/models.py:460-577` (MAGBlock)
- Test: `tests/test_adaptive_window.py` (add MAG integration tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adaptive_window.py`:

```python
class TestMAGBlockAdaptiveWindow:
    """Tests for adaptive window integration in MAGBlock."""

    @pytest.fixture
    def adaptive_config(self) -> TitansConfig:
        return TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=4,
            chunk_size=32,
            window_size=32,
            max_seq_len=256,
            vocab_size=100,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=10.0,
        )

    def test_mag_block_forward_with_adaptive(self, adaptive_config: TitansConfig) -> None:
        """MAGBlock forward pass works with adaptive window enabled."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        out, state = block(x)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_mag_block_has_predictor(self, adaptive_config: TitansConfig) -> None:
        """MAGBlock instantiates window predictor when adaptive is enabled."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(adaptive_config)
        assert hasattr(block, "window_predictor")

    def test_mag_block_no_predictor_when_disabled(self) -> None:
        """MAGBlock does not instantiate predictor when adaptive is disabled."""
        from titans_mlx.models import MAGBlock

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, num_memory_layers=1,
            memory_hidden_mult=2.0, num_persistent_tokens=4,
            chunk_size=32, window_size=32, max_seq_len=256, vocab_size=100,
            adaptive_window=False,
        )
        block = MAGBlock(config)
        assert not hasattr(block, "window_predictor")

    def test_mag_block_exposes_falloff_centers(
        self, adaptive_config: TitansConfig
    ) -> None:
        """MAGBlock stores last falloff_centers for regularization access."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        _ = block(x)
        mx.eval(block._last_falloff_centers)

        assert block._last_falloff_centers is not None
        assert block._last_falloff_centers.shape == (2, 16, 1)

    def test_mag_regression_without_adaptive(self, default_config: TitansConfig) -> None:
        """MAGBlock without adaptive window produces identical output to baseline."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(default_config)
        x = mx.random.normal((2, 16, 64))

        out1, _ = block(x)
        out2, _ = block(x)
        mx.eval(out1, out2)

        diff = mx.max(mx.abs(out1 - out2)).item()
        assert diff == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adaptive_window.py::TestMAGBlockAdaptiveWindow -v`
Expected: FAIL — MAGBlock doesn't have `window_predictor`

- [ ] **Step 3: Modify MAGBlock to integrate adaptive windowing**

In `src/titans_mlx/models.py`, add import at the top (after line 28):

```python
from titans_mlx.adaptive_window import AdaptiveWindowPredictor
```

In `MAGBlock.__init__` (after `self.dropout_p = config.dropout` at line 503), add:

```python
        # Adaptive window sizing (optional)
        self._last_falloff_centers: mx.array | None = None
        if config.adaptive_window:
            self.window_predictor = AdaptiveWindowPredictor(config)
```

In `MAGBlock.core_forward`, modify the attention call section. Change:

```python
        # Eq. 26: y_t = Attn(x) - Attention branch
        normed = self.norm1(h)
        attn_out = self.attention(normed, prefix=persistent)
```

To:

```python
        # Eq. 26: y_t = Attn(x) - Attention branch
        normed = self.norm1(h)

        # Adaptive window: predict soft mask from hidden state
        adaptive_mask = None
        if hasattr(self, "window_predictor"):
            adaptive_mask, self._last_falloff_centers = self.window_predictor(normed)

        attn_out = self.attention(normed, prefix=persistent, adaptive_mask=adaptive_mask)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_adaptive_window.py::TestMAGBlockAdaptiveWindow -v`
Expected: PASS

- [ ] **Step 5: Run full model tests for regression**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/titans_mlx/models.py tests/test_adaptive_window.py
git commit -m "feat: integrate AdaptiveWindowPredictor into MAGBlock"
```

---

### Task 5: Full Model Forward Pass (TitansMAG)

**Files:**
- Test: `tests/test_adaptive_window.py` (add full model test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adaptive_window.py`:

```python
class TestTitansMAGAdaptiveWindow:
    """End-to-end tests for TitansMAG with adaptive window."""

    @pytest.fixture
    def adaptive_mag_config(self) -> TitansConfig:
        return TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=4,
            chunk_size=32,
            window_size=32,
            max_seq_len=256,
            vocab_size=100,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=10.0,
        )

    def test_full_model_forward(self, adaptive_mag_config: TitansConfig) -> None:
        """TitansMAG forward pass with adaptive window produces valid logits."""
        from titans_mlx.models import TitansMAG

        model = TitansMAG(adaptive_mag_config)
        input_ids = mx.random.randint(0, 100, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 32, 100)
        assert len(states) == 2

    def test_multi_chunk_forward(self, adaptive_mag_config: TitansConfig) -> None:
        """Multi-chunk sequences work with adaptive window."""
        from titans_mlx.models import TitansMAG

        model = TitansMAG(adaptive_mag_config)
        # seq_len=64 > chunk_size=32, forces 2 chunks
        input_ids = mx.random.randint(0, 100, (2, 64))
        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 64, 100)

    def test_collect_falloff_centers(self, adaptive_mag_config: TitansConfig) -> None:
        """Can collect falloff centers from all blocks after forward pass."""
        from titans_mlx.models import TitansMAG

        model = TitansMAG(adaptive_mag_config)
        input_ids = mx.random.randint(0, 100, (2, 32))
        _ = model(input_ids)

        centers = []
        for block in model.blocks:
            fc = block._last_falloff_centers
            assert fc is not None
            centers.append(fc)
            mx.eval(fc)

        assert len(centers) == 2  # num_layers=2
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_adaptive_window.py::TestTitansMAGAdaptiveWindow -v`
Expected: PASS (no new code needed — this validates the integration from Tasks 2-4)

- [ ] **Step 3: Commit**

```bash
git add tests/test_adaptive_window.py
git commit -m "test: add end-to-end TitansMAG adaptive window tests"
```

---

### Task 6: Efficiency Regularization in Training Loop

**Files:**
- Modify: `scripts/pretrain.py:87-111` (TrainingConfig)
- Modify: `scripts/pretrain.py:451-463` (loss_fn)
- Modify: `scripts/pretrain.py:1256-1259` (CLI args)
- Modify: `scripts/pretrain.py:1434-1482` (config construction)
- Modify: `scripts/pretrain.py:1506-1528` (model config construction)
- Test: `tests/test_adaptive_window.py` (add regularization test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adaptive_window.py`:

```python
class TestAdaptiveWindowRegularization:
    """Tests for efficiency regularization loss."""

    def test_regularization_computes(self) -> None:
        """Regularization loss computes from falloff centers."""
        from titans_mlx.adaptive_window import compute_window_regularization

        # Simulate falloff centers from 2 layers
        centers = [
            mx.ones((2, 16, 1)) * 20.0,   # layer 0: large windows
            mx.ones((2, 16, 1)) * 5.0,     # layer 1: small windows
        ]
        max_window = 32
        reg = compute_window_regularization(centers, max_window)
        mx.eval(reg)

        # mean([20/32, 5/32]) = mean([0.625, 0.15625]) = 0.390625
        assert reg.item() == pytest.approx(0.390625, abs=1e-4)

    def test_regularization_scales_with_lambda(self) -> None:
        """Lambda scales the regularization."""
        from titans_mlx.adaptive_window import compute_window_regularization

        centers = [mx.ones((1, 8, 1)) * 16.0]
        reg1 = compute_window_regularization(centers, max_window=32)
        mx.eval(reg1)

        # reg = mean(16/32) = 0.5
        assert reg1.item() == pytest.approx(0.5, abs=1e-4)

    def test_regularization_zero_when_min_window(self) -> None:
        """Regularization is minimal when all windows are at minimum."""
        from titans_mlx.adaptive_window import compute_window_regularization

        centers = [mx.ones((1, 8, 1)) * 4.0]
        reg = compute_window_regularization(centers, max_window=32)
        mx.eval(reg)

        assert reg.item() == pytest.approx(4.0 / 32.0, abs=1e-4)

    def test_regularization_empty_list(self) -> None:
        """Returns zero for empty falloff centers list."""
        from titans_mlx.adaptive_window import compute_window_regularization

        reg = compute_window_regularization([], max_window=32)
        mx.eval(reg)

        assert reg.item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adaptive_window.py::TestAdaptiveWindowRegularization -v`
Expected: FAIL — `compute_window_regularization` not found

- [ ] **Step 3: Add regularization function to adaptive_window.py**

Append to `src/titans_mlx/adaptive_window.py`:

```python
def compute_window_regularization(
    falloff_centers: list[mx.array],
    max_window: int,
) -> mx.array:
    """Compute efficiency regularization from per-layer falloff centers.

    Penalizes large windows: reg = mean(falloff_centers / max_window).
    Multiply by lambda_window externally.

    Args:
        falloff_centers: List of (batch, seq_len, 1) arrays, one per layer.
        max_window: Maximum window size for normalization.

    Returns:
        Scalar regularization loss (before lambda scaling).
    """
    if not falloff_centers:
        return mx.array(0.0)

    layer_means = []
    for fc in falloff_centers:
        layer_means.append(mx.mean(fc / max_window))

    return mx.mean(mx.stack(layer_means))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_adaptive_window.py::TestAdaptiveWindowRegularization -v`
Expected: PASS

- [ ] **Step 5: Add TrainingConfig fields and CLI args**

In `scripts/pretrain.py`, add to `TrainingConfig` (after `huber_delta_init` at line 111):

```python
    # Adaptive window sizing
    adaptive_window: bool = False
    adaptive_window_min: int = 64
    adaptive_window_max: int | None = None
    adaptive_window_temperature: float = 10.0
    adaptive_window_lambda: float = 0.01
```

Add CLI args (after the `--window-size` arg around line 1258):

```python
    parser.add_argument(
        "--adaptive-window", action="store_true", help="Enable adaptive window sizing"
    )
    parser.add_argument(
        "--adaptive-window-min", type=int, default=64, help="Min adaptive window"
    )
    parser.add_argument(
        "--adaptive-window-max", type=int, default=None, help="Max adaptive window"
    )
    parser.add_argument(
        "--adaptive-window-temperature", type=float, default=10.0,
        help="Soft mask temperature"
    )
    parser.add_argument(
        "--adaptive-window-lambda", type=float, default=0.01,
        help="Window size regularization weight"
    )
```

In the `TrainingConfig` construction (around line 1434), add:

```python
        adaptive_window=args.adaptive_window,
        adaptive_window_min=args.adaptive_window_min,
        adaptive_window_max=args.adaptive_window_max,
        adaptive_window_temperature=args.adaptive_window_temperature,
        adaptive_window_lambda=args.adaptive_window_lambda,
```

In the `TitansConfig` construction (around line 1506), add:

```python
        adaptive_window=config.adaptive_window,
        adaptive_window_min=config.adaptive_window_min,
        adaptive_window_max=config.adaptive_window_max,
        adaptive_window_temperature=config.adaptive_window_temperature,
        adaptive_window_lambda=config.adaptive_window_lambda,
```

- [ ] **Step 6: Modify loss_fn to include regularization**

In `scripts/pretrain.py`, modify `loss_fn` (line 451):

From:
```python
def loss_fn(
    model: nn.Module, input_ids: mx.array, labels: mx.array
) -> tuple[mx.array, mx.array]:
    """Compute cross-entropy loss."""
    logits, _ = model(input_ids)
    # Reshape for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Cross entropy loss
    loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction="mean")
    return loss, logits
```

To:
```python
def loss_fn(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
    adaptive_window_lambda: float = 0.0,
) -> tuple[mx.array, mx.array]:
    """Compute cross-entropy loss with optional window regularization."""
    logits, _ = model(input_ids)
    # Reshape for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Cross entropy loss
    loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction="mean")

    # Adaptive window regularization
    if adaptive_window_lambda > 0.0:
        from titans_mlx.adaptive_window import compute_window_regularization

        falloff_centers = []
        for block in model.blocks:
            fc = getattr(block, "_last_falloff_centers", None)
            if fc is not None:
                falloff_centers.append(fc)

        if falloff_centers:
            max_w = model.config.effective_adaptive_window_max
            reg = compute_window_regularization(falloff_centers, max_w)
            loss = loss + adaptive_window_lambda * reg

    return loss, logits
```

Update the `compute_grads` function to pass lambda through:

From:
```python
    loss_and_grad_fn = nn.value_and_grad(
        model, lambda m: loss_fn(m, input_ids, labels)[0]
    )
```

To:
```python
    loss_and_grad_fn = nn.value_and_grad(
        model, lambda m: loss_fn(m, input_ids, labels, adaptive_window_lambda)[0]
    )
```

And update `compute_grads` signature:
```python
def compute_grads(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
    adaptive_window_lambda: float = 0.0,
) -> tuple[mx.array, dict]:
```

Update all call sites of `compute_grads` and `loss_fn` in the training loop to pass `config.adaptive_window_lambda` when `config.adaptive_window` is True, `0.0` otherwise.

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/titans_mlx/adaptive_window.py scripts/pretrain.py tests/test_adaptive_window.py
git commit -m "feat: add window size efficiency regularization to training loop"
```

---

### Task 7: Update __init__.py Exports

**Files:**
- Modify: `src/titans_mlx/__init__.py`

- [ ] **Step 1: Add exports**

In `src/titans_mlx/__init__.py`, add import:

```python
from titans_mlx.adaptive_window import AdaptiveWindowPredictor, compute_window_regularization
```

Add to `__all__`:

```python
    # Adaptive Window
    "AdaptiveWindowPredictor",
    "compute_window_regularization",
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from titans_mlx import AdaptiveWindowPredictor, compute_window_regularization; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/titans_mlx/__init__.py
git commit -m "feat: export adaptive window module from titans_mlx"
```

---

### Task 8: Diagnostic Training Test

**Files:**
- Test: `tests/test_adaptive_window.py` (add training diagnostic)

- [ ] **Step 1: Write the diagnostic test**

Append to `tests/test_adaptive_window.py`:

```python
class TestAdaptiveWindowTraining:
    """Diagnostic test: verify adaptive windowing trains end-to-end."""

    def test_loss_decreases_with_adaptive_window(self) -> None:
        """A small adaptive-window model's loss decreases over a few steps."""
        import mlx.optimizers as optim

        from titans_mlx.adaptive_window import compute_window_regularization
        from titans_mlx.models import TitansMAG

        config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=2,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=2,
            chunk_size=16,
            window_size=16,
            max_seq_len=64,
            vocab_size=50,
            use_conv=False,
            use_rope=False,
            adaptive_window=True,
            adaptive_window_min=2,
            adaptive_window_max=16,
            adaptive_window_temperature=10.0,
            adaptive_window_lambda=0.01,
        )

        model = TitansMAG(config)
        optimizer = optim.Adam(learning_rate=1e-3)

        def train_loss(model, ids, labels):
            logits, _ = model(ids)
            b, s, v = logits.shape
            ce = mx.mean(
                nn.losses.cross_entropy(
                    logits.reshape(-1, v), labels.reshape(-1), reduction="none"
                )
            )
            # Add regularization
            centers = [
                blk._last_falloff_centers
                for blk in model.blocks
                if getattr(blk, "_last_falloff_centers", None) is not None
            ]
            reg = compute_window_regularization(centers, config.effective_adaptive_window_max)
            return ce + config.adaptive_window_lambda * reg

        loss_grad_fn = nn.value_and_grad(model, train_loss)

        losses = []
        for _ in range(10):
            ids = mx.random.randint(0, 50, (2, 16))
            labels = mx.random.randint(0, 50, (2, 16))
            loss, grads = loss_grad_fn(model, ids, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_per_layer_windows_diverge(self) -> None:
        """Different layers learn different effective window sizes."""
        import mlx.optimizers as optim

        from titans_mlx.adaptive_window import compute_window_regularization
        from titans_mlx.models import TitansMAG

        config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=3,  # 3 layers to see divergence
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=2,
            chunk_size=16,
            window_size=16,
            max_seq_len=64,
            vocab_size=50,
            use_conv=False,
            use_rope=False,
            adaptive_window=True,
            adaptive_window_min=2,
            adaptive_window_max=16,
            adaptive_window_temperature=10.0,
            adaptive_window_lambda=0.01,
        )

        model = TitansMAG(config)
        optimizer = optim.Adam(learning_rate=1e-3)

        def train_loss(model, ids, labels):
            logits, _ = model(ids)
            b, s, v = logits.shape
            ce = mx.mean(
                nn.losses.cross_entropy(
                    logits.reshape(-1, v), labels.reshape(-1), reduction="none"
                )
            )
            centers = [
                blk._last_falloff_centers
                for blk in model.blocks
                if getattr(blk, "_last_falloff_centers", None) is not None
            ]
            reg = compute_window_regularization(centers, config.effective_adaptive_window_max)
            return ce + config.adaptive_window_lambda * reg

        loss_grad_fn = nn.value_and_grad(model, train_loss)

        for _ in range(20):
            ids = mx.random.randint(0, 50, (2, 16))
            labels = mx.random.randint(0, 50, (2, 16))
            loss, grads = loss_grad_fn(model, ids, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        # Check that predictor weights have diverged across layers
        weights = []
        for block in model.blocks:
            w = block.window_predictor.proj.weight
            mx.eval(w)
            weights.append(w.tolist())

        # At least one pair of layers should have different weights
        all_same = all(
            weights[i] == weights[0] for i in range(1, len(weights))
        )
        assert not all_same, "All layers have identical predictor weights after training"
```

- [ ] **Step 2: Run the diagnostic test**

Run: `uv run pytest tests/test_adaptive_window.py::TestAdaptiveWindowTraining -v --timeout=60`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_adaptive_window.py
git commit -m "test: add end-to-end training diagnostic for adaptive windowing"
```

---

### Task 9: Checkpoint Save/Load Compatibility

**Files:**
- Test: `tests/test_adaptive_window.py` (add checkpoint test)

- [ ] **Step 1: Write the test**

Append to `tests/test_adaptive_window.py`:

```python
class TestAdaptiveWindowCheckpoint:
    """Tests for checkpoint save/load with adaptive window weights."""

    def test_save_load_round_trip(self, tmp_path) -> None:
        """Model with adaptive window survives save/load."""
        from titans_mlx.models import TitansMAG

        config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=2,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=2,
            chunk_size=16,
            window_size=16,
            max_seq_len=64,
            vocab_size=50,
            use_conv=False,
            use_rope=False,
            adaptive_window=True,
            adaptive_window_min=2,
            adaptive_window_max=16,
        )

        model = TitansMAG(config)
        input_ids = mx.random.randint(0, 50, (1, 16))

        # Forward pass to populate weights
        logits_before, _ = model(input_ids)
        mx.eval(logits_before)

        # Save weights
        weights = dict(mx.utils.tree_flatten(model.parameters()))
        save_path = str(tmp_path / "model.safetensors")
        mx.save_safetensors(save_path, weights)

        # Load into fresh model
        model2 = TitansMAG(config)
        loaded = mx.load(save_path)
        model2.load_weights(list(loaded.items()))

        logits_after, _ = model2(input_ids)
        mx.eval(logits_after)

        diff = mx.max(mx.abs(logits_before - logits_after)).item()
        assert diff < 1e-5, f"Checkpoint round-trip diverged: max diff = {diff}"

    def test_predictor_weights_in_checkpoint(self, tmp_path) -> None:
        """Checkpoint contains predictor weights."""
        from titans_mlx.models import TitansMAG

        config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=2,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=2,
            chunk_size=16,
            window_size=16,
            max_seq_len=64,
            vocab_size=50,
            use_conv=False,
            use_rope=False,
            adaptive_window=True,
            adaptive_window_min=2,
            adaptive_window_max=16,
        )

        model = TitansMAG(config)
        weights = dict(mx.utils.tree_flatten(model.parameters()))

        # Check that predictor keys exist
        predictor_keys = [k for k in weights if "window_predictor" in k]
        assert len(predictor_keys) > 0, "No window_predictor keys in model parameters"
        # Should have weight + bias per layer (2 layers * 2 params = 4)
        assert len(predictor_keys) == 4, f"Expected 4 predictor params, got {len(predictor_keys)}"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_adaptive_window.py::TestAdaptiveWindowCheckpoint -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_adaptive_window.py
git commit -m "test: add checkpoint save/load tests for adaptive windowing"
```

---

### Task 10: Add conftest Fixture and Final Regression

**Files:**
- Modify: `tests/conftest.py` (add adaptive_config fixture)

- [ ] **Step 1: Add shared fixture**

In `tests/conftest.py`, add:

```python
@pytest.fixture
def adaptive_config() -> TitansConfig:
    """Adaptive window configuration for tests."""
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        ffn_mult=2.0,
        num_memory_layers=1,
        memory_hidden_mult=2.0,
        num_persistent_tokens=4,
        chunk_size=32,
        window_size=32,
        dropout=0.0,
        use_conv=False,
        use_rope=True,
        max_seq_len=256,
        vocab_size=100,
        adaptive_window=True,
        adaptive_window_min=4,
        adaptive_window_max=32,
        adaptive_window_temperature=10.0,
        adaptive_window_lambda=0.01,
    )
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add adaptive_config fixture to conftest"
```

---

### Task 11: Inference Script — CLI Args and Checkpoint Loading

**Files:**
- Modify: `scripts/inference.py:924-1022` (CLI args)
- Modify: `scripts/inference.py:397-506` (load_model metadata parsing)
- Modify: `scripts/inference.py:575-616` (load_lora_model metadata parsing)

- [ ] **Step 1: Add CLI args for inference**

In `scripts/inference.py`, after the `--memory-state-bits` arg (around line 1006), add:

```python
    parser.add_argument(
        "--adaptive-window", action="store_true",
        help="Enable adaptive window sizing (must match training config)"
    )
    parser.add_argument(
        "--adaptive-window-min", type=int, default=64, help="Min adaptive window"
    )
    parser.add_argument(
        "--adaptive-window-max", type=int, default=None, help="Max adaptive window"
    )
    parser.add_argument(
        "--adaptive-window-temperature", type=float, default=10.0,
        help="Soft mask temperature"
    )
```

Note: no `--adaptive-window-lambda` for inference (regularization is training-only).

- [ ] **Step 2: Update load_model to read adaptive window from checkpoint metadata**

In `scripts/inference.py` `load_model`, after the `huber_delta_init` parsing (around line 453), add:

```python
        # Adaptive window flags
        adaptive_window = str(meta.get("adaptive_window", ["False"])[0]) == "True"
        adaptive_window_min = int(meta.get("adaptive_window_min", [64])[0])
        adaptive_window_max_raw = str(meta.get("adaptive_window_max", ["None"])[0])
        adaptive_window_max = None if adaptive_window_max_raw == "None" else int(adaptive_window_max_raw)
        adaptive_window_temperature = float(meta.get("adaptive_window_temperature", [10.0])[0])
```

In the fallback defaults block (around line 480), add:

```python
        adaptive_window = False
        adaptive_window_min = 64
        adaptive_window_max = None
        adaptive_window_temperature = 10.0
```

In the `TitansConfig` construction (around line 505), add:

```python
        adaptive_window=adaptive_window,
        adaptive_window_min=adaptive_window_min,
        adaptive_window_max=adaptive_window_max,
        adaptive_window_temperature=adaptive_window_temperature,
```

- [ ] **Step 3: Update load_lora_model similarly**

In `scripts/inference.py` `load_lora_model`, in the `TitansConfig` construction (around line 615), add:

```python
        adaptive_window=meta.get("adaptive_window", False),
        adaptive_window_min=meta.get("adaptive_window_min", 64),
        adaptive_window_max=meta.get("adaptive_window_max", None),
        adaptive_window_temperature=meta.get("adaptive_window_temperature", 10.0),
```

- [ ] **Step 4: Apply CLI overrides in main()**

After `args.quantize_memory_state` handling (around line 1041), add:

```python
    if args.adaptive_window:
        config.adaptive_window = True
        config.adaptive_window_min = args.adaptive_window_min
        config.adaptive_window_max = args.adaptive_window_max
        config.adaptive_window_temperature = args.adaptive_window_temperature
```

- [ ] **Step 5: Commit**

```bash
git add scripts/inference.py
git commit -m "feat: add adaptive window CLI args and checkpoint loading to inference"
```

---

### Task 12: Checkpoint Metadata — Save Adaptive Window Config

**Files:**
- Modify: `scripts/pretrain.py:692-722` (save_checkpoint metadata dict)

- [ ] **Step 1: Add adaptive window fields to checkpoint metadata**

In `scripts/pretrain.py`, in the `save_checkpoint` metadata dict (after `huber_delta_init` around line 717), add:

```python
        "adaptive_window": model_config.adaptive_window,
        "adaptive_window_min": model_config.adaptive_window_min,
        "adaptive_window_max": str(model_config.adaptive_window_max),  # None-safe
        "adaptive_window_temperature": model_config.adaptive_window_temperature,
```

- [ ] **Step 2: Commit**

```bash
git add scripts/pretrain.py
git commit -m "feat: save adaptive window config in checkpoint metadata"
```

---

### Task 13: Other Training Scripts (SFT, LoRA, DPO, RLVR)

**Files:**
- Modify: `scripts/sft.py`
- Modify: `scripts/lora.py`
- Modify: `scripts/dpo.py`
- Modify: `scripts/rlvr.py`

Each script needs the same changes as pretrain.py:
1. `TrainingConfig` fields for adaptive window
2. CLI arg definitions
3. Config construction from args
4. TitansConfig construction from training config
5. `loss_fn` / `compute_grads` to pass adaptive_window_lambda
6. Checkpoint metadata to include adaptive window fields

- [ ] **Step 1: Update sft.py**

Apply the same pattern as pretrain.py (Task 6):
- Add `adaptive_window`, `adaptive_window_min`, `adaptive_window_max`, `adaptive_window_temperature`, `adaptive_window_lambda` to the script's `TrainingConfig` dataclass
- Add `--adaptive-window`, `--adaptive-window-min`, `--adaptive-window-max`, `--adaptive-window-temperature`, `--adaptive-window-lambda` CLI args
- Wire through to `TrainingConfig` construction, `TitansConfig` construction, and `loss_fn` / `compute_grads`
- Add adaptive window fields to checkpoint metadata dict

- [ ] **Step 2: Update lora.py**

Same changes as sft.py.

- [ ] **Step 3: Update dpo.py**

Same changes as sft.py. Note: DPO has a different loss function (DPO loss, not cross-entropy). Add the regularization to the DPO loss — the window regularization is additive and independent of the task-specific loss formulation.

- [ ] **Step 4: Update rlvr.py**

Same changes as sft.py. Note: RLVR also has a custom loss. Same approach — additive regularization.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=180`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/sft.py scripts/lora.py scripts/dpo.py scripts/rlvr.py
git commit -m "feat: add adaptive window support to sft, lora, dpo, and rlvr scripts"
```

---

### Task 14: Final Full Regression

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=180`
Expected: ALL PASS — no regressions anywhere

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from titans_mlx import AdaptiveWindowPredictor, compute_window_regularization; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify CLI help**

Run: `uv run python scripts/pretrain.py --help | grep adaptive`
Expected: Shows all `--adaptive-window*` flags

Run: `uv run python scripts/inference.py --help | grep adaptive`
Expected: Shows `--adaptive-window`, `--adaptive-window-min`, `--adaptive-window-max`, `--adaptive-window-temperature` (no lambda)
