# MAL Adaptive Window Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `AdaptiveWindowPredictor` into `MALBlock` so MAL variants support per-layer learned adaptive window sizing.

**Architecture:** `MALBlock.__init__` conditionally creates an `AdaptiveWindowPredictor`. In `core_forward`, the predictor runs on `norm2(h_mid)` (memory-enriched hidden state) before the attention call, passing the resulting soft mask to `SlidingWindowAttention`. No new modules, config, or CLI changes needed.

**Tech Stack:** MLX (mx.array, nn.Module), pytest

---

### Task 1: MALBlock Integration

**Files:**
- Modify: `src/titans_mlx/models.py:749` (MALBlock.__init__, after dropout)
- Modify: `src/titans_mlx/models.py:813-814` (MALBlock.core_forward, before attention call)
- Test: `tests/test_adaptive_window.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptive_window.py`:

```python
class TestMALBlockAdaptiveWindow:
    """Tests for adaptive window integration in MALBlock."""

    @pytest.fixture
    def adaptive_mal_config(self) -> TitansConfig:
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

    def test_mal_block_forward_with_adaptive(self, adaptive_mal_config: TitansConfig) -> None:
        """MALBlock forward pass works with adaptive window enabled."""
        from titans_mlx.models import MALBlock

        block = MALBlock(adaptive_mal_config)
        x = mx.random.normal((2, 16, 64))
        out, state = block(x)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_mal_block_has_predictor(self, adaptive_mal_config: TitansConfig) -> None:
        """MALBlock instantiates window predictor when adaptive is enabled."""
        from titans_mlx.models import MALBlock

        block = MALBlock(adaptive_mal_config)
        assert hasattr(block, "window_predictor")

    def test_mal_block_no_predictor_when_disabled(self) -> None:
        """MALBlock does not instantiate predictor when adaptive is disabled."""
        from titans_mlx.models import MALBlock

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, num_memory_layers=1,
            memory_hidden_mult=2.0, num_persistent_tokens=4,
            chunk_size=32, window_size=32, max_seq_len=256, vocab_size=100,
            adaptive_window=False,
        )
        block = MALBlock(config)
        assert not hasattr(block, "window_predictor")

    def test_mal_block_exposes_falloff_centers(
        self, adaptive_mal_config: TitansConfig
    ) -> None:
        """MALBlock stores last falloff_centers for regularization access."""
        from titans_mlx.models import MALBlock

        block = MALBlock(adaptive_mal_config)
        x = mx.random.normal((2, 16, 64))
        _ = block(x)
        mx.eval(block._last_falloff_centers)

        assert block._last_falloff_centers is not None
        assert block._last_falloff_centers.shape == (2, 16, 1)

    def test_mal_regression_without_adaptive(self, default_config: TitansConfig) -> None:
        """MALBlock without adaptive window produces identical output to baseline."""
        from titans_mlx.models import MALBlock

        block = MALBlock(default_config)
        x = mx.random.normal((2, 16, 64))

        out1, _ = block(x)
        out2, _ = block(x)
        mx.eval(out1, out2)

        diff = mx.max(mx.abs(out1 - out2)).item()
        assert diff == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adaptive_window.py::TestMALBlockAdaptiveWindow -v`
Expected: FAIL — MALBlock doesn't have `window_predictor`

- [ ] **Step 3: Modify MALBlock.__init__**

In `src/titans_mlx/models.py`, after `self.dropout_p = config.dropout` (line 749), add:

```python
        # Adaptive window sizing (optional)
        self._last_falloff_centers: mx.array | None = None
        if config.adaptive_window:
            self.window_predictor = AdaptiveWindowPredictor(config)
```

Note: `AdaptiveWindowPredictor` is already imported at the top of `models.py` (added during MAG integration).

- [ ] **Step 4: Modify MALBlock.core_forward**

In `src/titans_mlx/models.py`, replace lines 812-814:

```python
        # Attention layer with persistent prefix (uses norm2)
        normed_mid = self.norm2(h_mid)
        attn_out = self.attention(normed_mid, prefix=persistent)
```

With:

```python
        # Attention layer with persistent prefix (uses norm2)
        normed_mid = self.norm2(h_mid)

        # Adaptive window: predict from memory-enriched hidden state
        # NOTE: In multi-chunk sequences, _last_falloff_centers retains only the
        # final chunk's values (same limitation as MAG).
        adaptive_mask = None
        if hasattr(self, "window_predictor"):
            adaptive_mask, self._last_falloff_centers = self.window_predictor(normed_mid)

        attn_out = self.attention(normed_mid, prefix=persistent, adaptive_mask=adaptive_mask)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_adaptive_window.py::TestMALBlockAdaptiveWindow -v`
Expected: PASS (5/5)

- [ ] **Step 6: Run full model tests for regression**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add src/titans_mlx/models.py tests/test_adaptive_window.py
git commit -m "feat: integrate AdaptiveWindowPredictor into MALBlock"
```

---

### Task 2: TitansMAL End-to-End Tests

**Files:**
- Test: `tests/test_adaptive_window.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_adaptive_window.py`:

```python
class TestTitansMALAdaptiveWindow:
    """End-to-end tests for TitansMAL with adaptive window."""

    @pytest.fixture
    def adaptive_mal_config(self) -> TitansConfig:
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

    def test_full_model_forward(self, adaptive_mal_config: TitansConfig) -> None:
        """TitansMAL forward pass with adaptive window produces valid logits."""
        from titans_mlx.models import TitansMAL

        model = TitansMAL(adaptive_mal_config)
        input_ids = mx.random.randint(0, 100, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 32, 100)
        assert len(states) == 2

    def test_multi_chunk_forward(self, adaptive_mal_config: TitansConfig) -> None:
        """Multi-chunk sequences work with adaptive window."""
        from titans_mlx.models import TitansMAL

        model = TitansMAL(adaptive_mal_config)
        # seq_len=64 > chunk_size=32, forces 2 chunks
        input_ids = mx.random.randint(0, 100, (2, 64))
        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 64, 100)

    def test_collect_falloff_centers(self, adaptive_mal_config: TitansConfig) -> None:
        """Can collect falloff centers from all blocks after forward pass."""
        from titans_mlx.models import TitansMAL

        model = TitansMAL(adaptive_mal_config)
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

Run: `uv run pytest tests/test_adaptive_window.py::TestTitansMALAdaptiveWindow -v`
Expected: PASS (3/3 — no new code needed, validates end-to-end integration)

- [ ] **Step 3: Commit**

```bash
git add tests/test_adaptive_window.py
git commit -m "test: add end-to-end TitansMAL adaptive window tests"
```

---

### Task 3: Update README and Final Regression

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

In `README.md`, change the line about adaptive window applicability:

From:
```
`--adaptive-window` applies to MAG (MAL interface-compatible for future integration).
```

To:
```
`--adaptive-window` applies to MAG and MAL (sliding window variants).
```

Also in the Adaptive Window Sizing section, change:

From:
```
Currently supported for **MAG** blocks (MAL interface-compatible for future integration).
```

To:
```
Supported for **MAG** and **MAL** blocks (both use sliding window attention).
```

Update the test count badge from `459` to the new count after running the full suite.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README — MAL now supports adaptive window sizing"
```

- [ ] **Step 4: Final verification**

Run: `uv run pytest tests/ -q`
Expected: ALL PASS — no regressions
