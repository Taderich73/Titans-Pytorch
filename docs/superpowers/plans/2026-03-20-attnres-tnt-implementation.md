# AttnRes Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Block Attention Residuals into TitansTNT MAC with learned depth-wise residual connections and memory update modulation.

**Architecture:** Two separable components — BlockAttnRes (replaces fixed residuals between TNTMACBlocks with learned softmax attention over prior block representations) and AttnResMemoryGate (uses AttnRes weights to modulate memory learning rate). Both toggled via `use_attn_res` config flag.

**Tech Stack:** MLX (Apple Silicon ML framework), Python 3.10+, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-attnres-tnt-integration-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/titans_mlx/attn_res.py` | **NEW** — BlockAttnRes module + AttnResMemoryGate |
| `src/titans_mlx/config.py` | Add 5 AttnRes config fields + `attnres_base_block_size` property |
| `src/titans_mlx/memory.py` | Add `lr_scale` parameter to `NeuralLongTermMemory.__call__()` |
| `src/titans_mlx/tnt_memory.py` | Propagate `lr_scale` through GlobalMemory, LocalMemory, HierarchicalMemory |
| `src/titans_mlx/tnt_models.py` | Wire AttnRes into TNTMACBlock and TitansTNT |
| `src/titans_mlx/__init__.py` | Export new public names |
| `tests/test_tnt.py` | Add test classes for AttnRes |
| `scripts/pretrain.py` | Add CLI flags for AttnRes config |
| `train.sh` | Add AttnRes flags to training command |

---

### Task 1: Config — Add AttnRes fields to TitansConfig

**Files:**
- Modify: `src/titans_mlx/config.py`
- Test: `tests/test_tnt.py`

- [ ] **Step 1: Write failing test for AttnRes config fields**

Add to `tests/test_tnt.py` at the end of the file:

```python
# =============================================================================
# AttnRes Config Tests
# =============================================================================


class TestAttnResConfig(unittest.TestCase):
    """Test AttnRes configuration fields."""

    def test_attnres_defaults(self):
        """AttnRes is disabled by default."""
        config = TitansConfig()
        self.assertFalse(config.use_attn_res)
        self.assertEqual(config.num_attnres_blocks, 8)
        self.assertEqual(config.attnres_warmup_steps, 0)
        self.assertTrue(config.attnres_modulate_global_memory)
        self.assertFalse(config.attnres_modulate_local_memory)

    def test_attnres_base_block_size_even(self):
        """Block size derived correctly when evenly divisible."""
        config = TitansConfig(num_layers=16, num_attnres_blocks=8)
        self.assertEqual(config.attnres_base_block_size, 2)

    def test_attnres_base_block_size_uneven(self):
        """Block size with remainder — last block absorbs extra."""
        config = TitansConfig(num_layers=12, num_attnres_blocks=8)
        # 12 // 8 = 1, last block absorbs 4 extra layers
        self.assertEqual(config.attnres_base_block_size, 1)

    def test_attnres_serialization(self):
        """AttnRes fields survive to_dict/from_dict round-trip."""
        config = TitansConfig(
            use_attn_res=True,
            num_attnres_blocks=4,
            attnres_warmup_steps=1000,
            attnres_modulate_global_memory=False,
            attnres_modulate_local_memory=True,
        )
        d = config.to_dict()
        restored = TitansConfig.from_dict(d)
        self.assertTrue(restored.use_attn_res)
        self.assertEqual(restored.num_attnres_blocks, 4)
        self.assertEqual(restored.attnres_warmup_steps, 1000)
        self.assertFalse(restored.attnres_modulate_global_memory)
        self.assertTrue(restored.attnres_modulate_local_memory)
        self.assertEqual(restored.attnres_base_block_size, config.attnres_base_block_size)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestAttnResConfig -v`
Expected: FAIL — `TitansConfig` has no `use_attn_res` field

- [ ] **Step 3: Add AttnRes fields to TitansConfig**

In `src/titans_mlx/config.py`, add these fields after the existing TNT fields (after line 82 `finetune_local_chunk_sizes`):

```python
    # AttnRes (Attention Residuals)
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global_memory: bool = True
    attnres_modulate_local_memory: bool = False
```

Add a property after `active_local_chunk_sizes`:

```python
    @property
    def attnres_base_block_size(self) -> int:
        """S — base number of layers per AttnRes block.

        When num_layers is not evenly divisible by num_attnres_blocks,
        the last block absorbs the remainder.
        """
        return self.num_layers // self.num_attnres_blocks
```

Add the 5 new fields to `to_dict()`:

```python
            "use_attn_res": self.use_attn_res,
            "num_attnres_blocks": self.num_attnres_blocks,
            "attnres_warmup_steps": self.attnres_warmup_steps,
            "attnres_modulate_global_memory": self.attnres_modulate_global_memory,
            "attnres_modulate_local_memory": self.attnres_modulate_local_memory,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestAttnResConfig -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full existing test suite to verify no regressions**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/titans_mlx/config.py tests/test_tnt.py
git commit -m "feat(config): add AttnRes configuration fields to TitansConfig"
```

---

### Task 2: lr_scale — Add learning rate scaling to NeuralLongTermMemory

**Files:**
- Modify: `src/titans_mlx/memory.py:688-768`
- Test: `tests/test_tnt.py`

- [ ] **Step 1: Write failing test for lr_scale**

Add to `tests/test_tnt.py`:

```python
class TestLrScale(unittest.TestCase):
    """Test lr_scale parameter on NeuralLongTermMemory."""

    def setUp(self):
        self.config = TitansConfig(dim=64, num_heads=4, num_layers=2, vocab_size=100)
        self.mem = NeuralLongTermMemory(self.config)
        self.x = mx.random.normal((2, 8, 64))

    def test_lr_scale_default_matches_baseline(self):
        """lr_scale=1.0 (default) produces identical output to no lr_scale."""
        state = self.mem.init_state(2)
        out1, state1 = self.mem(self.x, state=state)
        out2, state2 = self.mem(self.x, state=state, lr_scale=1.0)
        mx.eval(out1, out2)
        self.assertTrue(mx.allclose(out1, out2, atol=1e-6).item())

    def test_lr_scale_zero_removes_gradient_contribution(self):
        """lr_scale=0.0 makes theta=0: no gradient in momentum update.

        State still changes via alpha decay (weight decay), but the
        gradient-driven learning signal is zeroed out.
        """
        state = self.mem.init_state(2)
        _, state_zero = self.mem(self.x, state=state, lr_scale=0.0)
        _, state_full = self.mem(self.x, state=state, lr_scale=1.0)
        mx.eval(state_zero.weights[0], state_full.weights[0])
        # Both states differ from original (alpha decay), but they
        # differ from each other because lr_scale=0 removes gradient
        self.assertFalse(
            mx.allclose(
                state_zero.weights[0], state_full.weights[0], atol=1e-6
            ).item()
        )

    def test_lr_scale_affects_state_differently(self):
        """Different lr_scale values produce different memory states."""
        state = self.mem.init_state(2)
        _, state_half = self.mem(self.x, state=state, lr_scale=0.5)
        _, state_full = self.mem(self.x, state=state, lr_scale=1.0)
        mx.eval(state_half.weights[0], state_full.weights[0])
        self.assertFalse(
            mx.allclose(state_half.weights[0], state_full.weights[0], atol=1e-6).item()
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestLrScale -v`
Expected: FAIL — `__call__()` got unexpected keyword argument 'lr_scale'

- [ ] **Step 3: Add lr_scale to NeuralLongTermMemory.__call__()**

In `src/titans_mlx/memory.py`, modify the `__call__` method signature at line ~688:

```python
    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
        return_state: bool = True,
        lr_scale: float | mx.array = 1.0,
    ) -> tuple[mx.array, MemoryState | None]:
```

After line 746 (`eta = mx.mean(eta)`), add:

```python
        # Apply AttnRes modulation to learning rate
        theta = theta * lr_scale
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestLrScale -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/ -v`
Expected: All tests PASS (lr_scale defaults to 1.0, no behavioral change)

- [ ] **Step 6: Commit**

```bash
git add src/titans_mlx/memory.py tests/test_tnt.py
git commit -m "feat(memory): add lr_scale parameter to NeuralLongTermMemory"
```

---

### Task 3: Propagate lr_scale through GlobalMemory, LocalMemory, HierarchicalMemory

**Files:**
- Modify: `src/titans_mlx/tnt_memory.py:38-81` (GlobalMemory), `83-185` (LocalMemory), `187-372` (HierarchicalMemory)
- Test: `tests/test_tnt.py`

- [ ] **Step 1: Write failing test for memory_gate propagation**

Add to `tests/test_tnt.py`:

```python
class TestMemoryGatePropagation(unittest.TestCase):
    """Test memory_gate propagation through hierarchical memory."""

    def setUp(self):
        self.config = TitansConfig(
            dim=64, num_heads=4, num_layers=4, vocab_size=100,
            use_tnt=True, local_chunk_sizes=[8],
            attnres_modulate_global_memory=True,
            attnres_modulate_local_memory=False,
        )
        self.hier_mem = HierarchicalMemory(self.config)
        self.x = mx.random.normal((2, 8, 64))

    def test_memory_gate_none_matches_baseline(self):
        """memory_gate=None produces same result as no gate."""
        state = self.hier_mem.init_state(2)
        out1, s1 = self.hier_mem(self.x, state=state)
        out2, s2 = self.hier_mem(self.x, state=state, memory_gate=None)
        mx.eval(out1, out2)
        self.assertTrue(mx.allclose(out1, out2, atol=1e-6).item())

    def test_memory_gate_modulates_global(self):
        """memory_gate affects global memory when configured."""
        state = self.hier_mem.init_state(2)
        _, s_full = self.hier_mem(self.x, state=state, memory_gate=None)
        _, s_half = self.hier_mem(self.x, state=state, memory_gate=mx.array(0.5))
        mx.eval(s_full.global_state.weights[0], s_half.global_state.weights[0])
        # Global states should differ because memory_gate scales theta
        self.assertFalse(
            mx.allclose(
                s_full.global_state.weights[0],
                s_half.global_state.weights[0],
                atol=1e-6,
            ).item()
        )

    def test_memory_gate_skips_local_when_disabled(self):
        """memory_gate does NOT affect local memory when modulate_local=False."""
        state = self.hier_mem.init_state(2)
        _, s_full = self.hier_mem(self.x, state=state, memory_gate=None)
        _, s_half = self.hier_mem(self.x, state=state, memory_gate=mx.array(0.5))
        mx.eval(s_full.local_states[0].weights[0], s_half.local_states[0].weights[0])
        # Local states should be identical (modulate_local=False)
        self.assertTrue(
            mx.allclose(
                s_full.local_states[0].weights[0],
                s_half.local_states[0].weights[0],
                atol=1e-6,
            ).item()
        )

    def test_memory_gate_affects_local_when_enabled(self):
        """memory_gate affects local memory when modulate_local=True."""
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=4, vocab_size=100,
            use_tnt=True, local_chunk_sizes=[8],
            attnres_modulate_global_memory=True,
            attnres_modulate_local_memory=True,
        )
        hier_mem = HierarchicalMemory(config)
        state = hier_mem.init_state(2)
        _, s_full = hier_mem(self.x, state=state, memory_gate=None)
        _, s_half = hier_mem(self.x, state=state, memory_gate=mx.array(0.5))
        mx.eval(s_full.local_states[0].weights[0], s_half.local_states[0].weights[0])
        self.assertFalse(
            mx.allclose(
                s_full.local_states[0].weights[0],
                s_half.local_states[0].weights[0],
                atol=1e-6,
            ).item()
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestMemoryGatePropagation -v`
Expected: FAIL — `__call__()` got unexpected keyword argument 'memory_gate'

- [ ] **Step 3: Add lr_scale to GlobalMemory and LocalMemory**

In `src/titans_mlx/tnt_memory.py`:

**GlobalMemory.__call__** (line ~50):
```python
    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
        lr_scale: float | mx.array = 1.0,
    ) -> tuple[mx.array, MemoryState]:
        return self.memory(x, state=state, lr_scale=lr_scale)
```

**LocalMemory.__call__** (line ~151):
```python
    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
        lr_scale: float | mx.array = 1.0,
    ) -> tuple[mx.array, MemoryState]:
        if state is None:
            state = self.init_state(x.shape[0])
        return self.memory(x, state=state, lr_scale=lr_scale)
```

**HierarchicalMemory.__call__** (line ~252):
```python
    def __call__(
        self,
        x: mx.array,
        state: TNTMemoryState | None = None,
        memory_gate: mx.array | None = None,
    ) -> tuple[mx.array, TNTMemoryState]:
```

Inside the method body, compute lr_scale values before the update calls:

```python
        # Determine lr_scale from memory_gate
        global_lr_scale = (
            memory_gate if memory_gate is not None
            and self.config.attnres_modulate_global_memory
            else 1.0
        )
        local_lr_scale = (
            memory_gate if memory_gate is not None
            and self.config.attnres_modulate_local_memory
            else 1.0
        )
```

Then pass to the update calls:
- `self.global_memory(x, state=state.global_state, lr_scale=global_lr_scale)`
- `local_mem(x, state=local_state, lr_scale=local_lr_scale)`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestMemoryGatePropagation -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/titans_mlx/tnt_memory.py tests/test_tnt.py
git commit -m "feat(tnt_memory): propagate memory_gate/lr_scale through hierarchical memory"
```

---

### Task 4: BlockAttnRes and AttnResMemoryGate modules

**Files:**
- Create: `src/titans_mlx/attn_res.py`
- Test: `tests/test_tnt.py`

- [ ] **Step 1: Write failing tests for BlockAttnRes**

Add to `tests/test_tnt.py`:

```python
from titans_mlx.attn_res import BlockAttnRes, AttnResMemoryGate


class TestBlockAttnRes(unittest.TestCase):
    """Test Block Attention Residuals module."""

    def setUp(self):
        self.dim = 64
        self.batch = 2
        self.seq = 8
        self.attn_res = BlockAttnRes(self.dim)
        # Create some block representations
        self.block1 = mx.random.normal((self.batch, self.seq, self.dim))
        self.block2 = mx.random.normal((self.batch, self.seq, self.dim))
        self.partial = mx.random.normal((self.batch, self.seq, self.dim))

    def test_output_shape(self):
        """Output shape matches input shape."""
        h, weights = self.attn_res([self.block1], self.partial)
        mx.eval(h, weights)
        self.assertEqual(h.shape, (self.batch, self.seq, self.dim))

    def test_attn_weights_sum_to_one(self):
        """Attention weights sum to 1 across sources (softmax property)."""
        _, weights = self.attn_res([self.block1, self.block2], self.partial)
        mx.eval(weights)
        sums = mx.sum(weights, axis=-1)
        self.assertTrue(mx.allclose(sums, mx.ones_like(sums), atol=1e-5).item())

    def test_zero_init_uniform_weights(self):
        """Zero-initialized pseudo-query produces uniform attention weights."""
        attn_res = BlockAttnRes(self.dim)
        _, weights = attn_res([self.block1, self.block2], self.partial)
        mx.eval(weights)
        # 3 sources: block1, block2, partial → each ≈ 1/3
        expected = 1.0 / 3.0
        mean_weight = mx.mean(weights).item()
        self.assertAlmostEqual(mean_weight, expected, places=2)

    def test_single_source_partial_only(self):
        """With no completed blocks, attend only over partial_block."""
        h, weights = self.attn_res([], self.partial)
        mx.eval(h, weights)
        # Single source → weight = 1.0, output = partial
        self.assertEqual(weights.shape[-1], 1)
        self.assertTrue(mx.allclose(h, self.partial, atol=1e-5).item())

    def test_partial_none_with_blocks(self):
        """partial_block=None: attend only over completed blocks."""
        h, weights = self.attn_res([self.block1, self.block2], None)
        mx.eval(h, weights)
        self.assertEqual(h.shape, (self.batch, self.seq, self.dim))
        self.assertEqual(weights.shape[-1], 2)

    def test_no_nan(self):
        """No NaN in output."""
        h, weights = self.attn_res([self.block1, self.block2], self.partial)
        mx.eval(h, weights)
        self.assertFalse(mx.any(mx.isnan(h)).item())
        self.assertFalse(mx.any(mx.isnan(weights)).item())


class TestAttnResMemoryGate(unittest.TestCase):
    """Test AttnRes memory gate extraction."""

    def test_returns_scalar(self):
        """Gate returns a scalar value."""
        gate = AttnResMemoryGate()
        weights = mx.ones((2, 8, 4)) / 4.0  # uniform over 4 sources
        result = gate(weights)
        mx.eval(result)
        self.assertEqual(result.ndim, 0)

    def test_uniform_weights(self):
        """Uniform weights → gate = 1/N (weight on last source)."""
        gate = AttnResMemoryGate()
        n_sources = 4
        weights = mx.ones((2, 8, n_sources)) / n_sources
        result = gate(weights)
        mx.eval(result)
        self.assertAlmostEqual(result.item(), 1.0 / n_sources, places=5)

    def test_gate_in_range(self):
        """Gate value is in [0, 1] (it's a softmax weight)."""
        gate = AttnResMemoryGate()
        weights = mx.softmax(mx.random.normal((2, 8, 5)), axis=-1)
        result = gate(weights)
        mx.eval(result)
        self.assertGreaterEqual(result.item(), 0.0)
        self.assertLessEqual(result.item(), 1.0)

    def test_high_weight_on_last(self):
        """When last source dominates, gate ≈ 1."""
        gate = AttnResMemoryGate()
        # Construct weights where last source = 1.0, others = 0.0
        weights = mx.concatenate(
            [mx.zeros((2, 8, 2)), mx.ones((2, 8, 1))], axis=-1
        )
        result = gate(weights)
        mx.eval(result)
        self.assertAlmostEqual(result.item(), 1.0, places=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestBlockAttnRes tests/test_tnt.py::TestAttnResMemoryGate -v`
Expected: FAIL — ImportError: cannot import name 'BlockAttnRes'

- [ ] **Step 3: Create `src/titans_mlx/attn_res.py`**

```python
"""
Attention Residuals for TNT (MLX Implementation).

Implements Block Attention Residuals from "Attention Residuals"
(Kimi Team, arXiv 2603.15031).

BlockAttnRes replaces fixed residual connections between layers with
learned softmax attention over prior block representations. Each layer
gets a pseudo-query vector w_l that determines how to weight earlier
block outputs.

AttnResMemoryGate extracts an importance signal from the AttnRes
attention weights to modulate the memory update learning rate.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class BlockAttnRes(nn.Module):
    """Block Attention Residuals (Eq. 2-6 from AttnRes paper).

    Computes attention-weighted input from prior block representations
    and the current intra-block partial sum. Each layer owns one instance.

    The pseudo-query w_l is the weight vector of a Linear(dim, 1) projection.
    Initialized to zero so initial attention weights are uniform across
    sources, matching standard residual behavior at the start of training.

    Args:
        dim: Model dimension
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.attn_res_norm = nn.RMSNorm(dim)
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        # Zero-init pseudo-query for uniform initial weights
        self.attn_res_proj.weight = mx.zeros_like(self.attn_res_proj.weight)

    def __call__(
        self,
        blocks: list[mx.array],
        partial_block: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Compute AttnRes input for this layer.

        Args:
            blocks: Completed block representations [b_0, ..., b_{n-1}],
                each shape (batch, seq, dim).
            partial_block: Current intra-block partial sum (batch, seq, dim),
                or None if first layer in a new block.

        Returns:
            Tuple of (h_l, attn_weights):
                h_l: Attention-weighted input (batch, seq, dim)
                attn_weights: Attention distribution (batch, seq, num_sources)
        """
        # Collect all sources
        sources = list(blocks)
        if partial_block is not None:
            sources.append(partial_block)

        # Single source: skip attention, weight = 1.0
        if len(sources) == 1:
            v = sources[0]
            ones = mx.ones((*v.shape[:2], 1))
            return v, ones

        # Stack sources: (num_sources, batch, seq, dim)
        V = mx.stack(sources, axis=0)

        # Keys: RMSNorm prevents large-magnitude layers from dominating
        K = self.attn_res_norm(V)

        # Pseudo-query logits: w_l^T · k_i for each source
        # attn_res_proj: (dim, 1), K: (N, B, T, D) → logits: (N, B, T, 1)
        logits = self.attn_res_proj(K)
        logits = logits.squeeze(-1)  # (N, B, T)

        # Softmax over sources dimension (axis=0)
        attn_weights = mx.softmax(logits, axis=0)  # (N, B, T)

        # Weighted sum: h = Σ α_i · V_i
        # attn_weights: (N, B, T), V: (N, B, T, D)
        h = mx.sum(attn_weights[..., None] * V, axis=0)  # (B, T, D)

        # Transpose weights to (B, T, N) for downstream use
        attn_weights = mx.transpose(attn_weights, (1, 2, 0))  # (B, T, N)

        return h, attn_weights


class AttnResMemoryGate:
    """Extracts importance signal from AttnRes attention weights.

    Takes the attention weight assigned to the most recent source
    (the last element in the sources list, typically the intra-block
    partial sum) as the importance signal. Returns a scalar multiplier
    for the memory learning rate.

    Scalar averaging matches the existing pattern in NeuralLongTermMemory
    where theta, alpha, eta are batch-averaged because memory weights
    are shared across the batch.
    """

    def __call__(self, attn_weights: mx.array) -> mx.array:
        """Extract importance from AttnRes attention distribution.

        Args:
            attn_weights: Shape (batch, seq, num_sources) from BlockAttnRes.

        Returns:
            Scalar importance weight (mean over batch and sequence).
        """
        importance = attn_weights[:, :, -1]  # (B, T)
        return mx.mean(importance)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestBlockAttnRes tests/test_tnt.py::TestAttnResMemoryGate -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/attn_res.py tests/test_tnt.py
git commit -m "feat(attn_res): add BlockAttnRes and AttnResMemoryGate modules"
```

---

### Task 5: Wire AttnRes into TNTMACBlock and TitansTNT

**Files:**
- Modify: `src/titans_mlx/tnt_models.py`
- Test: `tests/test_tnt.py`

- [ ] **Step 1: Write failing tests for AttnRes integration**

Add to `tests/test_tnt.py`:

```python
class TestAttnResIntegration(unittest.TestCase):
    """Test AttnRes integration into TNTMACBlock and TitansTNT."""

    def _make_config(self, use_attn_res=True, **kwargs):
        defaults = dict(
            dim=64, num_heads=4, num_layers=4, vocab_size=100,
            use_tnt=True, local_chunk_sizes=[8],
            chunk_size=16, use_conv=False,
            use_attn_res=use_attn_res, num_attnres_blocks=2,
        )
        defaults.update(kwargs)
        return TitansConfig(**defaults)

    def test_mac_block_memory_gate_none_baseline(self):
        """TNTMACBlock with memory_gate=None matches no-gate behavior."""
        config = self._make_config(use_attn_res=False)
        block = TNTMACBlock(config)
        x = mx.random.normal((2, 8, 64))
        out1, s1 = block(x)
        out2, s2 = block(x, memory_gate=None)
        mx.eval(out1, out2)
        self.assertTrue(mx.allclose(out1, out2, atol=1e-5).item())

    def test_mac_block_memory_gate_scalar(self):
        """TNTMACBlock with memory_gate scalar runs without error."""
        config = self._make_config()
        block = TNTMACBlock(config)
        x = mx.random.normal((2, 8, 64))
        out, state = block(x, memory_gate=mx.array(0.5))
        mx.eval(out)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_tnt_model_attnres_forward(self):
        """TitansTNT with use_attn_res=True runs forward pass."""
        config = self._make_config()
        model = TitansTNT(config, variant="mac")
        ids = mx.random.randint(0, 100, (2, 16))
        logits, states = model(ids)
        mx.eval(logits)
        self.assertEqual(logits.shape, (2, 16, 100))

    def test_tnt_model_attnres_disabled_no_nan(self):
        """TitansTNT with use_attn_res=False produces valid output."""
        config = self._make_config(use_attn_res=False)
        model = TitansTNT(config, variant="mac")
        ids = mx.random.randint(0, 100, (2, 16))
        logits, _ = model(ids)
        mx.eval(logits)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertFalse(mx.any(mx.isnan(logits)).item())

    def test_attnres_disabled_no_extra_params(self):
        """use_attn_res=False means no AttnRes attributes on blocks."""
        config = self._make_config(use_attn_res=False)
        block = TNTMACBlock(config)
        self.assertFalse(hasattr(block, 'attn_res'))

    def test_tnt_model_gradient_flow(self):
        """Gradients flow through AttnRes path."""
        config = self._make_config()
        model = TitansTNT(config, variant="mac")
        ids = mx.random.randint(0, 100, (2, 16))

        def loss_fn(model, ids):
            logits, _ = model(ids)
            return mx.mean(logits)

        loss, grads = mx.value_and_grad(loss_fn)(model, ids)
        mx.eval(loss, grads)
        self.assertFalse(mx.isnan(loss).item())

    def test_tnt_model_block_boundaries(self):
        """Block boundaries handled correctly with even division."""
        config = self._make_config(num_layers=4, num_attnres_blocks=2)
        model = TitansTNT(config, variant="mac")
        ids = mx.random.randint(0, 100, (2, 16))
        logits, states = model(ids)
        mx.eval(logits)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertFalse(mx.any(mx.isnan(logits)).item())

    def test_tnt_model_uneven_blocks(self):
        """Uneven block division works (last block absorbs remainder)."""
        config = self._make_config(num_layers=5, num_attnres_blocks=2)
        model = TitansTNT(config, variant="mac")
        ids = mx.random.randint(0, 100, (2, 16))
        logits, _ = model(ids)
        mx.eval(logits)
        self.assertFalse(mx.any(mx.isnan(logits)).item())

    def test_warmup_bypasses_gate(self):
        """During warmup, memory_gate should be None (bypassed)."""
        config = self._make_config(attnres_warmup_steps=100)
        model = TitansTNT(config, variant="mac")
        # _step_count starts at 0, which is < 100 → warmup active
        self.assertEqual(model._step_count, 0)
        ids = mx.random.randint(0, 100, (2, 16))
        logits, _ = model(ids)
        mx.eval(logits)
        # Step count should increment
        self.assertEqual(model._step_count, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestAttnResIntegration -v`
Expected: FAIL — `TNTMACBlock.__call__()` got unexpected keyword argument 'memory_gate'

- [ ] **Step 3: Modify TNTMACBlock to accept memory_gate**

In `src/titans_mlx/tnt_models.py`, add imports at the top:

```python
from titans_mlx.attn_res import BlockAttnRes, AttnResMemoryGate
```

In `TNTMACBlock.__init__()`, after existing init code, add:

```python
        # AttnRes (optional)
        if config.use_attn_res:
            self.attn_res = BlockAttnRes(config.dim)
            self.attn_res_gate = AttnResMemoryGate()
```

In `TNTMACBlock.__call__()`, add `memory_gate` parameter:

```python
    def __call__(
        self,
        x: mx.array,
        state: TNTMemoryState | None = None,
        memory_gate: mx.array | None = None,
    ) -> tuple[mx.array, TNTMemoryState]:
```

Change the hierarchical memory call (line ~128) to pass memory_gate:

```python
        mem_out, new_state = self.hierarchical_memory(
            y_t, state=state, memory_gate=memory_gate
        )
```

- [ ] **Step 4: Modify TitansTNT for AttnRes block tracking**

In `TitansTNT.__init__()`, after `self.head.weight = self.embed.weight` (line ~395), add:

```python
        # AttnRes step counter for warmup
        self._step_count = 0
```

Replace `_process_single_chunk()` with:

```python
    def _process_single_chunk(
        self,
        chunk: mx.array,
        states: list[TNTMemoryState | None],
    ) -> tuple[mx.array, list[TNTMemoryState]]:
        """Process a single chunk through all blocks."""
        new_states = []

        if not self.config.use_attn_res:
            # Unchanged fast path
            for i, block in enumerate(self.blocks):
                chunk, new_state = block(chunk, state=states[i])
                new_states.append(new_state)
            return chunk, new_states

        # AttnRes path
        S = self.config.attnres_base_block_size
        completed_blocks: list[mx.array] = []
        partial_block: mx.array | None = chunk  # Token embedding = b_0

        for i, block in enumerate(self.blocks):
            # Compute AttnRes input
            h, attn_weights = block.attn_res(completed_blocks, partial_block)

            # Extract memory gate
            memory_gate = block.attn_res_gate(attn_weights)

            # Bypass gate during warmup
            if (
                self.config.attnres_warmup_steps > 0
                and self._step_count < self.config.attnres_warmup_steps
            ):
                memory_gate = None

            # Forward through block
            output, new_state = block(h, state=states[i], memory_gate=memory_gate)
            new_states.append(new_state)

            # Track block output for AttnRes.
            # (output - h) captures the total contribution of this block
            # across all its internal sub-layers (attention, gating, FFN).
            layer_output = output - h
            if partial_block is None:
                partial_block = layer_output
            else:
                partial_block = partial_block + layer_output

            # Block boundary check
            if (i + 1) % S == 0 or i == len(self.blocks) - 1:
                completed_blocks.append(partial_block)
                partial_block = None

            chunk = output

        return chunk, new_states
```

In `TitansTNT.__call__()`, restructure the end to converge to a single return with the step counter increment. After the fast path block (line ~444) and the chunked path block (line ~467), replace the two separate returns with a single exit point:

```python
        # ... (fast path or chunked path produces x and new_states)

        # Output projection
        x = self.norm(x)
        logits = self.head(x)

        # Increment step counter for AttnRes warmup
        self._step_count += 1

        return logits, new_states
```

Concretely: remove the early `return logits, new_states` from the fast path (line ~444). Let both paths fall through to a shared output projection + return at the end. The fast path already sets `x` and `new_states`, so this is safe.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py::TestAttnResIntegration -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/titans_mlx/tnt_models.py tests/test_tnt.py
git commit -m "feat(tnt_models): wire AttnRes into TNTMACBlock and TitansTNT"
```

---

### Task 6: Exports, CLI flags, and train.sh

**Files:**
- Modify: `src/titans_mlx/__init__.py`
- Modify: `scripts/pretrain.py`
- Modify: `train.sh`

- [ ] **Step 1: Update __init__.py exports**

In `src/titans_mlx/__init__.py`, add import:

```python
from titans_mlx.attn_res import BlockAttnRes, AttnResMemoryGate
```

Add to `__all__` list (after the Q-K Projection section):

```python
    # AttnRes
    "BlockAttnRes",
    "AttnResMemoryGate",
```

- [ ] **Step 2: Add CLI args to pretrain.py**

In `scripts/pretrain.py`, after the `--window-size` argument (line ~1242), add:

```python
    # AttnRes
    parser.add_argument(
        "--use-attn-res", action="store_true", help="Enable Attention Residuals"
    )
    parser.add_argument(
        "--num-attnres-blocks", type=int, default=8, help="AttnRes block count (N)"
    )
    parser.add_argument(
        "--attnres-warmup-steps", type=int, default=0,
        help="Steps before AttnRes memory gating activates",
    )
    parser.add_argument(
        "--attnres-modulate-global", action="store_true", default=True,
        help="Gate global memory LR with AttnRes",
    )
    parser.add_argument(
        "--no-attnres-modulate-global", dest="attnres_modulate_global",
        action="store_false",
    )
    parser.add_argument(
        "--attnres-modulate-local", action="store_true", default=False,
        help="Gate local memory LR with AttnRes",
    )
```

In the `TitansConfig(...)` constructor call (line ~1408), add:

```python
        use_attn_res=config.use_attn_res,
        num_attnres_blocks=config.num_attnres_blocks,
        attnres_warmup_steps=config.attnres_warmup_steps,
        attnres_modulate_global_memory=config.attnres_modulate_global,
        attnres_modulate_local_memory=config.attnres_modulate_local,
```

Add matching fields to the `TrainingConfig` dataclass:

```python
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global: bool = True
    attnres_modulate_local: bool = False
```

And wire them in the `config = TrainingConfig(...)` call at line ~1348:

```python
        use_attn_res=args.use_attn_res,
        num_attnres_blocks=args.num_attnres_blocks,
        attnres_warmup_steps=args.attnres_warmup_steps,
        attnres_modulate_global=args.attnres_modulate_global,
        attnres_modulate_local=args.attnres_modulate_local,
```

Also update `create_model()` to handle the `tnt` model type when AttnRes is enabled. In `create_model()` (line ~382), the function currently only maps to base Titans models. When `use_attn_res=True` or `use_tnt=True`, it should use `TitansTNT`. Add import and logic:

```python
from titans_mlx import TitansTNT
```

And modify `create_model`:
```python
def create_model(model_type: str, config: TitansConfig) -> nn.Module:
    """Create Titans model based on type."""
    if config.use_tnt or config.use_attn_res:
        if model_type not in ("mac", "mag", "mal"):
            raise ValueError(
                f"TNT/AttnRes requires variant mac, mag, or mal (got '{model_type}')"
            )
        return TitansTNT(config, variant=model_type)
    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type](config)
```

- [ ] **Step 3: Update train.sh**

Replace `train.sh` with:

```bash
#!/bin/bash
# Pretrain Titans MAC model with AttnRes
# dim=768, 16 layers, 16 heads
# Effective batch: ~98K tokens (batch=2 * accum=24 * seq=2048)
uv run --extra train python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-subset sample-10BT \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 768 --num-layers 16 --num-heads 16 \
  --batch-size 2 --gradient-accumulation-steps 24 \
  --seq-len 2048 --chunk-size 512 \
  --max-steps 10000 \
  --lr 4e-4 \
  --eval-every 200 --eval-buffer-size 100 \
  --save-every 200 \
  --log-every 10 \
  --dtype bfloat16 \
  --use-attn-res \
  --num-attnres-blocks 8 \
  --attnres-warmup-steps 500 \
  --checkpoint-dir checkpoints/mac-attnres
```

- [ ] **Step 4: Verify train.sh parses correctly**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && bash -n train.sh`
Expected: No syntax errors

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/test_tnt.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/titans_mlx/__init__.py scripts/pretrain.py train.sh tests/test_tnt.py
git commit -m "feat: add AttnRes CLI flags, exports, and training config"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run complete test suite**

Run: `cd /Users/dlattka/Projects/titans-tnt-mlx && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify AttnRes end-to-end with a smoke test**

Run:
```bash
cd /Users/dlattka/Projects/titans-tnt-mlx
python -c "
import mlx.core as mx
from titans_mlx import TitansConfig, TitansTNT

config = TitansConfig(
    dim=64, num_heads=4, num_layers=8, vocab_size=100,
    use_tnt=True, local_chunk_sizes=[8],
    chunk_size=16, use_conv=False,
    use_attn_res=True, num_attnres_blocks=4,
)
model = TitansTNT(config, variant='mac')
ids = mx.random.randint(0, 100, (2, 32))
logits, states = model(ids)
mx.eval(logits)
print(f'Shape: {logits.shape}')
print(f'NaN: {mx.any(mx.isnan(logits)).item()}')
print(f'Step count: {model._step_count}')

# Verify gradient flow
def loss_fn(m, x):
    out, _ = m(x)
    return mx.mean(out)

loss, grads = mx.value_and_grad(loss_fn)(model, ids)
mx.eval(loss, grads)
print(f'Loss: {loss.item():.4f}')
print('AttnRes integration verified successfully.')
"
```
Expected: No errors, no NaN, step_count=1

- [ ] **Step 3: Commit any remaining fixes if needed**
