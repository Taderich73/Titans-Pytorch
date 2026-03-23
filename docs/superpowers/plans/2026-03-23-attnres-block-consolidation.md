# AttnRes Integration & Block Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AttnRes a composable residual-replacement across MAC/MAG/MAL blocks, consolidate 6 block classes into 3 with config-driven memory, and align with the Attention Residuals paper.

**Architecture:** Blocks expose `core_forward`/`ffn_forward` sub-layers. A shared `process_chunk` function handles both standard residuals and AttnRes orchestration. Memory type (single vs hierarchical) is selected by config flag, eliminating TNT block duplicates.

**Tech Stack:** Python 3.13, MLX, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-attnres-block-consolidation-design.md`

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `src/titans_mlx/models.py` | All block classes (MAC, MAG, MAL, LMM) and model classes | Major refactor |
| `src/titans_mlx/tnt_models.py` | TNT block/model duplicates | Delete |
| `src/titans_mlx/memory.py` | `NeuralLongTermMemory` | Add `memory_gate` param |
| `src/titans_mlx/config.py` | Config properties | Replace `attnres_base_block_size` |
| `src/titans_mlx/__init__.py` | Public exports | Update |
| `src/titans_mlx/attn_res.py` | `BlockAttnRes`, `AttnResMemoryGate` | Unchanged |
| `src/titans_mlx/tnt_memory.py` | `HierarchicalMemory` et al. | Unchanged |
| `scripts/pretrain.py` | Training script, model routing | Fix routing + checkpoint migration |
| `examples/tnt_usage.py` | Usage examples | Rewrite for consolidated API |
| `tests/test_models.py` | Block and model tests | Add AttnRes + TNT config tests |
| `tests/test_tnt.py` | TNT-specific tests | Update imports |

---

### Task 1: Add `memory_gate` parameter to `NeuralLongTermMemory`

Align the interface so blocks can call `self.memory(..., memory_gate=gate)` regardless of memory type.

**Files:**
- Modify: `src/titans_mlx/memory.py:688-694`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write failing test**

In `tests/test_memory.py`, add:
```python
def test_memory_gate_parameter():
    """memory_gate should modulate lr_scale."""
    config = TitansConfig(dim=32, num_heads=4, num_layers=2, vocab_size=100)
    mem = NeuralLongTermMemory(config)
    x = mx.random.normal((1, 8, 32))
    state = mem.init_state(1)

    # With memory_gate=None, should behave like lr_scale=1.0
    out_none, st_none = mem(x, state=state, memory_gate=None)
    out_default, st_default = mem(x, state=state)
    assert mx.allclose(out_none, out_default, atol=1e-5)

    # With memory_gate=scalar, should modulate learning rate
    gate = mx.array(0.5)
    out_gated, _ = mem(x, state=state, memory_gate=gate)
    out_lr, _ = mem(x, state=state, lr_scale=0.5)
    assert mx.allclose(out_gated, out_lr, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dlattka/Code/dlattka/titans-tnt-mlx && uv run pytest tests/test_memory.py::test_memory_gate_parameter -v`
Expected: FAIL — `memory_gate` is not a valid parameter

- [ ] **Step 3: Add `memory_gate` parameter**

In `src/titans_mlx/memory.py`, modify the `__call__` signature at line 688:

```python
def __call__(
    self,
    x: mx.array,
    state: MemoryState | None = None,
    return_state: bool = True,
    lr_scale: float | mx.array = 1.0,
    memory_gate: mx.array | None = None,
) -> tuple[mx.array, MemoryState | None]:
```

Add before line 756 (the existing `theta = theta * lr_scale`):
```python
        # memory_gate overrides lr_scale when provided (interface alignment
        # with HierarchicalMemory which also accepts memory_gate)
        if memory_gate is not None:
            lr_scale = memory_gate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_memory.py::test_memory_gate_parameter -v`
Expected: PASS

- [ ] **Step 5: Run full memory test suite**

Run: `uv run pytest tests/test_memory.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/titans_mlx/memory.py tests/test_memory.py
git commit -m "feat(memory): add memory_gate parameter to NeuralLongTermMemory for interface alignment"
```

---

### Task 2: Update config — replace `attnres_base_block_size`

**Files:**
- Modify: `src/titans_mlx/config.py:127-134`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

In `tests/test_config.py`, add:
```python
def test_attnres_sub_layer_block_size():
    """Sub-layer block size accounts for 2 sub-layers per block."""
    config = TitansConfig(
        dim=32, num_heads=4, num_layers=12, vocab_size=100,
        use_attn_res=True, num_attnres_blocks=4,
    )
    # 12 layers * 2 sub-layers / 4 blocks = 6 sub-layers per block
    assert config.attnres_sub_layer_block_size == 6

def test_attnres_sub_layer_block_size_default():
    """Default 8 blocks with 12 layers."""
    config = TitansConfig(
        dim=32, num_heads=4, num_layers=12, vocab_size=100,
        use_attn_res=True, num_attnres_blocks=8,
    )
    # 12 * 2 / 8 = 3
    assert config.attnres_sub_layer_block_size == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_attnres_sub_layer_block_size -v`
Expected: FAIL — property doesn't exist

- [ ] **Step 3: Replace property in config.py**

In `src/titans_mlx/config.py`, replace the `attnres_base_block_size` property (lines 127-134) with:

```python
    @property
    def attnres_sub_layer_block_size(self) -> int:
        """S — number of sub-layers per AttnRes block.

        Each transformer block has 2 sub-layers (core + FFN), so the total
        sub-layer count is num_layers * 2. When not evenly divisible by
        num_attnres_blocks, the last block absorbs the remainder.

        Returns at least 1 to prevent division-by-zero when num_layers
        is small relative to num_attnres_blocks.
        """
        return max(1, (self.num_layers * 2) // self.num_attnres_blocks)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (new tests pass, existing unrelated tests still pass)

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/config.py tests/test_config.py
git commit -m "feat(config): replace attnres_base_block_size with attnres_sub_layer_block_size"
```

---

### Task 3: Refactor MACBlock — `core_forward` / `ffn_forward` split

This is the template refactor. MAG and MAL follow the same pattern in later tasks.

**Files:**
- Modify: `src/titans_mlx/models.py:73-200` (MACBlock)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for sub-layer methods**

In `tests/test_models.py`, add the import `from titans_mlx.tnt_memory import HierarchicalMemory` at the top. Then add near the `TestMACBlock` class:

```python
class TestMACBlockSubLayers:
    """Test MACBlock core_forward / ffn_forward decomposition."""

    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            chunk_size=16,
        )
        self.block = MACBlock(self.config)
        self.batch, self.seq, self.dim = 2, 16, 32

    def test_core_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        core_out, new_state = self.block.core_forward(x, state=None)
        assert core_out.shape == (self.batch, self.seq, self.dim)
        assert new_state is not None

    def test_ffn_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        ffn_out = self.block.ffn_forward(x)
        assert ffn_out.shape == (self.batch, self.seq, self.dim)

    def test_sublayers_match_residual_path(self):
        """core + ffn with manual residuals should match old __call__ behavior."""
        x = mx.random.normal((self.batch, self.seq, self.dim))
        mx.eval(x)

        # Sub-layer path with standard residuals
        core_out, state = self.block.core_forward(x, state=None)
        h = x + core_out
        ffn_out = self.block.ffn_forward(h)
        result_sublayers = h + ffn_out

        # Verify shapes match
        assert result_sublayers.shape == (self.batch, self.seq, self.dim)

    def test_core_forward_with_memory_gate(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        gate = mx.array(0.5)
        core_out, state = self.block.core_forward(x, state=None, memory_gate=gate)
        assert core_out.shape == (self.batch, self.seq, self.dim)

    def test_tnt_memory_selection(self):
        """use_tnt=True should select HierarchicalMemory."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            use_tnt=True,
        )
        block = MACBlock(config)
        assert isinstance(block.memory, HierarchicalMemory)

    def test_default_memory_selection(self):
        """use_tnt=False should select NeuralLongTermMemory."""
        block = MACBlock(self.config)
        assert isinstance(block.memory, NeuralLongTermMemory)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py::TestMACBlockSubLayers -v`
Expected: FAIL — `core_forward`, `ffn_forward` don't exist, `HierarchicalMemory` not used

- [ ] **Step 3: Refactor MACBlock**

In `src/titans_mlx/models.py`, refactor `MACBlock` (lines 73-200). The key changes:

1. **`__init__`**: Replace `self.memory = NeuralLongTermMemory(config)` with config-driven selection. Add conditional AttnRes modules.

2. **Split `__call__` into `core_forward` + `ffn_forward`**:

```python
class MACBlock(nn.Module):
    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Config-driven memory selection
        if config.use_tnt:
            from titans_mlx.tnt_memory import HierarchicalMemory
            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)

        # Learned query for memory retrieval
        self.memory_query = mx.random.normal((1, 1, config.dim)) * config.init_std

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Segmented attention
        self.attention = SegmentedAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        # Gating normalization
        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

        # AttnRes (optional)
        if config.use_attn_res:
            from titans_mlx.attn_res import BlockAttnRes, AttnResMemoryGate
            self.attn_res_core = BlockAttnRes(config.dim)
            self.attn_res_ffn = BlockAttnRes(config.dim)
            self.attn_res_gate = AttnResMemoryGate()

    def core_forward(
        self,
        h: mx.array,
        state: MemoryState | None = None,
        memory_gate: mx.array | None = None,
    ) -> tuple[mx.array, MemoryState]:
        """Core sub-layer: retrieve + attention + memory update + gating.

        Args:
            h: Input (batch, seq, dim) — from residual stream or AttnRes
            state: Memory state from previous chunk
            memory_gate: Optional scalar importance weight from AttnRes

        Returns:
            Tuple of (core_out, new_state). core_out is the net contribution
            (attn_out + gated), excluding h.
        """
        batch_size = h.shape[0]

        # Initialize memory state if needed
        if state is None:
            state = self.memory.init_state(batch_size)

        # Retrieve from memory using learned query
        query = mx.broadcast_to(self.memory_query, (batch_size, 1, self.config.dim))
        memory_retrieved = self.memory.retrieve(query, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        # Attention on [persistent || memory || norm(h)]
        persistent = self.persistent(batch_size)
        normed = self.norm1(h)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)

        # Internal residual: memory and gating see full representation
        y_t = h + attn_out

        # Memory update
        mem_out, new_state = self.memory(y_t, state=state, memory_gate=memory_gate)

        # Gating
        gated = mx.sigmoid(self.gate_norm_attn(y_t)) * mx.sigmoid(
            self.gate_norm_mem(mem_out)
        )

        # Net contribution (excludes h)
        core_out = attn_out + gated
        return core_out, new_state

    def ffn_forward(self, h: mx.array) -> mx.array:
        """FFN sub-layer.

        Args:
            h: Input (batch, seq, dim)

        Returns:
            FFN output (net contribution, excludes h).
        """
        normed = self.norm2(h)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        return ffn_out
```

Keep `__call__` as a backward-compatible wrapper so existing model classes and tests continue working between tasks:

```python
    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
        """Backward-compatible wrapper: standard residuals over sub-layers."""
        core_out, new_state = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state
```

- [ ] **Step 4: Run new tests**

Run: `uv run pytest tests/test_models.py::TestMACBlockSubLayers -v`
Expected: PASS

- [ ] **Step 5: Run full MACBlock tests (existing tests still work via `__call__` wrapper)**

Run: `uv run pytest tests/test_models.py::TestMACBlock tests/test_models.py::TestMACBlockSubLayers -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/titans_mlx/models.py tests/test_models.py
git commit -m "refactor(mac): split MACBlock into core_forward/ffn_forward with config-driven memory"
```

---

### Task 4: Refactor MAGBlock — same pattern as MAC

**Files:**
- Modify: `src/titans_mlx/models.py:347-445` (MAGBlock)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for MAGBlock sub-layers**

Mirror the `TestMACBlockSubLayers` pattern for MAG. Key differences: MAG uses `SlidingWindowAttention`, memory receives `normed` (not `y_t`).

```python
class TestMAGBlockSubLayers:
    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            chunk_size=16, window_size=16,
        )
        self.block = MAGBlock(self.config)
        self.batch, self.seq, self.dim = 2, 16, 32

    def test_core_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        core_out, new_state = self.block.core_forward(x, state=None)
        assert core_out.shape == (self.batch, self.seq, self.dim)

    def test_ffn_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        ffn_out = self.block.ffn_forward(x)
        assert ffn_out.shape == (self.batch, self.seq, self.dim)

    def test_tnt_memory_selection(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            use_tnt=True, window_size=16,
        )
        block = MAGBlock(config)
        assert isinstance(block.memory, HierarchicalMemory)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models.py::TestMAGBlockSubLayers -v`
Expected: FAIL

- [ ] **Step 3: Refactor MAGBlock**

Same pattern as MACBlock. Key difference in `core_forward`: MAG's memory receives `normed` (the pre-attention normalized input), not `y_t`:

```python
def core_forward(self, h, state=None, memory_gate=None):
    batch_size = h.shape[0]
    if state is None:
        state = self.memory.init_state(batch_size)

    persistent = self.persistent(batch_size)

    # Attention
    normed = self.norm1(h)
    attn_out = self.attention(normed, prefix=persistent)
    if self.dropout_p > 0:
        attn_out = nn.Dropout(self.dropout_p)(attn_out)
    y_t = h + attn_out  # internal residual

    # Memory receives normalized input (not y_t — MAG design)
    if persistent is not None:
        mem_input = mx.concatenate([persistent, normed], axis=1)
    else:
        mem_input = normed
    mem_out_full, new_state = self.memory(mem_input, state=state, memory_gate=memory_gate)
    if persistent is not None:
        mem_out = mem_out_full[:, persistent.shape[1]:, :]
    else:
        mem_out = mem_out_full

    # Gating
    gated = mx.sigmoid(self.gate_norm_attn(y_t)) * mx.sigmoid(
        self.gate_norm_mem(mem_out)
    )

    core_out = attn_out + gated
    return core_out, new_state
```

`ffn_forward` is identical to MACBlock's. Keep `__call__` as a backward-compatible wrapper (same pattern as MACBlock Task 3).

- [ ] **Step 4: Run tests (existing MAGBlock tests still work via `__call__`)**

Run: `uv run pytest tests/test_models.py::TestMAGBlock tests/test_models.py::TestMAGBlockSubLayers -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/models.py tests/test_models.py
git commit -m "refactor(mag): split MAGBlock into core_forward/ffn_forward with config-driven memory"
```

---

### Task 5: Refactor MALBlock — same pattern, different core

**Files:**
- Modify: `src/titans_mlx/models.py:517-601` (MALBlock)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Same pattern. MAL's core is memory → attention (opposite order from MAC/MAG).

```python
class TestMALBlockSubLayers:
    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            chunk_size=16, window_size=16,
        )
        self.block = MALBlock(self.config)
        self.batch, self.seq, self.dim = 2, 16, 32

    def test_core_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        core_out, new_state = self.block.core_forward(x, state=None)
        assert core_out.shape == (self.batch, self.seq, self.dim)

    def test_ffn_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        ffn_out = self.block.ffn_forward(x)
        assert ffn_out.shape == (self.batch, self.seq, self.dim)
```

- [ ] **Step 2: Run to verify failure, then implement**

MAL's `core_forward`:
```python
def core_forward(self, h, state=None, memory_gate=None):
    batch_size = h.shape[0]
    if state is None:
        state = self.memory.init_state(batch_size)

    persistent = self.persistent(batch_size)

    # Memory layer first (MAL: memory before attention)
    normed = self.norm1(h)
    if persistent is not None:
        mem_input = mx.concatenate([persistent, normed], axis=1)
    else:
        mem_input = normed
    mem_out_full, new_state = self.memory(mem_input, state=state, memory_gate=memory_gate)
    if persistent is not None:
        mem_out = mem_out_full[:, persistent.shape[1]:, :]
    else:
        mem_out = mem_out_full
    if self.dropout_p > 0:
        mem_out = nn.Dropout(self.dropout_p)(mem_out)

    # Internal residual: attention sees h + mem_out
    h_mid = h + mem_out

    # Attention
    normed = self.norm2(h_mid)
    attn_out = self.attention(normed, prefix=persistent)
    if self.dropout_p > 0:
        attn_out = nn.Dropout(self.dropout_p)(attn_out)

    core_out = mem_out + attn_out
    return core_out, new_state
```

Note: MAL uses `self.norm3` for FFN (not `self.norm2`). Adjust `ffn_forward` accordingly. Keep `__call__` as a backward-compatible wrapper (same pattern as MACBlock Task 3).

- [ ] **Step 3: Run tests (existing MALBlock tests still work via `__call__`)**

Run: `uv run pytest tests/test_models.py::TestMALBlock tests/test_models.py::TestMALBlockSubLayers -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/titans_mlx/models.py tests/test_models.py
git commit -m "refactor(mal): split MALBlock into core_forward/ffn_forward with config-driven memory"
```

---

### Task 6: Implement shared `process_chunk` orchestrator

The core of the AttnRes fix — a single function handling both residual paths.

**Files:**
- Modify: `src/titans_mlx/models.py` (add function before model classes)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
class TestProcessChunk:
    """Test the shared process_chunk orchestrator."""

    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16,
        )

    def test_standard_residual_path(self):
        """Without AttnRes, process_chunk applies standard residuals."""
        blocks = [MACBlock(self.config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        output, new_states = process_chunk(blocks, chunk, states, self.config, step_count=0)
        assert output.shape == (2, 16, 32)
        assert len(new_states) == 4

    def test_attnres_path(self):
        """With AttnRes, process_chunk uses AttnRes orchestration."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
        )
        blocks = [MACBlock(config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        output, new_states = process_chunk(blocks, chunk, states, config, step_count=0)
        assert output.shape == (2, 16, 32)
        assert len(new_states) == 4

    def test_attnres_warmup_bypasses_gate(self):
        """During warmup, memory_gate should be None."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
            attnres_warmup_steps=100,
        )
        blocks = [MACBlock(config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        # step_count=0 < warmup=100, so gate should be bypassed
        output, _ = process_chunk(blocks, chunk, states, config, step_count=0)
        assert output.shape == (2, 16, 32)

    def test_attnres_with_tnt_memory(self):
        """AttnRes + TNT memory should work together."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, use_tnt=True,
            num_attnres_blocks=2,
        )
        blocks = [MACBlock(config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        output, new_states = process_chunk(blocks, chunk, states, config, step_count=0)
        assert output.shape == (2, 16, 32)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models.py::TestProcessChunk -v`
Expected: FAIL — `process_chunk` not defined

- [ ] **Step 3: Implement `process_chunk`**

Add to `src/titans_mlx/models.py`, before the model classes:

```python
def process_chunk(
    blocks: list,
    chunk: mx.array,
    states: list,
    config: TitansConfig,
    step_count: int = 0,
) -> tuple[mx.array, list]:
    """Process a single chunk through all blocks.

    Handles both standard residuals and AttnRes paths.
    Used by TitansMAC, TitansMAG, TitansMAL.

    Args:
        blocks: List of block modules (MACBlock, MAGBlock, or MALBlock)
        chunk: Embedded input (batch, seq, dim)
        states: List of memory states, one per block
        config: Model configuration
        step_count: Current training step (for AttnRes warmup)

    Returns:
        Tuple of (output, new_states)
    """
    new_states = []

    if not config.use_attn_res:
        # Standard residual path
        x = chunk
        for i, block in enumerate(blocks):
            core_out, new_state = block.core_forward(x, state=states[i])
            x = x + core_out
            ffn_out = block.ffn_forward(x)
            x = x + ffn_out
            new_states.append(new_state)
        return x, new_states

    # AttnRes path — replaces residual connections per paper
    S = config.attnres_sub_layer_block_size
    completed_blocks: list[mx.array] = [chunk]  # b_0 = embedding
    partial_block: mx.array | None = None
    sub_idx = 0
    warmup = (
        config.attnres_warmup_steps > 0
        and step_count < config.attnres_warmup_steps
    )

    for i, block in enumerate(blocks):
        # --- Core sub-layer ---
        h, attn_weights = block.attn_res_core(completed_blocks, partial_block)

        # Memory gate (bypassed during warmup)
        memory_gate = None
        if not warmup:
            memory_gate = block.attn_res_gate(attn_weights)

        core_out, new_state = block.core_forward(h, state=states[i], memory_gate=memory_gate)
        new_states.append(new_state)

        if partial_block is None:
            partial_block = core_out
        else:
            partial_block = partial_block + core_out
        sub_idx += 1

        # Block boundary check
        if sub_idx % S == 0:
            completed_blocks.append(partial_block)
            partial_block = None

        # --- FFN sub-layer ---
        h, _ = block.attn_res_ffn(completed_blocks, partial_block)
        ffn_out = block.ffn_forward(h)

        if partial_block is None:
            partial_block = ffn_out
        else:
            partial_block = partial_block + ffn_out
        sub_idx += 1

        # Block boundary check
        if sub_idx % S == 0 or i == len(blocks) - 1:
            completed_blocks.append(partial_block)
            partial_block = None

        # Track model output (last hidden state)
        chunk = h + ffn_out

    return chunk, new_states
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_models.py::TestProcessChunk -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/models.py tests/test_models.py
git commit -m "feat: add shared process_chunk orchestrator with AttnRes support"
```

---

### Task 7: Wire model classes to use `process_chunk`

Replace each model class's internal loop with `process_chunk`. Add `_step_count`.

**Files:**
- Modify: `src/titans_mlx/models.py` (TitansMAC, TitansMAG, TitansMAL)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
class TestTitansMACAttnRes:
    """Test TitansMAC with AttnRes enabled."""

    def test_forward_with_attn_res(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAC(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_forward_with_attn_res_and_tnt(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, use_tnt=True,
            num_attnres_blocks=2,
        )
        model = TitansMAC(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_step_count_increments(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAC(config)
        input_ids = mx.array([[1, 2, 3, 4]])
        model(input_ids)
        assert model._step_count == 1
        model(input_ids)
        assert model._step_count == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models.py::TestTitansMACAttnRes -v`
Expected: FAIL

- [ ] **Step 3: Refactor TitansMAC**

Update `TitansMAC.__init__` to add `self._step_count = 0`.

Replace `_process_single_chunk` and `_process_all_chunks_compiled` with a call to `process_chunk`:

```python
def __call__(self, input_ids, states=None):
    batch_size, seq_len = input_ids.shape
    chunk_size = self.config.chunk_size

    if states is None:
        states = [None] * len(self.blocks)

    x = self.embed(input_ids)

    if seq_len <= chunk_size:
        x, new_states = process_chunk(
            self.blocks, x, states, self.config, self._step_count
        )
    else:
        outputs = []
        new_states = list(states)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk = x[:, chunk_start:chunk_end]
            chunk, new_states = process_chunk(
                self.blocks, chunk, new_states, self.config, self._step_count
            )
            outputs.append(chunk)
        x = mx.concatenate(outputs, axis=1)

    x = self.norm(x)
    logits = self.head(x)
    self._step_count += 1
    return logits, new_states
```

Apply the same pattern to `TitansMAG` and `TitansMAL` (they currently don't chunk, but should for consistency — or keep their simple loop but route through `process_chunk`).

- [ ] **Step 4: Run all model tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: All PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/models.py tests/test_models.py
git commit -m "feat: wire TitansMAC/MAG/MAL to use process_chunk with AttnRes support"
```

---

### Task 8: Fix `pretrain.py` model routing

**Files:**
- Modify: `scripts/pretrain.py:50,387-405`

- [ ] **Step 1: Fix `create_model`**

Replace lines 387-405:
```python
def create_model(model_type: str, config: TitansConfig) -> nn.Module:
    """Create Titans model based on type."""
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

- [ ] **Step 2: Remove `TitansTNT` from imports**

At line 50, remove `TitansTNT` from the import:
```python
# Before:
from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL, TitansTNT

# After:
from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
```

- [ ] **Step 3: Add checkpoint key remapping**

In the `load_checkpoint` function (around line 733), add key remapping for old TNT checkpoints:

```python
def _remap_tnt_keys(weights: dict) -> dict:
    """Remap old TNT checkpoint keys to consolidated format.

    Old: blocks.N.hierarchical_memory.* -> New: blocks.N.memory.*
    """
    remapped = {}
    for k, v in weights.items():
        new_key = k.replace(".hierarchical_memory.", ".memory.")
        remapped[new_key] = v
    return remapped
```

Call this before loading weights into the model.

- [ ] **Step 4: Verify training script still works**

Run: `cd /Users/dlattka/Code/dlattka/titans-tnt-mlx && uv run python scripts/pretrain.py --help`
Expected: Help text with no import errors

- [ ] **Step 5: Commit**

```bash
git add scripts/pretrain.py
git commit -m "fix(pretrain): decouple use_attn_res from TitansTNT, add checkpoint key remapping"
```

---

### Task 9: Delete `tnt_models.py` and update exports

**Files:**
- Delete: `src/titans_mlx/tnt_models.py`
- Modify: `src/titans_mlx/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

Remove lines importing from `tnt_models`:
```python
# Remove these imports:
from titans_mlx.tnt_models import TNTMACBlock, TNTMAGBlock, TNTMALBlock, TitansTNT
```

Remove these from `__all__`:
```python
# Remove from __all__:
"TNTMACBlock", "TNTMAGBlock", "TNTMALBlock", "TitansTNT",
```

- [ ] **Step 2: Delete `tnt_models.py`**

```bash
git rm src/titans_mlx/tnt_models.py
```

- [ ] **Step 3: Verify no remaining references**

Run: `grep -r "tnt_models\|TNTMACBlock\|TNTMAGBlock\|TNTMALBlock\|TitansTNT" src/ scripts/ --include="*.py" | grep -v __pycache__`
Expected: No matches (except possibly in docs/specs)

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: Failures only in `tests/test_tnt.py` (TNT imports) — fixed in next task

- [ ] **Step 5: Commit**

```bash
git add src/titans_mlx/__init__.py
git rm src/titans_mlx/tnt_models.py
git commit -m "refactor: remove tnt_models.py, update exports"
```

---

### Task 10: Update tests and examples

**Files:**
- Modify: `tests/test_tnt.py`
- Modify: `examples/tnt_usage.py`

- [ ] **Step 1: Update `test_tnt.py` imports**

Replace imports of TNT block classes with base block classes + `use_tnt=True` config. For example:

```python
# Before:
from titans_mlx.tnt_models import TNTMACBlock, TitansTNT

# After:
from titans_mlx.models import MACBlock, TitansMAC
```

Update test configs to use `use_tnt=True`:
```python
config = TitansConfig(..., use_tnt=True)
block = MACBlock(config)  # Now uses HierarchicalMemory
```

Update test methods that call `block(x, state=...)` to use sub-layer API.

- [ ] **Step 2: Update `examples/tnt_usage.py`**

Replace `TitansTNT(config, variant=...)` with `TitansMAC(config)` / `TitansMAG(config)` / `TitansMAL(config)` using `use_tnt=True` in config. Update `example_tnt_model()` function.

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Run examples**

Run: `cd /Users/dlattka/Code/dlattka/titans-tnt-mlx && PYTHONPATH=src uv run python examples/tnt_usage.py`
Expected: Runs without errors

- [ ] **Step 5: Commit**

```bash
git add tests/test_tnt.py examples/tnt_usage.py
git commit -m "test: update TNT tests and examples for consolidated block API"
```

---

### Task 11: End-to-end integration test

Verify the full training loop works with all flag combinations.

**Files:**
- Test: `tests/test_models.py`

- [ ] **Step 1: Write integration tests**

```python
class TestFlagCombinations:
    """Verify all flag combinations produce valid forward passes."""

    @pytest.mark.parametrize("model_type,block_cls", [
        ("mac", MACBlock), ("mag", MAGBlock), ("mal", MALBlock),
    ])
    @pytest.mark.parametrize("use_tnt", [False, True])
    @pytest.mark.parametrize("use_attn_res", [False, True])
    def test_forward_pass(self, model_type, block_cls, use_tnt, use_attn_res):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, window_size=16,
            use_tnt=use_tnt, use_attn_res=use_attn_res,
            num_attnres_blocks=2,
        )
        model_cls = {"mac": TitansMAC, "mag": TitansMAG, "mal": TitansMAL}[model_type]
        model = model_cls(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)
        assert len(states) == 4
```

- [ ] **Step 2: Write multi-chunk AttnRes test**

```python
def test_multi_chunk_attn_res():
    """AttnRes should work when seq_len > chunk_size (multiple chunks)."""
    config = TitansConfig(
        dim=32, num_heads=4, num_layers=4, vocab_size=100,
        chunk_size=8, use_attn_res=True, num_attnres_blocks=2,
    )
    model = TitansMAC(config)
    # seq_len=16 > chunk_size=8 → 2 chunks
    input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    logits, states = model(input_ids)
    assert logits.shape == (1, 16, 100)

def test_checkpoint_key_remapping():
    """Old TNT checkpoint keys should be remapped to consolidated format."""
    old_keys = {
        "blocks.0.hierarchical_memory.global_memory.memory.proj_k.weight": mx.zeros((32, 32)),
        "blocks.0.norm1.weight": mx.ones((32,)),
    }
    from scripts.pretrain import _remap_tnt_keys
    new_keys = _remap_tnt_keys(old_keys)
    assert "blocks.0.memory.global_memory.memory.proj_k.weight" in new_keys
    assert "blocks.0.norm1.weight" in new_keys  # non-TNT keys unchanged
```

- [ ] **Step 3: Run integration tests**

Run: `uv run pytest tests/test_models.py::TestFlagCombinations tests/test_models.py::test_multi_chunk_attn_res tests/test_models.py::test_checkpoint_key_remapping -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_models.py
git commit -m "test: add integration tests for all flag combinations"
```
