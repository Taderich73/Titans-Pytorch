# AttnRes Integration into TitansTNT MAC — Design Spec

**Date:** 2026-03-20
**Scope:** TNTMACBlock / TitansTNT (MAC variant only)
**Papers:**
- Attention Residuals (Kimi Team, arXiv 2603.15031)
- TNT: Improving Chunkwise Training for Test-Time Memorization (Li et al., 2025)

## 1. Overview

Integrate Block Attention Residuals (Block AttnRes) into the TitansTNT MAC architecture with two separable components:

1. **Block AttnRes residual connections** — replace fixed residual accumulation between TNTMACBlocks with learned softmax attention over prior block representations
2. **AttnRes memory gate** — use the learned depth-wise attention weights to modulate the memory update learning rate in HierarchicalMemory

Both components are independently toggleable. When `use_attn_res=False` (default), behavior is identical to the current codebase with zero runtime cost.

## 2. Component 1: BlockAttnRes Module

### 2.1 Mechanism

Standard residuals accumulate all prior layer outputs with fixed unit weights:
```
h_l = h_{l-1} + f_{l-1}(h_{l-1}) = Σ v_i
```

AttnRes replaces this with learned softmax attention over depth:
```
h_l = Σ α_{i→l} · v_i
```

where `α_{i→l} = softmax(w_l^T · RMSNorm(k_i))` and `w_l ∈ R^d` is a learned pseudo-query vector per layer.

### 2.2 Block Variant

Full AttnRes requires O(Ld) memory. Block AttnRes groups L layers into N blocks of S layers:

- **Intra-block:** Standard residual connections. Each layer adds its output to a running partial sum `b_n^i`.
- **Inter-block:** Softmax attention over completed block representations `{b_0, ..., b_{n-1}}` plus the current partial sum.

For this project, each `TNTMACBlock` = one "layer" for AttnRes purposes. With 16 blocks and N=8 AttnRes blocks, S=2 TNTMACBlocks per AttnRes block.

### 2.3 Module Design

**New file: `src/titans_mlx/attn_res.py`**

```python
class BlockAttnRes(nn.Module):
    """Block Attention Residuals per AttnRes paper Eq. 2-6.

    Each TNTMACBlock owns one instance. Computes attention-weighted
    input from prior block representations + intra-block partial sum.

    Parameters:
        attn_res_proj: Linear(dim, 1, bias=False) — pseudo-query projection
        attn_res_norm: RMSNorm(dim) — normalizes keys before attention

    The pseudo-query w_l is the weight vector of attn_res_proj.
    Initialized to zero so initial attention weights are uniform.
    """

    def __init__(self, dim: int):
        ...

    def __call__(
        self,
        blocks: list[mx.array],
        partial_block: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Compute AttnRes input for this layer.

        Args:
            blocks: Completed block representations [b_0, ..., b_{n-1}],
                     each shape (batch, seq, dim)
            partial_block: Current intra-block partial sum (batch, seq, dim),
                           or None if this is the first layer in a new block.

        When partial_block is None:
            - If blocks is non-empty: attend only over completed blocks.
              This occurs at the first layer of any block after the first.
            - If blocks is also empty: this should not happen — the token
              embedding is always passed as the initial partial_block.

        When partial_block is not None:
            - Attend over [blocks..., partial_block]. The partial_block
              is always the last source in the attention.

        Returns:
            Tuple of:
              - h_l: attention-weighted input (batch, seq, dim)
              - attn_weights: attention distribution (batch, seq, num_sources)
                for use by AttnResMemoryGate
        """
        ...
```

**Key implementation details:**
- Values V: collected from `blocks` + `partial_block` (if not None). Shape `(num_sources, B, T, D)`.
- Keys K = `RMSNorm(V)` per the paper — prevents layers with large magnitudes from dominating.
- The pseudo-query weight `w_l` is obtained via `self.attn_res_proj.weight.squeeze(0)` to get shape `(d,)`, or equivalently, use `attn_res_proj(K)` to produce `(num_sources, B, T, 1)` logits then squeeze.
- Logits → softmax over the `num_sources` dimension.
- Output `h = Σ α_i · V_i` via einsum.
- Token embedding `b_0 = h_1` is always included as the first source (the paper defines `v_0 = h_1`). In `TitansTNT._process_single_chunk()`, the embedded input is passed as the initial `partial_block`.

### 2.4 Pseudo-Query Initialization

All pseudo-query vectors MUST be initialized to zero. This ensures:
- Initial attention weights are uniform across sources (1/N)
- Training starts equivalent to standard residuals
- Prevents early training volatility (validated empirically in the paper)

## 3. Component 2: AttnResMemoryGate

### 3.1 Mechanism

The AttnRes attention weights encode depth-wise importance. The weight assigned to the most recent source (current partial block) indicates how structurally important the current representation is. This scalar gates the memory learning rate:

```
effective_theta = theta × importance_weight
```

where `theta` is the existing data-dependent learning rate gate from `NeuralLongTermMemory` (computed as `sigmoid(gate_lr_proj(x_mean)) * config.memory_lr`).

When `importance_weight` is high → the model values the current processing → commit strongly to memory.
When `importance_weight` is low → the model relies on earlier representations → write less to memory.

Combined with the existing Titans surprise mechanism:
```
Update = effective_theta × gradient
       = (theta × attnres_weight) × gradient
```

### 3.2 Module Design

```python
class AttnResMemoryGate:
    """Extracts importance signal from AttnRes attention weights.

    Takes the attention weight assigned to the most recent source
    (the intra-block partial sum) as the importance signal.
    Returns a scalar multiplier for the memory learning rate.
    """

    def __call__(self, attn_weights: mx.array) -> mx.array:
        """Extract importance from AttnRes attention distribution.

        Args:
            attn_weights: shape (batch, seq, num_sources) from BlockAttnRes

        Returns:
            Scalar importance weight, averaged over batch and sequence.
            Shape: scalar or (1,)
        """
        # Weight on the last source (current partial block / most recent)
        importance = attn_weights[:, :, -1]  # (B, T)
        # Average over batch and sequence for a single scalar gate
        return mx.mean(importance)
```

**Why scalar averaging:** The memory weights in `NeuralLongTermMemory` are shared across the batch (not per-sample copies). The existing data-dependent gates (`theta`, `alpha`, `eta`) are already batch-averaged for the same reason (see `memory.py` lines 744-746). Per-sample gating would require per-sample weight copies, which is a different (much more expensive) design. The AttnRes gate follows the same scalar-averaging pattern for consistency.

### 3.3 Warmup

Two warmup mechanisms (both available, composable):

1. **Implicit (default):** Zero-init pseudo-queries → uniform weights → importance ≈ 1/N → memory LR starts at `base_lr/N`. As queries learn, the gate becomes selective.

2. **Explicit (optional):** `attnres_warmup_steps` config parameter. During warmup, `memory_gate=None` → use base_lr directly. After warmup, switch to AttnRes-modulated LR.

## 4. Config Changes

New fields on `TitansConfig`:

```python
# AttnRes configuration
use_attn_res: bool = False                       # Master toggle
num_attnres_blocks: int = 8                      # N — number of AttnRes blocks
attnres_warmup_steps: int = 0                    # Explicit warmup (0 = implicit only)
attnres_modulate_global_memory: bool = True      # Gate global memory LR
attnres_modulate_local_memory: bool = False       # Gate local memory LR
```

**Derived property:**
```python
@property
def attnres_base_block_size(self) -> int:
    """S — base number of TNTMACBlocks per AttnRes block.

    This is the minimum block size. When num_layers is not evenly
    divisible by num_attnres_blocks, the last block absorbs the
    remainder and will be larger than this value.
    """
    return self.num_layers // self.num_attnres_blocks
```

**Validation:** If `num_layers` is not evenly divisible by `num_attnres_blocks`, the last AttnRes block absorbs the remainder. The block boundary logic in `_process_single_chunk()` handles this via `(i + 1) % S == 0 or i == len(self.blocks) - 1`.

**Serialization:** All new fields added to `to_dict()` / `from_dict()`. Note: `from_dict()` uses `cls(**d)` which relies on dataclass defaults. Loading an older config (without AttnRes fields) into new code works because all new fields have defaults. Loading a new config into older code will raise `TypeError` on unknown keys — this is acceptable since AttnRes is a new feature requiring updated code.

## 5. Integration Points

### 5.1 TitansTNT Changes

**Step counter for warmup:** `TitansTNT.__init__()` initializes `self._step_count = 0`. It is incremented by 1 at the end of each `__call__()` invocation (not per chunk — per forward pass). This counts training steps for warmup comparison against `attnres_warmup_steps`.

**`_process_single_chunk()` — AttnRes path:**

```python
def _process_single_chunk(self, chunk, states):
    new_states = []

    if not self.config.use_attn_res:
        # Unchanged fast path
        for i, block in enumerate(self.blocks):
            chunk, new_state = block(chunk, state=states[i])
            new_states.append(new_state)
        return chunk, new_states

    # AttnRes path
    S = self.config.attnres_base_block_size
    completed_blocks = []      # List of completed block representations
    partial_block = chunk      # Start with token embedding as b_0
    # Note: chunk here IS the embedded input h_1, which serves as v_0

    for i, block in enumerate(self.blocks):
        # Compute AttnRes input
        h, attn_weights = block.attn_res(completed_blocks, partial_block)

        # Extract memory gate
        memory_gate = block.attn_res_gate(attn_weights)

        # Check warmup — bypass gate during warmup
        if (self.config.attnres_warmup_steps > 0
                and self._step_count < self.config.attnres_warmup_steps):
            memory_gate = None

        # Forward through block
        output, new_state = block(h, state=states[i], memory_gate=memory_gate)
        new_states.append(new_state)

        # Track block output for AttnRes.
        # The difference (output - h) captures the total contribution of this
        # block across all its internal sub-layers (attention residual, gated
        # memory output, FFN residual). This is the correct block-level
        # representation for AttnRes — it is what this block "added."
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

### 5.2 TNTMACBlock.__call__() Modified Body

```python
def __call__(
    self,
    x: mx.array,
    state: TNTMemoryState | None = None,
    memory_gate: mx.array | None = None,    # NEW
) -> tuple[mx.array, TNTMemoryState]:
```

The `memory_gate` is injected at the hierarchical memory call site. The modified body (showing only changed lines with # CHANGED comments):

```python
    # ... (steps 1-3 unchanged: memory retrieval, attention, residual)
    y_t = x + attn_out

    # Step 4: Update hierarchical memory — pass memory_gate for LR modulation
    mem_out, new_state = self.hierarchical_memory(
        y_t, state=state, memory_gate=memory_gate  # CHANGED: added memory_gate
    )

    # ... (steps 5-6 unchanged: gating, FFN)
```

The gate is applied at the memory update step (step 4 in the MAC architecture), which is the natural injection point — it modulates how strongly new information is written to memory. Steps 1-3 (retrieval, attention, residual) and steps 5-6 (gating, FFN) are unchanged.

### 5.3 HierarchicalMemory.__call__() Change

```python
def __call__(
    self,
    x: mx.array,
    state: TNTMemoryState | None = None,
    memory_gate: mx.array | None = None,    # NEW
) -> tuple[mx.array, TNTMemoryState]:
```

The parameter is named `memory_gate` at this level (semantic: "how important is this block's processing"). It is translated to `lr_scale` when passed to sub-memories (semantic: "scale factor for the learning rate"). This naming transition reflects the conceptual shift from a gating signal to a multiplicative scale factor.

Determines `lr_scale` for global and local memories based on config flags:
- If `memory_gate is not None` and `config.attnres_modulate_global_memory`: pass `lr_scale=memory_gate` to `self.global_memory(x, state=..., lr_scale=memory_gate)`
- If `memory_gate is not None` and `config.attnres_modulate_local_memory`: pass `lr_scale=memory_gate` to each `local_mem(x, state=..., lr_scale=memory_gate)`
- Otherwise: pass `lr_scale=1.0` (default, no modulation)

### 5.4 GlobalMemory and LocalMemory Signature Changes

Both `GlobalMemory.__call__()` and `LocalMemory.__call__()` gain an `lr_scale` parameter that is passed through to their inner `self.memory()` call:

```python
class GlobalMemory(nn.Module):
    def __call__(self, x, state=None, lr_scale=1.0):
        return self.memory(x, state=state, lr_scale=lr_scale)

class LocalMemory(nn.Module):
    def __call__(self, x, state=None, lr_scale=1.0):
        return self.memory(x, state=state, lr_scale=lr_scale)
```

### 5.5 NeuralLongTermMemory.__call__() Change

```python
def __call__(
    self,
    x: mx.array,
    state: MemoryState | None = None,
    lr_scale: float | mx.array = 1.0,       # NEW
) -> tuple[mx.array, MemoryState]:
```

The `lr_scale` is applied after computing the data-dependent gate `theta`:

```python
# Existing code computes theta:
theta = sigmoid(gate_lr_proj(x_mean)) * config.memory_lr  # (B, 1, 1)
theta = mx.mean(theta)  # scalar

# NEW: apply AttnRes modulation
theta = theta * lr_scale
```

This applies uniformly before either the linear parallel path (`_parallel_memory_update_linear`) or the deep gradient path, since both receive `theta` as a parameter. No changes needed inside the update methods themselves.

## 6. New Module Ownership

Each `TNTMACBlock` owns:
- `self.attn_res: BlockAttnRes` — its AttnRes attention module (when `use_attn_res=True`)
- `self.attn_res_gate: AttnResMemoryGate` — its memory gate extractor

These are created in `__init__` only when `config.use_attn_res=True`.

## 7. Testing Strategy

New test classes in `tests/test_tnt.py`:

1. **TestBlockAttnRes**
   - Attention weights sum to 1 (softmax property)
   - Zero-init queries → uniform weights
   - Output shape matches input shape
   - Single block (no prior blocks) → correct behavior with only partial_block
   - Multiple blocks → correct aggregation
   - partial_block=None with completed blocks → attends only over blocks
   - No NaN/Inf

2. **TestAttnResMemoryGate**
   - Returns scalar
   - Uniform weights → 1/N
   - Extreme weights → correct extraction
   - Gate value in [0, 1]

3. **TestAttnResIntegration**
   - TNTMACBlock with `memory_gate=None` → same as baseline
   - TNTMACBlock with `memory_gate` → different memory state evolution
   - TitansTNT with `use_attn_res=True` → forward/backward passes
   - TitansTNT with `use_attn_res=False` → identical to current behavior
   - Gradient flow through AttnRes path
   - Block boundary handling (even and uneven division)
   - Warmup: gate bypassed during warmup steps
   - lr_scale propagation: verify NeuralLongTermMemory receives and applies lr_scale

4. **TestAttnResConfig**
   - New fields serialize/deserialize correctly
   - `attnres_base_block_size` property
   - Validation of block count vs layer count

## 8. Files Changed

| File | Change |
|------|--------|
| `src/titans_mlx/attn_res.py` | **NEW** — BlockAttnRes, AttnResMemoryGate |
| `src/titans_mlx/__init__.py` | Export BlockAttnRes, AttnResMemoryGate |
| `src/titans_mlx/config.py` | Add AttnRes config fields, `attnres_base_block_size` property |
| `src/titans_mlx/tnt_models.py` | TNTMACBlock: add attn_res, accept memory_gate. TitansTNT: AttnRes block tracking, _step_count |
| `src/titans_mlx/tnt_memory.py` | HierarchicalMemory, GlobalMemory, LocalMemory: accept and propagate memory_gate / lr_scale |
| `src/titans_mlx/memory.py` | NeuralLongTermMemory: accept lr_scale, apply to theta |
| `tests/test_tnt.py` | New test classes for AttnRes |
| `train.sh` | Add AttnRes config flags to training command |

## 9. Parameter Overhead

Per TNTMACBlock (when AttnRes enabled):
- `attn_res_proj`: d parameters (768)
- `attn_res_norm`: d parameters (768)
- Total per block: 1,536 parameters

For 16 blocks: 24,576 parameters total. Negligible vs model size.

## 10. Backward Compatibility

- `use_attn_res=False` (default): zero code path changes, zero parameter overhead
- Existing `TNTMemoryState` dataclass: unchanged
- Existing tests: unchanged behavior
- MAG/MAL blocks: not modified (out of scope)
- `memory_gate=None` and `lr_scale=1.0` throughout the call chain = current behavior preserved
- `NeuralLongTermMemory` default `lr_scale=1.0` means existing callers (non-TNT Titans variants) are unaffected
