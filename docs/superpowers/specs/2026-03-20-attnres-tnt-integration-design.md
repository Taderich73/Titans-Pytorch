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
        partial_block: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute AttnRes input for this layer.

        Args:
            blocks: Completed block representations [b_0, ..., b_{n-1}],
                     each shape (batch, seq, dim)
            partial_block: Current intra-block partial sum (batch, seq, dim),
                           or None if first layer in block

        Returns:
            Tuple of:
              - h_l: attention-weighted input (batch, seq, dim)
              - attn_weights: attention distribution (batch, seq, num_sources)
                for use by AttnResMemoryGate
        """
        ...
```

**Key implementation details:**
- Values V = stack of `[blocks..., partial_block]` → shape `(n+1, B, T, D)` (or `(n, B, T, D)` if first layer in first block, with only token embedding)
- Keys K = `RMSNorm(V)` per the paper
- Logits = `einsum('d, n b t d -> n b t', w_l, K)` → softmax over n dimension
- Output h = `einsum('n b t, n b t d -> b t d', α, V)`
- Token embedding `b_0 = h_1` is always included as the first source (the paper defines `v_0 = h_1`)

### 2.4 Pseudo-Query Initialization

All pseudo-query vectors MUST be initialized to zero. This ensures:
- Initial attention weights are uniform across sources (1/N)
- Training starts equivalent to standard residuals
- Prevents early training volatility (validated empirically in the paper)

## 3. Component 2: AttnResMemoryGate

### 3.1 Mechanism

The AttnRes attention weights encode depth-wise importance. The weight assigned to the most recent source (current partial block) indicates how structurally important the current representation is. This scalar gates the memory learning rate:

```
effective_lr = base_lr × importance_weight
```

When `importance_weight` is high → the model values the current processing → commit strongly to memory.
When `importance_weight` is low → the model relies on earlier representations → write less to memory.

Combined with the existing Titans surprise mechanism:
```
Update = effective_lr × Surprise × gradient
       = (base_lr × attnres_weight) × Surprise × gradient
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
def attnres_block_size(self) -> int:
    """S — number of TNTMACBlocks per AttnRes block."""
    return self.num_layers // self.num_attnres_blocks
```

**Validation:** If `num_layers` is not evenly divisible by `num_attnres_blocks`, the last AttnRes block absorbs the remainder.

**Serialization:** All new fields added to `to_dict()` / `from_dict()`.

## 5. Integration Points

### 5.1 TitansTNT._process_single_chunk()

Currently a simple loop over blocks. With AttnRes enabled:

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
    S = self.config.attnres_block_size
    completed_blocks = []      # List of completed block representations
    partial_block = chunk      # Start with token embedding as b_0
    # Note: chunk here IS the embedded input h_1, which serves as v_0

    for i, block in enumerate(self.blocks):
        # Compute AttnRes input
        h, attn_weights = block.attn_res(completed_blocks, partial_block)

        # Extract memory gate
        memory_gate = block.attn_res_gate(attn_weights)

        # Check warmup
        if (self.config.attnres_warmup_steps > 0
                and self._step_count < self.config.attnres_warmup_steps):
            memory_gate = None

        # Forward through block
        output, new_state = block(h, state=states[i], memory_gate=memory_gate)
        new_states.append(new_state)

        # Track block output (f_l(h_l)) for AttnRes
        layer_output = output - h  # Residual: what this layer actually added
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

### 5.2 TNTMACBlock.__call__() Signature Change

```python
def __call__(
    self,
    x: mx.array,
    state: TNTMemoryState | None = None,
    memory_gate: mx.array | None = None,    # NEW
) -> tuple[mx.array, TNTMemoryState]:
```

The `memory_gate` is passed through to `self.hierarchical_memory()`.

### 5.3 HierarchicalMemory.__call__() Change

```python
def __call__(
    self,
    x: mx.array,
    state: TNTMemoryState | None = None,
    memory_gate: mx.array | None = None,    # NEW
) -> tuple[mx.array, TNTMemoryState]:
```

Before calling `self.global_memory(x, state=...)`, if `memory_gate is not None` and `config.attnres_modulate_global_memory`:
- Temporarily scale the memory's learning rate: `effective_lr = self.config.memory_lr * memory_gate`

Same pattern for local memories if `config.attnres_modulate_local_memory`.

The learning rate scaling is applied by passing the gate value into `NeuralLongTermMemory`, which multiplies its `self.lr` by the gate before computing the update. This requires a minor change to `NeuralLongTermMemory.__call__()` to accept an optional `lr_scale` parameter.

### 5.4 NeuralLongTermMemory.__call__() Change

```python
def __call__(
    self,
    x: mx.array,
    state: MemoryState | None = None,
    lr_scale: float | mx.array = 1.0,       # NEW
) -> tuple[mx.array, MemoryState]:
    ...
    effective_lr = self.lr * lr_scale
    # Use effective_lr instead of self.lr in gradient update
```

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
   - Single block (no prior blocks) → identity-like behavior
   - Multiple blocks → correct aggregation
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

4. **TestAttnResConfig**
   - New fields serialize/deserialize correctly
   - `attnres_block_size` property
   - Validation of block count vs layer count

## 8. Files Changed

| File | Change |
|------|--------|
| `src/titans_mlx/attn_res.py` | **NEW** — BlockAttnRes, AttnResMemoryGate |
| `src/titans_mlx/config.py` | Add AttnRes config fields, `attnres_block_size` property |
| `src/titans_mlx/tnt_models.py` | TNTMACBlock: add attn_res, accept memory_gate. TitansTNT: AttnRes block tracking in chunk processing |
| `src/titans_mlx/tnt_memory.py` | HierarchicalMemory: accept and apply memory_gate |
| `src/titans_mlx/memory.py` | NeuralLongTermMemory: accept lr_scale parameter |
| `tests/test_tnt.py` | New test classes for AttnRes |

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
- `memory_gate=None` throughout the call chain = current behavior preserved
