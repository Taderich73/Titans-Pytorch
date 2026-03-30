# Memory Cross-Attention & State Persistence (DMCA v2)

**Date:** 2026-03-30
**Status:** Approved
**Source spec:** `todos/memory-cross-attention-spec.md`
**Revision note:** Revised from original DMCA spec. Replaces separate slot-based memory bank with cross-attention to existing NeuralLTM weight space. Adds state serialization for inference-time continual learning.

---

## 1. Overview

This design adds two capabilities to the Titans-TNT-MLX architecture:

1. **Memory Cross-Attention (MCA):** A cross-attention mechanism that reads from NeuralLongTermMemory's weight matrix rows, giving the model a second read interface into the same memory that's already being written to by the surprise-driven update mechanism.

2. **Memory State Persistence:** A serialization system (dump/load/inspect/diff/merge/fork) for `MemoryState`, enabling inference-time continual learning across sessions without weight modification.

### Why not a separate memory bank?

The original DMCA spec proposed a standalone slot tensor M alongside NeuralLTM. This architecture already has a powerful learned memory (NeuralLTM's MLP weights, updated at test time via gradient descent). Adding a separate flat slot bank is:
- **Redundant:** Two surprise-driven memory systems learning the same signal
- **Training-hostile:** A separate M starts as zeros, requiring memory dropout to train cross-attention projections
- **More state to manage:** Separate MemoryBankState alongside MemoryState

Instead, we cross-attend directly to NeuralLTM's weight matrix rows. This gives:
- **No new state:** Reads from existing MemoryState
- **No training tricks:** Weight rows are meaningful from the start of training
- **No separate updates:** NeuralLTM's existing surprise mechanism handles writes
- **Two complementary reads:** MLP retrieval (nonlinear, precise) + cross-attention (linear blend of memory directions)

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Memory source | NeuralLTM weight rows | No redundant state; reads from existing learned memory |
| Which weight matrix | First layer (input-facing) | Captures key-space associations; later layers are nonlinear transforms |
| LMM support | No MCA | LMM stays pure memory baseline |
| Insertion layers | Auto midpoint, configurable | Generalizes spec intent to variable num_layers |
| AttnRes interaction | MCA as third sub-layer | Full AttnRes control over all pathways |
| Integration approach | Conditional component in existing blocks | Follows codebase patterns (use_tnt, use_attn_res) |
| Serialization format | npz + JSON metadata | Reuses existing save/load infrastructure |

### Scope

- **In scope:** MemoryCrossAttention module, MemoryDumpManager, config extensions, block modifications for MAC/MAG/MAL, process_chunk changes, quantization compatibility, tests
- **Out of scope:** LMM MCA support, training convergence ablations, memory dropout (not needed), separate memory bank

---

## 2. Memory Cross-Attention (MCA)

### 2.1 How it works

NeuralLTM's first weight matrix has shape `[memory_hidden_dim, dim]` (deep memory) or `[dim, dim]` (linear memory). Each row is a dim-dimensional vector encoding a learned association direction. Cross-attention treats these rows as addressable memory:

```
W = state.weights[0]               # [num_rows, dim] — first layer of memory MLP
Q = Wq(norm(x))                    # [B, T, dim] — what am I looking for?
K = Wk(W)                          # [num_rows, dim] — what associations exist?
V = Wv(W)                          # [num_rows, dim] — what do they contain?
attn = softmax(Q @ K^T / sqrt(d))  # [B, T, num_rows]
out = attn @ V                      # [B, T, dim]
gate = sigmoid(Wg(x))              # [B, T, 1] — near-zero initially
result = gate * Wo(out)             # [B, T, dim] — net contribution for residual
```

### 2.2 Why this differs from NeuralLTM's existing retrieval

| | MLP Retrieval (existing) | Cross-Attention (new) |
|---|---|---|
| Operation | `output = MLP(query)` | `output = softmax(Q @ K^T) @ V` |
| Nature | Nonlinear function of query | Linear blend of memory directions |
| What it captures | "What does memory predict for this exact input?" | "Which learned associations are most relevant?" |
| Selectivity | Hard — one output per query | Soft — weighted blend of all memory rows |

These are genuinely complementary reads on the same memory. The MLP retrieval is precise (good for exact key-value lookup). The cross-attention is exploratory (good for discovering which memory directions are active for the current context).

### 2.3 Number of "slots"

The number of rows to attend over depends on the memory configuration:
- **Linear memory** (num_memory_layers=1): `dim` rows (e.g., 512 for dim=512)
- **Deep memory** (num_memory_layers≥2): `memory_hidden_dim` rows (e.g., 2048 for dim=512, memory_hidden_mult=4.0)

Computational cost: O(T * num_rows * head_dim) per head. For T=512, num_rows=2048, head_dim=48 with 8 heads — roughly 2x the cost of one self-attention layer. Acceptable for a single insertion point.

### 2.4 TNT (HierarchicalMemory) compatibility

When `use_tnt=True`, blocks use `HierarchicalMemory` which contains a global memory and N local memories. The cross-attention reads from the **global memory's** first weight matrix:

```python
if isinstance(state, TNTMemoryState):
    W = state.global_state.weights[0]
else:
    W = state.weights[0]
```

Global memory captures long-range context and persists across the full sequence — the right target for cross-attention. Local memories are fine-grained and reset at shard boundaries, making them poor cross-attention targets.

### 2.5 MemoryCrossAttention module

```python
class MemoryCrossAttention(nn.Module):
    """Cross-attention from token representations to NeuralLTM weight rows.

    Q from the residual stream, K/V from the memory MLP's first weight matrix.
    Gated output — gate initialized near-zero so MCA has no effect until
    the gate learns to open.
    """

    def __init__(self, config: TitansConfig):
        dim = config.dim
        self.num_heads = config.mca_num_heads
        self.head_dim = dim // self.num_heads

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

        # Scalar gate — sigmoid(-3) ~ 0.047
        gate_out = 1 if config.mca_gate_type == "scalar" else dim
        self.Wg = nn.Linear(dim, gate_out, bias=True)
        self.Wg.bias = mx.full(self.Wg.bias.shape, config.mca_gate_bias_init)

        self.norm = RMSNorm(dim)

    def __call__(
        self, x: mx.array, memory_weights: mx.array
    ) -> mx.array:
        """Cross-attend from x to memory weight rows.

        Args:
            x: Token representations [B, T, dim]
            memory_weights: First weight matrix from NeuralLTM
                [num_rows, dim] where num_rows is dim (linear) or
                memory_hidden_dim (deep)

        Returns:
            Gated output [B, T, dim] — net contribution for residual add
        """
        B, T, dim = x.shape
        num_rows = memory_weights.shape[0]

        Q = self.Wq(self.norm(x))  # [B, T, dim]
        K = self.Wk(memory_weights)  # [num_rows, dim]
        V = self.Wv(memory_weights)  # [num_rows, dim]

        # Reshape for multi-head attention
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, num_rows, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(1, num_rows, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Broadcast K, V across batch
        K = mx.broadcast_to(K, (B, self.num_heads, num_rows, self.head_dim))
        V = mx.broadcast_to(V, (B, self.num_heads, num_rows, self.head_dim))

        # Scaled dot-product attention (no causal mask — all rows visible)
        scale = self.head_dim ** -0.5
        attn_scores = Q @ K.transpose(0, 1, 3, 2) * scale  # [B, heads, T, num_rows]
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_out = attn_weights @ V  # [B, heads, T, head_dim]

        # Reshape and project
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, dim)
        attn_out = self.Wo(attn_out)

        # Gated output
        gate = mx.sigmoid(self.Wg(x))  # [B, T, 1] or [B, T, dim]
        return gate * attn_out
```

---

## 3. Configuration

New fields added to `TitansConfig`:

```python
# Memory Cross-Attention (MCA)
use_mca: bool = False
mca_insertion_layers: list[int] | None = None    # None = auto [num_layers // 2]
mca_num_heads: int = 8                           # can differ from self-attn heads
mca_gate_type: str = "scalar"                    # "scalar" | "vector"
mca_gate_bias_init: float = -3.0

# Memory dump
mca_auto_dump: bool = False
mca_dump_trigger: str = "session_end"            # "session_end" | "every_n" | "surprise_threshold"
mca_dump_every_n: int = 1000
mca_dump_surprise_threshold: float = 0.85
mca_dump_path: str = "./memory_dumps/"
mca_dump_keep_last_n: int = 10
```

Computed property:

```python
@property
def mca_active_insertion_layers(self) -> list[int]:
    """Resolved MCA insertion layers."""
    if not self.use_mca:
        return []
    if self.mca_insertion_layers is not None:
        return self.mca_insertion_layers
    return [self.num_layers // 2]
```

Validation:

```python
def __post_init__(self):
    if self.use_mca:
        for idx in self.mca_active_insertion_layers:
            if idx >= self.num_layers:
                raise ValueError(
                    f"MCA insertion layer {idx} >= num_layers {self.num_layers}"
                )
```

Updated `attnres_sub_layer_block_size`:

```python
@property
def attnres_sub_layer_block_size(self) -> int:
    num_mca_layers = len(self.mca_active_insertion_layers)
    total_sub_layers = (self.num_layers * 2) + num_mca_layers
    return max(1, total_sub_layers // self.num_attnres_blocks)
```

All new fields added to `to_dict()` / `from_dict()`.

Note: Config uses `mca_` prefix (Memory Cross-Attention), not `dmca_`, to reflect the revised architecture that reads from existing memory rather than a dedicated bank.

---

## 4. Block Integration

### 4.1 Block construction — layer_idx parameter

All block constructors (`MACBlock`, `MAGBlock`, `MALBlock`) gain `layer_idx: int`. Model classes pass it:

```python
self.blocks = [MACBlock(config, layer_idx=i) for i in range(config.num_layers)]
```

### 4.2 Conditional MCA init

Added to each block's `__init__` (identical across MAC, MAG, MAL):

```python
self.has_mca = layer_idx in config.mca_active_insertion_layers
if self.has_mca:
    from titans_mlx.mca import MemoryCrossAttention
    self.mca = MemoryCrossAttention(config)
    if config.use_attn_res:
        from titans_mlx.attn_res import BlockAttnRes
        self.attn_res_mca = BlockAttnRes(config.dim)
```

### 4.3 mca_forward method

Added to each block type:

```python
def mca_forward(self, h: mx.array, mem_state) -> mx.array:
    """MCA sub-layer: cross-attend to NeuralLTM weight rows.

    Only called on blocks where self.has_mca is True.

    Args:
        h: Current hidden state [B, T, dim]
        mem_state: MemoryState or TNTMemoryState from core_forward

    Returns:
        Net contribution [B, T, dim] for residual addition
    """
    # Extract first weight matrix from appropriate state type
    if hasattr(mem_state, "global_state"):
        # TNTMemoryState — read from global memory
        W = mem_state.global_state.weights[0]
    else:
        # MemoryState — read directly
        W = mem_state.weights[0]

    # Stop gradient: memory weights are updated by their own surprise
    # mechanism, not by backprop through MCA. Without this, gradients
    # from the MCA loss could interfere with the memory update computation.
    W = mx.stop_gradient(W)

    return self.mca(h, W)
```

Note: `mca_forward` reads from the **just-updated** memory state (output of `core_forward`), so cross-attention sees the most current associations.

### 4.4 process_chunk modifications

No signature change for state threading — MCA reads from existing `MemoryState`, doesn't produce new state.

**Standard residual path:**

```python
if not config.use_attn_res:
    x = chunk
    for i, block in enumerate(blocks):
        core_out, new_state = block.core_forward(x, state=states[i])
        x = x + core_out

        # MCA sub-layer (only at insertion layers)
        if block.has_mca:
            mca_out = block.mca_forward(x, new_state)
            x = x + mca_out

        ffn_out = block.ffn_forward(x)
        x = x + ffn_out
        new_states.append(new_state)
    return x, new_states
```

**AttnRes path:**

```python
for i, block in enumerate(blocks):
    # --- Core sub-layer ---
    h, attn_weights = block.attn_res_core(completed_blocks, partial_block)
    memory_gate = block.attn_res_gate(attn_weights) if not warmup else None
    core_out, new_state = block.core_forward(
        h, state=states[i], memory_gate=memory_gate
    )
    new_states.append(new_state)

    if partial_block is None:
        partial_block = core_out
    else:
        partial_block = partial_block + core_out
    sub_idx += 1
    if sub_idx % S == 0:
        completed_blocks.append(partial_block)
        partial_block = None

    # --- MCA sub-layer (only at insertion layers) ---
    if block.has_mca:
        h_mca, _ = block.attn_res_mca(completed_blocks, partial_block)
        mca_out = block.mca_forward(h_mca, new_state)

        if partial_block is None:
            partial_block = mca_out
        else:
            partial_block = partial_block + mca_out
        sub_idx += 1
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
    if sub_idx % S == 0 or i == len(blocks) - 1:
        completed_blocks.append(partial_block)
        partial_block = None

    chunk = h + ffn_out
```

### 4.5 Model-level changes

Minimal. Each model class (`TitansMAC`, `TitansMAG`, `TitansMAL`) updates:
1. Block construction: pass `layer_idx=i`
2. Optionally: auto-dump trigger in `__call__` after forward pass

`TitansLMM` unchanged — no MCA.

The `states` list type and threading are **unchanged** — MCA reads from states, doesn't add to them.

---

## 5. Memory State Persistence

### 5.1 Format

npz + sidecar JSON metadata (reuses existing `save_memory_states`/`load_memory_states` infrastructure):

```
memory_dumps/
  dump_20260330_143000/
    state.npz             # weight and momentum tensors
    metadata.json         # config, statistics, description
```

**state.npz keys** (per layer):
- `layer_{idx}_weight_{j}` — weight matrix j for layer idx
- `layer_{idx}_momentum_{j}` — momentum matrix j for layer idx

For TNT states, additionally:
- `global_weight_{j}`, `global_momentum_{j}`
- `local_{i}_weight_{j}`, `local_{i}_momentum_{j}`
- `local_{i}_init_{j}` — W_init snapshots
- `qk_projection_{i}` — Q-K projection matrices
- `local_step_counter_{i}`

**metadata.json:**
```json
{
  "version": "1.0",
  "model_dim": 768,
  "num_layers": 16,
  "num_memory_layers": 2,
  "memory_hidden_dim": 3072,
  "use_tnt": false,
  "created_at": "2026-03-30T14:30:00Z",
  "step_count": 4500,
  "description": "after processing legal corpus batch 3",
  "per_layer_stats": {
    "0": {
      "weight_norm": 12.4,
      "momentum_norm": 0.03,
      "weight_sparsity": 0.12
    }
  }
}
```

### 5.2 MemoryDumpManager

```python
class MemoryDumpManager:
    """Serialization and inspection for NeuralLTM memory state."""

    def __init__(self, config: TitansConfig):
        self.config = config
        self.dump_path = Path(config.mca_dump_path)

    def dump(
        self,
        states: list[MemoryState | TNTMemoryState],
        step_count: int = 0,
        description: str | None = None,
    ) -> Path:
        """Serialize all layer memory states to disk.

        Creates a timestamped directory with state.safetensors + metadata.json.
        Prunes old dumps if > keep_last_n.
        Returns path to dump directory.
        """

    def load(
        self, path: str | Path, strict: bool = True
    ) -> list[MemoryState | TNTMemoryState]:
        """Restore memory states from dump.

        strict=True: fails on dimension/layer count mismatch.
        strict=False: attempts projection (zero-pad or truncate weight matrices,
            linear projection if dims differ). Warns on mismatch.
        """

    def inspect(self, path: str | Path) -> dict:
        """Human-readable summary of memory state.

        Per-layer: weight norms, momentum norms, sparsity, condition number.
        For TNT: global vs local breakdown, shard positions.
        """

    def diff(self, path_a: str | Path, path_b: str | Path) -> dict:
        """Weight-level diff between two dumps.

        Per-layer: Frobenius distance, cosine similarity, momentum delta.
        Useful for understanding what the model learned between sessions.
        """

    def merge(
        self,
        paths: list[str | Path],
        strategy: str = "weighted_mean",
    ) -> list[MemoryState | TNTMemoryState]:
        """Combine multiple dumps.

        Strategies:
          - weighted_mean: weight by step_count (more steps = more authority)
          - max_norm: per weight matrix, take from dump with largest Frobenius norm
          - recency: take most recent dump's state entirely
        """

    def reset(
        self,
        states: list[MemoryState | TNTMemoryState],
        layers: list[int] | None = None,
    ) -> list[MemoryState | TNTMemoryState]:
        """Reset memory weights and momentum to initial values.

        If layers given, partial reset (only specified layers).
        Otherwise resets all.
        """

    def fork(
        self,
        states: list[MemoryState | TNTMemoryState],
        description: str | None = None,
    ) -> Path:
        """Snapshot current state without altering live state.

        Returns dump path. Enables branching memory for different tasks:
        e.g., fork before processing a specialized corpus, so you can
        revert if the domain shift is too aggressive.
        """
```

### 5.3 Auto-dump triggers

Wired into model `_step_count`:

```python
# In model __call__, after forward pass:
if self.config.mca_auto_dump and hasattr(self, '_dump_manager'):
    trigger = self.config.mca_dump_trigger
    if trigger == "every_n" and self._step_count % self.config.mca_dump_every_n == 0:
        self._dump_manager.dump(new_states, self._step_count)
    # session_end is called explicitly by the training/inference loop
    # surprise_threshold checked via momentum norms in states
```

### 5.4 Quantization compatibility

The existing `quantize_state.py` already handles `MemoryState`. No changes needed — quantized states serialize the same way via safetensors. The dump format stores whatever dtype the tensors happen to be (float32, float16, or quantized uint8 + scale/zero_point).

---

## 6. Compatibility Matrix

### Memory Cross-Attention (MCA)

| Feature Combination | Supported | Notes |
|---------------------|-----------|-------|
| MAC + MCA | Yes | MCA sub-layer between core and FFN |
| MAG + MCA | Yes | Same insertion point |
| MAL + MCA | Yes | Same insertion point |
| LMM + MCA | No | LMM stays pure memory; no attention mechanism |
| MCA + TNT | Yes | Cross-attends to global memory weight rows |
| MCA + AttnRes | Yes | MCA becomes third sub-layer in AttnRes |
| MCA + TNT + AttnRes | Yes | All three compose independently |
| MCA + quantized state | Yes | Existing quantization handles it |
| MCA multi-layer | Yes | Multiple insertion layers, each reads own block's memory |

### State Persistence (MemoryDumpManager)

| Feature Combination | Supported | Notes |
|---------------------|-----------|-------|
| MAC + persistence | Yes | Serializes per-layer MemoryState |
| MAG + persistence | Yes | Same format |
| MAL + persistence | Yes | Same format |
| LMM + persistence | Yes | LMM has MemoryState too; dump/load/fork/merge all work |
| TNT + persistence | Yes | Serializes TNTMemoryState (global + local + Q-K projections) |
| Quantized + persistence | Yes | Quantized tensors serialize via safetensors |
| AttnRes + persistence | Yes | AttnRes has no persistent state; memory states unaffected |
| Persistence + MCA | Yes | MCA reads from the restored state — no extra state needed |

---

## 7. File Layout

New files:
- `src/titans_mlx/mca.py` — MemoryCrossAttention module
- `src/titans_mlx/memory_dump.py` — MemoryDumpManager
- `tests/test_mca.py` — Unit tests for MCA module
- `tests/test_memory_dump.py` — Serialization tests
- `tests/test_mca_integration.py` — Full model composition tests

Modified files:
- `src/titans_mlx/config.py` — MCA config fields, computed properties, validation
- `src/titans_mlx/models.py` — Block constructors (layer_idx), conditional MCA init, mca_forward, process_chunk
- `tests/conftest.py` — MCA test fixtures

---

## 8. Testing Strategy

### 8.1 Unit tests (`test_mca.py`)

```
test_cross_attention_shapes                — output [B, T, dim] for various num_rows
test_cross_attention_gate_init             — gate bias=-3.0 produces near-zero output
test_cross_attention_zero_weights          — zero weight matrix produces near-zero gated output
test_cross_attention_gradient_flow         — gradients through Wq/Wk/Wv/Wg, stopped at memory weights
test_cross_attention_linear_memory         — works with [dim, dim] weight matrix
test_cross_attention_deep_memory           — works with [memory_hidden_dim, dim] weight matrix
test_cross_attention_multi_head            — different num_heads produces same-shape output
test_cross_attention_scalar_vs_vector_gate — both gate types produce correct shapes
```

### 8.2 Serialization tests (`test_memory_dump.py`)

```
test_dump_roundtrip_memory_state           — dump then load produces identical MemoryState
test_dump_roundtrip_tnt_state              — dump then load produces identical TNTMemoryState
test_dump_strict_mismatch                  — mismatched dim raises with strict=True
test_dump_loose_projection                 — dim mismatch with strict=False projects
test_inspect_reports_norms                 — correct weight/momentum norms per layer
test_diff_identical                        — diff of same dump shows zero distance
test_diff_after_update                     — diff detects weight changes
test_merge_weighted_mean                   — merged state is weighted average by step_count
test_reset_full                            — all weights/momentum zeroed
test_reset_partial                         — only specified layers reset
test_fork_no_mutation                      — fork doesn't alter live state
test_dump_prunes_old                       — keep_last_n enforced
test_dump_quantized_state                  — quantized states serialize and restore
```

### 8.3 Integration tests (`test_mca_integration.py`)

```
test_mac_with_mca_forward                  — TitansMAC(use_mca=True) produces valid logits
test_mag_with_mca_forward                  — same for MAG
test_mal_with_mca_forward                  — same for MAL
test_lmm_ignores_mca                       — TitansLMM with use_mca=True runs unchanged

test_mca_reads_updated_state               — MCA sees weights AFTER core_forward update
test_mca_state_evolves_across_chunks       — memory state changes affect MCA output across chunks
test_mca_with_tnt                          — use_mca=True + use_tnt=True: reads global memory weights
test_mca_with_attn_res                     — use_mca=True + use_attn_res=True: MCA participates as sub-layer
test_mca_with_tnt_and_attn_res             — all three features enabled
test_mca_multi_layer                       — insertion_layers=[1,3,5]: each reads own block's memory

test_mca_no_regression_without             — use_mca=False produces identical output to baseline
test_attnres_sublayer_count                — sub-layer block size accounts for MCA layers
test_mca_with_quantized_state              — quantize_memory_state=True works with MCA

test_dump_load_resume                      — dump states, load into fresh model, outputs match
test_fork_and_diverge                      — fork, process different data, states diverge
```

### 8.4 Test fixtures

```python
@pytest.fixture
def mca_config() -> TitansConfig:
    return TitansConfig(
        dim=64, num_heads=4, num_layers=6, vocab_size=256,
        use_mca=True, mca_num_heads=4,
    )

@pytest.fixture
def mca_tnt_config() -> TitansConfig:
    return TitansConfig(
        dim=64, num_heads=4, num_layers=6, vocab_size=256,
        use_mca=True, mca_num_heads=4,
        use_tnt=True, global_chunk_size=32, local_chunk_sizes=[4, 8],
    )

@pytest.fixture
def mca_multi_config() -> TitansConfig:
    return TitansConfig(
        dim=64, num_heads=4, num_layers=6, vocab_size=256,
        use_mca=True, mca_insertion_layers=[1, 3, 5], mca_num_heads=4,
    )
```

### 8.5 Not tested at this stage

- Training convergence with MCA (requires real data, long runs)
- Optimal insertion layer position (requires ablation studies)
- Merge strategy quality (requires semantic evaluation)
- Cross-attention vs MLP retrieval complementarity (requires probing experiments)

---

## 9. Open Questions (Deferred)

| Question | Priority | Notes |
|----------|----------|-------|
| Which weight matrix layer to read from | Medium | First layer is default; deeper layers might capture different patterns |
| Multi-matrix cross-attention | Low | Attend to concatenation of all weight matrices — more info, more compute |
| Learned gating per-head | Low | Different heads attend to different memory directions — might specialize |
| Inference-time memory warm-up | Medium | Processing a "context document" to prime memory before task — workflow, not architecture |
| Dump compression | Low | safetensors is already efficient; gzip on top if needed |
