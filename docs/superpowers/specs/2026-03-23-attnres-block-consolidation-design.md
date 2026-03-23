# AttnRes Integration & Block Consolidation Design

**Date:** 2026-03-23
**Status:** Draft
**Scope:** Rearchitect AttnRes as a composable residual-replacement mechanism across all block types (MAC, MAG, MAL), consolidate TNT block variants into config-driven memory selection, and align the implementation with the Attention Residuals paper (arXiv 2603.15031).

## Problem

Enabling `--use-attn-res` in `pretrain.py` silently switches the model from `TitansMAC` (single `NeuralLongTermMemory`) to `TitansTNT` (hierarchical memory: 1 global + 2 local memories + QK projections). The AttnRes computation itself adds <1% overhead, but the accidental architecture switch causes an ~11x training slowdown.

Additionally, the current AttnRes implementation diverges from the paper in several ways:
- Applied once per block instead of twice per layer (before attention and before MLP)
- Standard residual connections are kept alongside AttnRes rather than replaced by it
- The token embedding is baked into the first block's partial sum instead of kept as a standalone source (b_0)
- AttnRes only works with `TitansTNT`, not with the base MAC/MAG/MAL models

Separately, the codebase has 6 near-duplicate block classes: `MACBlock`/`TNTMACBlock`, `MAGBlock`/`TNTMAGBlock`, `MALBlock`/`TNTMALBlock`. The TNT variants duplicate block logic, swapping `NeuralLongTermMemory` for `HierarchicalMemory`.

## Design

### 1. Sub-layer decomposition

Each block exposes two sub-layer methods instead of a monolithic `__call__`:

- **`core_forward(h, state, memory_gate=None) → (core_out, new_state)`**
  Attention + memory update + gating. Returns the net contribution (no between-sub-layer residual added).

- **`ffn_forward(h) → ffn_out`**
  Feed-forward network. Returns the net contribution (no between-sub-layer residual added).

Blocks no longer own between-sub-layer residual connections. The model-level orchestrator decides whether to apply standard residuals or AttnRes between sub-layers.

**Internal residuals within a sub-layer are preserved.** The core sub-layer uses `y_t = h + attn_out` (or equivalent) internally for its memory update and gating logic. This is critical: the memory must see the full representation (AttnRes-weighted context + attention contribution), not just the raw attention output. The `core_out` returned to the orchestrator is the net new information: `attn_out + gated` (excluding the input `h`). The internal `y_t = h + attn_out` is a computation detail inside the sub-layer, not a between-sub-layer residual.

Concrete example for MACBlock.core_forward:
```python
def core_forward(self, h, state, memory_gate=None):
    batch_size = h.shape[0]
    # Retrieve from memory
    query = mx.broadcast_to(self.memory_query, (batch_size, 1, self.config.dim))
    memory_retrieved = self.memory.retrieve(query, state)
    memory_tokens = self.norm_mem(memory_retrieved)
    persistent = self.persistent(batch_size)
    # Attention
    normed = self.norm1(h)
    attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
    # Internal residual: memory and gating see h + attn_out
    y_t = h + attn_out
    # Memory update
    mem_out, new_state = self.memory(y_t, state=state, memory_gate=memory_gate)
    # Gating
    gated = mx.sigmoid(self.gate_norm_attn(y_t)) * mx.sigmoid(self.gate_norm_mem(mem_out))
    # Return net contribution (excludes h)
    core_out = attn_out + gated
    return core_out, new_state
```

Sub-layer mapping per block type:

| Block | core_forward | ffn_forward |
|-------|-------------|-------------|
| **MAC** | retrieve → attention → y_t=h+attn_out → memory_update(y_t) → gating(y_t, mem_out) → return attn_out+gated | norm → FFN |
| **MAG** | attention → y_t=h+attn_out → memory_update(norm(h)) → gating(y_t, mem_out) → return attn_out+gated | norm → FFN |
| **MAL** | memory(norm(h)) → h_mid=h+mem_out → attention(norm(h_mid)) → return mem_out+attn_out | norm → FFN |

This matches the paper's 2-per-layer pattern: two AttnRes injection points per block.

### 2. AttnRes replaces residuals

The model-level orchestrator handles both paths:

**Standard residuals (AttnRes disabled):**
```
for each block:
    core_out, state = block.core_forward(x, state)
    x = x + core_out
    ffn_out = block.ffn_forward(x)
    x = x + ffn_out
```

**AttnRes (enabled):**
```
completed_blocks = [embedding]   # b_0 as standalone source
partial_block = None
sub_idx = 0
S = attnres_sub_layer_block_size  # sub-layers per AttnRes block

for each block:
    # Core sub-layer
    h, weights = block.attn_res_core(completed_blocks, partial_block)
    gate = block.attn_res_gate(weights)  # optional memory gate
    core_out, state = block.core_forward(h, state, memory_gate=gate)
    partial_block = accumulate(partial_block, core_out)
    sub_idx += 1
    if sub_idx % S == 0:
        completed_blocks.append(partial_block)
        partial_block = None

    # FFN sub-layer
    h, weights = block.attn_res_ffn(completed_blocks, partial_block)
    ffn_out = block.ffn_forward(h)
    partial_block = accumulate(partial_block, ffn_out)
    sub_idx += 1
    if sub_idx % S == 0 or last_block:
        completed_blocks.append(partial_block)
        partial_block = None

    chunk = h + ffn_out  # model output tracks last hidden state
```

Key corrections from current implementation:
- **Embedding as standalone b_0:** `completed_blocks` initialized with the token embedding, not baked into `partial_block`.
- **No subtraction hack:** Blocks return pure sub-layer outputs. No `output - h` needed.
- **Two AttnRes calls per block:** Each block has `attn_res_core` and `attn_res_ffn`, each with its own pseudo-query and RMSNorm, matching the paper's `attn_res_proj`/`mlp_res_proj`. This doubles per-block AttnRes modules from 1 to 2 (12 → 24 for a 12-layer model). Each adds only `dim + dim` parameters (RMSNorm weight + Linear(dim,1) weight), so the total increase is negligible.
- **Block boundaries at sub-layer granularity:** With 2 sub-layers per block × N blocks = 2N total slots. `S = 2N / num_attnres_blocks`.

**Model output:** After the final sub-layer, the orchestrator computes `chunk = h + ffn_out` — the last AttnRes-weighted input plus the last FFN contribution. This single exit-point residual ensures the output head sees a full hidden state rather than just the last sub-layer's contribution. In the standard residual path, `x` naturally accumulates all contributions, so no special handling is needed.

### 3. Block consolidation

Three block classes instead of six. Memory type is config-driven:

```python
class MACBlock:
    def __init__(self, config):
        if config.use_tnt:
            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)

        if config.use_attn_res:
            self.attn_res_core = BlockAttnRes(config.dim)
            self.attn_res_ffn = BlockAttnRes(config.dim)
            self.attn_res_gate = AttnResMemoryGate()
```

`TNTMACBlock`, `TNTMAGBlock`, `TNTMALBlock`, and `TitansTNT` are removed.

Flag combinations and their effect:

| Flags | Block memory | Residual handling |
|-------|-------------|-------------------|
| (neither) | NeuralLongTermMemory | Standard residuals |
| `--use-tnt` | HierarchicalMemory | Standard residuals |
| `--use-attn-res` | NeuralLongTermMemory | AttnRes |
| `--use-attn-res --use-tnt` | HierarchicalMemory | AttnRes |

**`LMMBlock` / `TitansLMM` are out of scope.** LMM is a standalone memory-only model without attention or FFN — the AttnRes sub-layer decomposition does not apply. LMM remains unchanged.

### 4. Memory interface alignment

`NeuralLongTermMemory` and `HierarchicalMemory` have different call signatures today:

- `NeuralLongTermMemory.__call__(x, state, return_state, lr_scale)` — uses `lr_scale` directly
- `HierarchicalMemory.__call__(x, state, memory_gate)` — translates `memory_gate` to `lr_scale` via config flags

To make blocks truly memory-agnostic, align on a single interface. Add a `memory_gate` parameter to `NeuralLongTermMemory.__call__` that performs the same translation as `HierarchicalMemory`:

```python
# NeuralLongTermMemory.__call__ gains memory_gate parameter:
def __call__(self, x, state=None, return_state=True, lr_scale=1.0, memory_gate=None):
    if memory_gate is not None:
        lr_scale = memory_gate  # direct pass-through for single memory
    # ... rest unchanged
```

Blocks call `self.memory(x, state=state, memory_gate=gate)` regardless of memory type. The `lr_scale` parameter remains for direct programmatic use but blocks don't need it.

The `return_state` parameter on `NeuralLongTermMemory` defaults to `True` and is not used by blocks. No change needed.

### 5. Shared orchestration

The chunk-processing loop is a standalone function used by all model classes:

```python
def process_chunk(blocks, chunk, states, config, step_count):
    """Process a single chunk through all blocks.

    Handles both standard residuals and AttnRes paths.
    Used by TitansMAC, TitansMAG, TitansMAL.
    """
```

Each model class (`TitansMAC`, `TitansMAG`, `TitansMAL`) calls this from its `__call__` method instead of implementing its own loop. This eliminates duplication of the AttnRes orchestration logic.

**Step counter for AttnRes warmup:** `step_count` is passed as a parameter to `process_chunk`. Each model class owns a `_step_count` attribute (integer, incremented per `__call__`) and passes it in. During warmup (`step_count < config.attnres_warmup_steps`), the memory gate is bypassed (set to None).

### 6. Memory gate

The memory gate mechanism is preserved as an optional feature. The core sub-layer's AttnRes attention weights feed the gate:

1. `attn_res_core` returns attention weights alongside `h`
2. `AttnResMemoryGate` extracts importance (weight on most recent source)
3. Gate is passed to `core_forward` as `memory_gate`
4. Inside `core_forward`, the memory module receives `memory_gate`

**New for MAG and MAL:** Currently only MACBlock (via TNTMACBlock) supports the memory gate. This design adds gate support to all three block types. The plumbing is the same: `core_forward` passes `memory_gate` through to `self.memory(...)`. When AttnRes is disabled, `memory_gate` is None and the memory ignores it.

During AttnRes warmup (`step < attnres_warmup_steps`), the gate is bypassed (set to None).

## File changes

| File | Action |
|------|--------|
| `models.py` | Refactor MAC/MAG/MAL blocks: config-driven memory, `core_forward`/`ffn_forward` split, conditional AttnRes modules, memory gate support on all blocks. Model classes use shared `process_chunk`. Add `_step_count` to each model class. |
| `tnt_models.py` | **Delete.** All TNT block variants and `TitansTNT` absorbed into `models.py`. |
| `tnt_memory.py` | Unchanged. `HierarchicalMemory`, `GlobalMemory`, `LocalMemory` remain. |
| `memory.py` | Add `memory_gate` parameter to `NeuralLongTermMemory.__call__` for interface alignment. |
| `attn_res.py` | Unchanged. `BlockAttnRes` and `AttnResMemoryGate` are already correct. Each block instantiates two `BlockAttnRes` (core + ffn) instead of one — this is a usage change, not a class change. |
| `config.py` | Replace `attnres_base_block_size` with `attnres_sub_layer_block_size` returning `(num_layers * 2) // num_attnres_blocks`. Remove the old property. |
| `pretrain.py` | Simplify `create_model` — just `models[model_type](config)`. Remove `TitansTNT` import and TNT-specific routing. `use_attn_res` and `use_tnt` are independent flags. |
| `__init__.py` | Update exports: remove `TitansTNT` and TNT block classes, ensure `HierarchicalMemory` is still exported. |
| `tests/test_tnt.py` | Update: replace `TNTMACBlock`/`TNTMAGBlock`/`TNTMALBlock`/`TitansTNT` imports with `MACBlock`/`MAGBlock`/`MALBlock` + `use_tnt=True` config. |
| `examples/tnt_usage.py` | Rewrite: use `TitansMAC`/`TitansMAG`/`TitansMAL` with `use_tnt=True` config instead of `TitansTNT(variant=...)`. |

## State type compatibility

`HierarchicalMemory` uses `TNTMemoryState` while `NeuralLongTermMemory` uses `MemoryState`. The orchestrator and model code treat states as opaque objects — they're passed through without inspection. The block's memory module owns its state type. No adapter needed; the type annotation on `core_forward` uses a union: `MemoryState | TNTMemoryState | None`.

## Checkpoint compatibility

**This is a breaking change for TNT checkpoints.** The TNT block variants use `self.hierarchical_memory` as the attribute name; the consolidated blocks use `self.memory`. Saved checkpoint keys like `blocks.0.hierarchical_memory.global_memory.*` will not match the new `blocks.0.memory.global_memory.*` paths.

Mitigation: add a key remapping function in `load_checkpoint` that translates `hierarchical_memory` → `memory` in weight keys. This is a one-time migration — new checkpoints use the consolidated names.

Non-TNT checkpoints (using `NeuralLongTermMemory`) already use `self.memory` and are unaffected.

## Config changes

Replace `attnres_base_block_size` (currently `num_layers // num_attnres_blocks`) with `attnres_sub_layer_block_size` returning `(num_layers * 2) // num_attnres_blocks`. The old property is removed — no code outside the orchestrator uses it.

## What does NOT change

- `BlockAttnRes` class — already matches the paper's `block_attn_res` function
- `AttnResMemoryGate` class — implementation-specific, orthogonal to paper alignment
- `HierarchicalMemory` / `GlobalMemory` / `LocalMemory` — TNT memory hierarchy stays intact
- `NeuralLongTermMemory` — gains `memory_gate` parameter but internal logic unchanged
- `QKProjection` — unchanged
- Memory update mechanics (gradient computation, parallel update, momentum) — unchanged
- `LMMBlock` / `TitansLMM` — out of scope, unchanged
