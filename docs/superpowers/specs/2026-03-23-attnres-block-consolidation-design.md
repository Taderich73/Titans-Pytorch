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
  Attention + memory update + gating. Returns the net contribution (no residual added).

- **`ffn_forward(h) → ffn_out`**
  Feed-forward network. Returns the net contribution (no residual added).

Blocks no longer own residual connections. The model-level orchestrator decides whether to apply standard residuals or AttnRes between sub-layers.

Internal residuals *within* a sub-layer are preserved where needed. For example, MAL's core computes `h_mid = h + mem_out` before passing to attention — this is internal to the sub-layer, not a between-sub-layer residual.

Sub-layer mapping per block type:

| Block | core_forward | ffn_forward |
|-------|-------------|-------------|
| **MAC** | retrieve → attention(persistent, memory, norm(h)) → memory_update → gating | norm → FFN |
| **MAG** | attention(persistent, norm(h)) → memory_update → gating | norm → FFN |
| **MAL** | memory(persistent, norm(h)) → attention(persistent, norm(h_mid)) | norm → FFN |

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
- **Two AttnRes calls per block:** Each block has `attn_res_core` and `attn_res_ffn`, each with its own pseudo-query and RMSNorm, matching the paper's `attn_res_proj`/`mlp_res_proj`.
- **Block boundaries at sub-layer granularity:** With 2 sub-layers per block × N blocks = 2N total slots. `S = 2N / num_attnres_blocks`.

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

Both memory types expose the same interface: `__call__(x, state, memory_gate)`, `retrieve(query, state)`, `init_state(batch_size)`. Blocks are agnostic to which they use.

`TNTMACBlock`, `TNTMAGBlock`, `TNTMALBlock`, and `TitansTNT` are removed.

Flag combinations and their effect:

| Flags | Block memory | Residual handling |
|-------|-------------|-------------------|
| (neither) | NeuralLongTermMemory | Standard residuals |
| `--use-tnt` | HierarchicalMemory | Standard residuals |
| `--use-attn-res` | NeuralLongTermMemory | AttnRes |
| `--use-attn-res --use-tnt` | HierarchicalMemory | AttnRes |

### 4. Shared orchestration

The chunk-processing loop is a standalone function used by all model classes:

```python
def process_chunk(blocks, chunk, states, config, step_count):
    """Process a single chunk through all blocks.

    Handles both standard residuals and AttnRes paths.
    Used by TitansMAC, TitansMAG, TitansMAL.
    """
```

Each model class (`TitansMAC`, `TitansMAG`, `TitansMAL`) calls this from its `__call__` method instead of implementing its own loop. This eliminates duplication of the AttnRes orchestration logic.

### 5. Memory gate

The memory gate mechanism is preserved as an optional feature. The core sub-layer's AttnRes attention weights feed the gate:

1. `attn_res_core` returns attention weights alongside `h`
2. `AttnResMemoryGate` extracts importance (weight on most recent source)
3. Gate is passed to `core_forward` as `memory_gate`
4. Inside `core_forward`, the memory module receives `memory_gate` as `lr_scale`

During AttnRes warmup (`step < attnres_warmup_steps`), the gate is bypassed (set to None).

### 6. Model output

After the final sub-layer, the model output is `h + ffn_out` — the last AttnRes-weighted input plus the last FFN contribution. This ensures the output head sees a full hidden state. The one explicit residual at the exit point matches how the paper's final layer produces its output.

## File changes

| File | Action |
|------|--------|
| `models.py` | Refactor MAC/MAG/MAL blocks: config-driven memory, `core_forward`/`ffn_forward` split, conditional AttnRes modules. Model classes use shared `process_chunk`. |
| `tnt_models.py` | **Delete.** All TNT block variants and `TitansTNT` absorbed into `models.py`. |
| `tnt_memory.py` | Unchanged. `HierarchicalMemory`, `GlobalMemory`, `LocalMemory` remain. |
| `attn_res.py` | Unchanged. `BlockAttnRes` and `AttnResMemoryGate` are already correct. |
| `config.py` | Update `attnres_base_block_size` property to account for 2 sub-layers per block. |
| `pretrain.py` | Simplify `create_model` — remove TNT-specific routing. `use_attn_res` and `use_tnt` are independent flags. |
| `__init__.py` | Update exports: remove `TitansTNT` and TNT block classes, ensure `HierarchicalMemory` is still exported. |

## State type compatibility

`HierarchicalMemory` uses `TNTMemoryState` while `NeuralLongTermMemory` uses `MemoryState`. The orchestrator and model code treat states as opaque objects — they're passed through without inspection. The block's memory module owns its state type. No adapter needed; the type annotation on `core_forward` uses a union: `MemoryState | TNTMemoryState | None`.

## Config changes

`attnres_base_block_size` currently returns `num_layers // num_attnres_blocks`. With 2 sub-layers per block, the effective block size in sub-layer units is `(num_layers * 2) // num_attnres_blocks`. A new property `attnres_sub_layer_block_size` computes this.

## What does NOT change

- `BlockAttnRes` class — already matches the paper's `block_attn_res` function
- `AttnResMemoryGate` class — implementation-specific, orthogonal to paper alignment
- `HierarchicalMemory` / `GlobalMemory` / `LocalMemory` — TNT memory hierarchy stays intact
- `NeuralLongTermMemory` — unchanged
- `QKProjection` — unchanged
- Memory update mechanics (gradient computation, parallel update, momentum) — unchanged
