# Memory Cross-Attention (MCA)

> **Paper alignment:** N/A — novel extension.
>
> **Implementation status:** Novel extension. No reference paper defines this component.
>
> **Details:** MCA adds cross-attention from the residual stream to the neural memory's weight-matrix rows, giving the model a second read interface in addition to the paper's MLP retrieval. The Titans paper only prescribes MLP-based retrieval; this is a project-specific augmentation. It is orthogonal to MAC/MAG/MAL and can be enabled via `use_mca=True`.

## Overview

MCA adds cross-attention from token representations to the NeuralLongTermMemory's weight matrix rows. This gives the model a second read interface into the same memory that is already being written to by the surprise-driven update mechanism.

| | MLP Retrieval (existing) | Cross-Attention (MCA) |
|---|---|---|
| Operation | `output = MLP(query)` | `softmax(Q @ K^T) @ V` |
| Nature | Nonlinear function of query | Linear blend of memory directions |
| What it captures | Precise key-value lookup | Soft discovery of relevant associations |

## How It Works

1. **Query** comes from the residual stream (token hidden states), projected through `Wq`.
2. **Key** and **Value** come from the first weight matrix of the NeuralLongTermMemory MLP, projected through `Wk` and `Wv` respectively. The memory weights are detached (no gradient flows back into memory through MCA).
3. Standard multi-head scaled dot-product attention is applied (no causal mask -- all memory rows are visible to all positions).
4. Output is projected through `Wo` and multiplied by a **sigmoid gate** initialized near-zero (`bias = -3.0`, so `sigmoid(-3) ~ 0.05`).

The near-zero gate initialization ensures MCA has essentially no effect at training start. The gate learns to open as training reveals that cross-attention adds value.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_mca` | `False` | Enable MCA |
| `mca_insertion_layers` | `None` | Layers where MCA is inserted (None = auto midpoint) |
| `mca_num_heads` | 8 | Cross-attention heads |
| `mca_gate_type` | `"scalar"` | Gate type: `"scalar"` or `"vector"` |
| `mca_gate_bias_init` | -3.0 | Gate bias initialization |

When `mca_insertion_layers` is `None`, MCA is inserted at `[num_layers // 2]` (the midpoint layer). You can specify multiple layers for more insertion points.

## Usage

```python
import torch
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    use_mca=True,
    mca_num_heads=8,
    # mca_insertion_layers=[6],  # auto = [num_layers // 2]
)
model = TitansMAC(config)
```

Works with MAC, MAG, and MAL block types.

## AttnRes Integration

When both `use_mca=True` and `use_attn_res=True`, MCA becomes a third sub-layer in the AttnRes framework. The AttnRes pseudo-query at MCA sub-layers attends over the same block history as the other sub-layers, maintaining consistent depth-wise attention.

## Key Class

- `MemoryCrossAttention` -- the cross-attention module with gated output
