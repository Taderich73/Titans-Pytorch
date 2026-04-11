# Attention Residuals (AttnRes)

> **Paper**: Kimi Team (2025). *Attention Residuals*. [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)

## Overview

Standard residual connections accumulate all prior layer outputs with uniform unit weights. AttnRes replaces this with learned depth-wise softmax attention over prior representations, allowing each layer to selectively weight which earlier layers it draws from.

```
Standard:  h_l = sum(v_i)                    (fixed unit weights)
AttnRes:   h_l = sum(alpha_{i->l} * v_i)     (learned softmax weights)
```

This mitigates **PreNorm dilution** -- the phenomenon where deep networks with pre-normalization progressively wash out useful signal from earlier layers by treating all prior contributions equally.

## How It Works

Each sub-layer has a learned **pseudo-query** vector (`Linear(dim, 1)`, zero-initialized). At each sub-layer:

1. Collect completed block representations from prior layers.
2. Apply `RMSNorm` to each as keys.
3. Compute attention logits via the pseudo-query projection.
4. Apply softmax over sources to get attention weights.
5. Weighted sum produces the input to this sub-layer.

Zero initialization ensures that at training start, all weights are uniform (equivalent to standard residual behavior). The model gradually learns which depths to attend to as training progresses.

## AttnRes Blocks

Sub-layers are grouped into **AttnRes blocks** of configurable size. Block boundaries define the granularity at which representations are tracked:

- **Two AttnRes calls per transformer block** -- one before the core (attention + memory) sub-layer, one before the FFN sub-layer.
- **Embedding as standalone source** (`b_0`) -- always available for attention.
- When MCA is enabled, it becomes a third sub-layer within the AttnRes framework.

## Memory Gating

AttnRes attention weights modulate the memory learning rate. The attention weight on the embedding source (`b_0`) indicates how much the current layer relies on raw input vs. processed representations. This weight is used to scale the memory update learning rate, providing an adaptive curriculum for memory training.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_attn_res` | `False` | Enable AttnRes |
| `num_attnres_blocks` | 8 | Number of AttnRes blocks (N) |
| `attnres_warmup_steps` | 0 | Steps before memory gating activates |
| `attnres_modulate_global_memory` | `True` | Gate global memory learning rate |
| `attnres_modulate_local_memory` | `False` | Gate local memory learning rate |
| `attnres_logit_clip` | 30.0 | Clamp attention logits for numerical stability |

## Usage

```python
import torch
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    use_attn_res=True,
    num_attnres_blocks=4,
    attnres_warmup_steps=200,
)
model = TitansMAC(config)
```

Works with any block type and composes independently with `use_tnt` and `use_mca`.

## Key Classes

- `BlockAttnRes` -- per-sub-layer module with the pseudo-query projection and softmax computation
- `AttnResMemoryGate` -- extracts the embedding attention weight and scales the memory learning rate
