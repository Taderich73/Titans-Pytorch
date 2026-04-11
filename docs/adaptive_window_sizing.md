# Adaptive Window Sizing

## Overview

Standard sliding window attention uses a fixed window size for all layers and all content. Adaptive window sizing lets each layer **learn its own effective window size** from the input, using differentiable soft masking.

Instead of a hard boolean cutoff at position `w` (attend to the last `w` tokens), adaptive window produces a smooth sigmoid falloff. Positions within the learned falloff center attend normally; positions beyond decay smoothly to zero. This makes the window boundary differentiable, allowing gradient-based optimization of window size.

## How It Works

Each layer gets a lightweight `AdaptiveWindowPredictor`:

1. **Single linear projection** (`Linear(dim, 1)`) maps hidden states to a raw scalar per position.
2. **Sigmoid + range mapping** converts the scalar to a falloff center in `[min_window, max_window]`.
3. **Soft mask** -- for each query-key pair, compute `sigmoid(temperature * (falloff_center - distance))`. Multiply by a causal mask.
4. The soft mask replaces the hard boolean window mask in `SlidingWindowAttention`.

An **efficiency regularization** term penalizes large windows during training:

```
reg_loss = lambda * mean(falloff_centers / max_window)
```

This encourages the model to use smaller windows where local context suffices, trading off quality vs. compute.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adaptive_window` | `False` | Enable per-layer learned window sizing |
| `adaptive_window_min` | 64 | Minimum window size floor |
| `adaptive_window_max` | `None` | Maximum window size (defaults to `window_size`) |
| `adaptive_window_temperature` | 10.0 | Sigmoid sharpness -- higher = more binary mask |
| `adaptive_window_lambda` | 0.01 | Efficiency regularization weight |

## Usage

```python
import torch
from titans import TitansConfig, TitansMAG

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    window_size=512,
    adaptive_window=True,
    adaptive_window_min=64,
    adaptive_window_lambda=0.01,
)
model = TitansMAG(config)

input_ids = torch.randint(0, 32000, (2, 1024))
logits, states = model(input_ids)
```

Supported for **MAG** and **MAL** blocks (both use sliding window attention). Not applicable to MAC (which uses segmented full-causal attention).

## Training Integration

The regularization loss must be added to the training loss:

```python
from titans import compute_window_regularization

# After forward pass, collect falloff_centers from each layer
reg_loss = compute_window_regularization(falloff_centers, config.window_size)
total_loss = lm_loss + config.adaptive_window_lambda * reg_loss
```

## Key Classes

- `AdaptiveWindowPredictor` -- per-layer module producing soft masks and falloff centers
- `compute_window_regularization` -- computes the efficiency regularization term from collected falloff centers
