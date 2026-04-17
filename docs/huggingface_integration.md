# HuggingFace Integration

> **Paper alignment:** N/A — packaging and serialization layer.
>
> **Implementation status:** Not paper-derived. `TitansMACConfig` serializes exactly the hyperparameters discussed in the `docs/configuration_guide.md` "Paper Origin Tags" section — some of which are paper-faithful, some deliberate deviations, and some novel extensions. Saving or loading a config therefore preserves whatever alignment stance the configured model has; it does not add or remove deviations by itself.
>
> **Details:** When you push a model to the Hub, all flags listed under "Paper Origin Tags" are round-tripped. A "Faithful" flag stays Faithful; a "Novel" flag stays Novel. Consumers who want to know whether a published checkpoint deviates from the papers should inspect the config against the Paper Origin Tags table.

## Overview

Titans v0.5.0 adds full HuggingFace transformers compatibility for the MAC architecture. This enables standard HF workflows: loading models via `from_pretrained()`, generating text via HF pipelines, training with the HF `Trainer`, and sharing models on the Hub.

The integration lives in `src/titans/hf/` as an optional subpackage. The base `titans` package has no dependency on `transformers` -- only `titans.hf` requires it.

**Current scope:** TitansMAC only. The pattern is designed for easy extension to MAG, MAL, and LMM variants.

## Installation

```bash
pip install titans[hf]
```

This installs `transformers>=5.0.0`, `safetensors>=0.4.0`, `huggingface_hub>=0.20.0`, and all training dependencies.

## Loading & Saving Models

### Direct Import (Recommended)

No `trust_remote_code` needed:

```python
from titans.hf import TitansMACForCausalLM

# Load from Hub
model = TitansMACForCausalLM.from_pretrained("your-org/titans-mac-1.5B")

# Save locally
model.save_pretrained("./my-model")
```

### Auto-Registration

If you `import titans.hf` first, `AutoModel` works:

```python
import titans.hf  # registers model type
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-org/titans-mac-1.5B")
```

### From Native Config

Build a model from scratch using HF conventions:

```python
from titans.hf import TitansMACConfig, TitansMACForCausalLM

config = TitansMACConfig(
    dim=1024, num_heads=16, num_layers=20, vocab_size=50257,
    chunk_size=512, use_tnt=True, use_attn_res=True,
)
model = TitansMACForCausalLM(config)
```

## Generation

`TitansMACForCausalLM` provides a custom `generate()` that handles Titans' chunked memory architecture. The prompt is processed in `chunk_size` chunks during prefill, and generated tokens are buffered with memory updates committed when the buffer fills.

```python
import torch
from titans.hf import TitansMACForCausalLM

model = TitansMACForCausalLM.from_pretrained("your-org/titans-mac-1.5B")
model.eval()

input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")

# Sampling
output = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=50)

# Greedy
output = model.generate(input_ids, max_new_tokens=200, do_sample=False)

# With pre-loaded memory state
output = model.generate(input_ids, max_new_tokens=200, memory_states=states)
```

The `generate()` method accepts `temperature`, `top_k`, `top_p`, `do_sample`, `memory_states`, and `max_new_tokens`.

## Training with HF Trainer

### TitansTrainer

`TitansTrainer` subclasses HF's `Trainer` with per-chunk truncated BPTT. It overrides `compute_loss()` to split sequences into `chunk_size` pieces, process each chunk with memory state carry, and detach states at chunk boundaries.

```python
from titans.hf import TitansMACConfig, TitansMACForCausalLM, TitansTrainer
from transformers import TrainingArguments

config = TitansMACConfig(dim=512, num_heads=8, num_layers=12, chunk_size=512)
model = TitansMACForCausalLM(config)

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    max_steps=10000,
    report_to="wandb",
)

trainer = TitansTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    reset_memory_per_batch=True,        # reset memory after each batch (default)
    state_carry_warmup_steps=500,       # if reset=False, still reset for first 500 steps
)

trainer.train()
```

### Memory State Lifecycle

| Parameter | Default | Behavior |
|-----------|---------|----------|
| `reset_memory_per_batch` | `True` | Memory states reset to `None` after each batch. Each batch starts fresh. |
| `state_carry_warmup_steps` | `500` | When `reset_memory_per_batch=False`, reset memory for the first N steps before allowing carry. |

Memory cost is ~32MB per block (bf16, batch=2) for the 1.5B config. 20 blocks = ~640MB GPU memory held between steps. This matches the cost of the native training loop.

### TitansChunkMixin

`TitansChunkMixin` extracts the per-chunk BPTT logic into a reusable mixin. Use it to add Titans memory support to any Trainer subclass:

```python
from titans.hf import TitansChunkMixin
from trl import SFTTrainer, DPOTrainer

class TitansSFTTrainer(TitansChunkMixin, SFTTrainer):
    def __init__(self, *args, reset_memory_per_batch=True,
                 state_carry_warmup_steps=500, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_titans_memory(reset_memory_per_batch, state_carry_warmup_steps)

class TitansDPOTrainer(TitansChunkMixin, DPOTrainer):
    def __init__(self, *args, reset_memory_per_batch=True,
                 state_carry_warmup_steps=500, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_titans_memory(reset_memory_per_batch, state_carry_warmup_steps)
```

## Converting Existing Checkpoints

The `convert_to_hf.py` script converts native `.pt` or `.safetensors` checkpoints to HF-compatible model directories.

```bash
# Basic conversion
python scripts/convert_to_hf.py \
    --checkpoint checkpoints/final.pt \
    --output-dir ./hf_model

# With tokenizer and chat template
python scripts/convert_to_hf.py \
    --checkpoint checkpoints/final.pt \
    --tokenizer gpt2 \
    --add-chat-template \
    --output-dir ./hf_model

# Push directly to Hub
python scripts/convert_to_hf.py \
    --checkpoint checkpoints/final.pt \
    --tokenizer gpt2 \
    --output-dir ./hf_model \
    --push-to-hub your-org/titans-mac-1.5B

# Explicit model type (for future variants)
python scripts/convert_to_hf.py \
    --checkpoint checkpoints/final.pt \
    --model-type mac \
    --output-dir ./hf_model
```

### What the Converter Produces

| File | Contents |
|------|----------|
| `config.json` | `TitansMACConfig` with all model parameters, `model_type`, `architectures`, `auto_map` |
| `model.safetensors` | Model weights with `model.` prefix (HF convention) |
| `generation_config.json` | Default generation parameters (temperature=0.8, top_k=50) |
| `tokenizer.json` (optional) | Tokenizer files from `--tokenizer` |
| `tokenizer_config.json` (optional) | Includes ChatML template if `--add-chat-template` is used |

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to native `.pt` or `.safetensors` checkpoint |
| `--output-dir` | required | Directory for HF model files |
| `--model-type` | `mac` | Model variant (`mac`; `mag`/`mal`/`lmm` planned) |
| `--tokenizer` | `None` | HF tokenizer name (e.g. `gpt2`). Omit to skip. |
| `--add-chat-template` | `False` | Add ChatML special tokens and template |
| `--torch-dtype` | `float32` | Dtype metadata in config |
| `--push-to-hub` | `None` | Hub repo ID to push to |
| `--upload-model-code` | `False` | Upload Python source for `trust_remote_code` |

## Chat Template

When `--add-chat-template` is used, the converter adds ChatML tokens (`<|im_start|>`, `<|im_end|>`) to the tokenizer and sets a Jinja chat template:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

This expands the GPT-2 vocab from 50257 to 50259 (two new special tokens). Only use this for chat-finetuned models, not base pretrained models.

## Configuration Mapping

`TitansMACConfig` is a bidirectional bridge between HF's `PretrainedConfig` and the native `TitansConfig` dataclass. All 58 `TitansConfig` fields are stored as top-level attributes in `config.json`.

```python
from titans.config import TitansConfig
from titans.hf import TitansMACConfig

# Native -> HF
titans_config = TitansConfig(dim=1024, num_heads=16, num_layers=20)
hf_config = TitansMACConfig.from_titans_config(titans_config)

# HF -> Native
restored = hf_config.to_titans_config()
assert titans_config.to_dict() == restored.to_dict()  # lossless roundtrip
```

## Key Classes

| Import | Description |
|--------|-------------|
| `TitansMACConfig` | `PretrainedConfig` subclass mapping all TitansConfig fields |
| `TitansMACForCausalLM` | `PreTrainedModel` wrapper with forward, generate, save/load |
| `TitansTrainer` | `Trainer` subclass with per-chunk truncated BPTT |
| `TitansChunkMixin` | Reusable mixin for adding chunk BPTT to any Trainer subclass |

All classes are imported from `titans.hf`:

```python
from titans.hf import (
    TitansMACConfig,
    TitansMACForCausalLM,
    TitansTrainer,
    TitansChunkMixin,
)
```

## Variant Extensibility

The HF integration is designed so adding MAG, MAL, or LMM variants later is a copy-and-adapt operation:

1. Add `TitansMAGConfig` with `model_type = "titans-mag"` to `configuration.py`
2. Add `TitansMAGForCausalLM` wrapping `TitansMAG` to `modeling.py`
3. Add one entry to `_VARIANT_REGISTRY` in `__init__.py`
4. Add `--model-type mag` case to `convert_to_hf.py`

The `TitansTrainer`, `TitansChunkMixin`, `generate()` logic, chat template, and conversion pipeline are all variant-agnostic and require no changes.
