# pretrain.py and inference.py Post-SFT/LoRA Updates — Design Spec

**Date**: 2026-03-25
**Status**: Approved

## Overview

Update `scripts/pretrain.py` and `scripts/inference.py` to support models produced by the new SFT and LoRA training scripts.

## pretrain.py

Single change: add `"chat_template": "none"` to checkpoint metadata in `save_checkpoint()`. Ensures inference.py always has a `chat_template` field regardless of which script produced the checkpoint.

## inference.py

### 1. Chat Template Support

- Read `chat_template` field from `.meta.npz` in `load_model()`, fall back to `"none"` for old checkpoints
- Add `format_chatml(messages) -> str` (same implementation as sft.py)
- `load_model()` return signature changes from 4-tuple to 5-tuple: `(model, config, model_type, tokenizer_name, chat_template)`
- Add `--chat` / `--no-chat` CLI flags to override auto-detection
- Auto-detection: if `chat_template != "none"`, chat mode is on by default

**Interactive mode with chat enabled:**
- Wrap user input as `[{"role": "user", "content": input}]`
- Format with ChatML: `<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n`
- Feed formatted text to tokenizer, generate until `<|im_end|>` token
- Strip ChatML delimiters from displayed output
- Add `<|im_start|>` and `<|im_end|>` as special tokens if tokenizer doesn't have them

**Single-shot `--prompt` mode with chat enabled:**
- Same wrapping logic — treat `--prompt` as a user message

**EOS handling:**
- When chat mode is active, resolve `<|im_end|>` token ID and include it as an additional EOS token for generation stopping

### 2. LoRA Adapter Loading

- Add `--adapters PATH` CLI flag
- New `load_lora_model(adapters_path, checkpoint_override, quantize)` function:
  1. Read `adapters.meta.json` for model config + LoRA params
  2. Create model from metadata config fields (model_type, dim, num_heads, etc.)
  3. Load base weights from `base_checkpoint` in metadata (overridable with `--checkpoint`)
  4. Call `wrap_lora_layers(model, targets, rank, alpha, dropout=0.0)` — dropout=0.0 at inference
  5. Call `load_adapters(model, adapters_path)`
  6. Return `(model, config, model_type, tokenizer_name, chat_template)`
- When `--adapters` is provided, use `load_lora_model()` instead of `load_model()`
- Copy these from `scripts/lora.py` into `scripts/inference.py` (self-contained):
  - `LoRALinear` class
  - `_recursive_find_linear()`
  - `wrap_lora_layers()`
  - `_find_lora_modules()`
  - `load_adapters()`

### 3. Updated load_model() Signature

Current: `load_model(checkpoint_path, quantize) -> (model, config, model_type, tokenizer_name)`
New: `load_model(checkpoint_path, quantize) -> (model, config, model_type, tokenizer_name, chat_template)`

All call sites updated accordingly.

## What's NOT Changing

- No shared utils extraction
- No changes to sft.py or lora.py
- No changes to model architecture or TitansConfig
- pretrain.py training logic untouched
- inference.py generation logic (`generate()`, `sample_top_p()`, etc.) untouched
