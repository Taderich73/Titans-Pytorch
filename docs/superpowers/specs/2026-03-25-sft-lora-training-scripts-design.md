# SFT and LoRA Training Scripts — Design Spec

**Date**: 2026-03-25
**Status**: Reviewed

## Overview

Two new self-contained training scripts for supervised fine-tuning of Titans MLX models:

- **`scripts/sft.py`** — Full-parameter supervised fine-tuning
- **`scripts/lora.py`** — LoRA (low-rank adapter) fine-tuning

Both scripts are self-contained (no shared `training_utils.py`). They import only from `titans_mlx` and copy the ~100 lines of training loop boilerplate (gradient accumulation, LR schedule, checkpointing) directly. This avoids coupling and lets each script evolve independently.

## Titans Architecture Considerations

Titans is not a standard transformer. Key differences that affect SFT/LoRA design:

- **Memory states**: Models return `(logits, states)`. The neural long-term memory updates its weights at test time. For SFT/LoRA, memory states are **discarded per-example** (not carried across conversation turns), matching pretrain.py's approach. Carrying memory across turns within a multi-turn conversation is a future consideration.
- **Non-standard layer names**: Attention projections are `proj_q`, `proj_k`, `proj_v`, `proj_out` (not the common `q_proj`/`k_proj`/`v_proj`/`o_proj`). LoRA targeting must use the correct names.
- **Unique layer types**: Memory MLPs, persistent memory, Q-K projections, and AttnRes cross-attention exist alongside standard attention/FFN. LoRA presets must be explicit about what they include.
- **TNT stage 2**: The config supports `tnt_stage=2` for fine-tuning with halved local chunk sizes. SFT/LoRA scripts should support `--tnt-stage 2` for stage 2 fine-tuning.
- **Weight tying**: `head.weight = embed.weight` is tied. Must be re-tied after each optimizer step (as pretrain.py does). LoRA must never wrap embed or head layers to avoid breaking the tie.

## Data Pipeline

### Source

HuggingFace datasets in `messages` format, streamed via `datasets` library. No full dataset download.

Primary target: [allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) — 2.15M multi-turn instruction examples across 11 domains.

Expected schema per row:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### Streaming with Shuffle Buffer

Stream from HuggingFace with a shuffle buffer of 1000 examples. Draw batches from the buffer to avoid training artifacts from dataset ordering (Dolci is multi-source and may be sorted by domain).

```python
ds = load_dataset(name, split="train", streaming=True)
ds = ds.shuffle(seed=seed, buffer_size=1000)
```

### Chat Template Formatting

1. If the tokenizer has a **non-None** `chat_template` attribute (`getattr(tokenizer, 'chat_template', None) is not None`), use `tokenizer.apply_chat_template(messages, tokenize=True)`
2. Otherwise, fall back to ChatML:

```
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
{content}<|im_end|>
```

For ChatML fallback, add `<|im_start|>`, `<|im_end|>` as special tokens to the tokenizer if not already present.

### Loss Masking

**Default: assistant content only.** Mask out:
- All user turn tokens
- All system turn tokens
- Role/delimiter tokens (e.g., `<|im_start|>assistant\n`) — the model gets these for free at inference time

Only the assistant's actual content tokens and the trailing `<|im_end|>` contribute to the loss. The EOS/end token is included so the model learns when to stop.

**Optional `--train-on-all` flag** disables masking and trains on the full conversation.

Implementation: build a `loss_mask` array (same shape as labels) during tokenization. The masked loss function reshapes logits to `(B*T, V)` and labels/mask to `(B*T,)` (matching pretrain.py's pattern), then uses per-token weighting:

```python
logits_flat = logits.reshape(-1, vocab_size)
labels_flat = labels.reshape(-1)
mask_flat = loss_mask.reshape(-1)
per_token_loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction="none")
masked_loss = (per_token_loss * mask_flat).sum() / mask_flat.sum().clip(min=1)
```

Note: `model(input_ids)` returns `(logits, states)` — states are discarded in the loss function, matching pretrain.py's approach.

## SFT Script (`scripts/sft.py`)

### Architecture

Self-contained script. Copies training utilities from pretrain.py:
- `create_model`, `count_parameters`
- `get_lr_schedule` (cosine with warmup)
- `compute_grads` (modified for masked loss)
- `apply_gradients`, `sanitize_and_clip_grads`
- `save_checkpoint`, `load_checkpoint`
- Gradient accumulation: `_tree_add`, `_tree_scale`, `_eval_grads`

### Training Flow

1. Load pretrained checkpoint via `--init-weights` (weights only, fresh step/epoch/schedule)
2. Stream dataset, format with chat template, build loss masks
3. Full parameter training with gradient accumulation
4. Cosine LR schedule with warmup

### Default Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LR | 2e-5 | Standard SFT LR for full fine-tuning |
| Weight decay | 0.1 | Same as pretraining |
| Warmup ratio | 0.03 | Same as pretraining |
| Grad clip | 1.0 | Same as pretraining |
| Batch size | 4 | Per-device |
| Gradient accumulation | 8 | Effective batch 32 |
| Seq len | 2048 | Shorter than pretrain (4096) — most instruction data is shorter |

### Checkpoint Format

Same as pretrain.py: `model.safetensors` + `model.meta.npz`. Adds `chat_template` field to metadata (value: `"chatml"` or the tokenizer's template name).

### CLI

```bash
uv run python scripts/sft.py \
    --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer gpt2 \
    --lr 2e-5 \
    --seq-len 2048 \
    --epochs 1
```

Key flags:
- `--init-weights PATH` — pretrained checkpoint (required for meaningful SFT)
- `--dataset NAME` — HuggingFace dataset name
- `--dataset-subset NAME` — dataset config/subset
- `--tokenizer NAME` — HuggingFace tokenizer
- `--train-on-all` — disable loss masking, train on full conversation
- `--messages-field NAME` — field name for messages array (default: `"messages"`)
- All pretrain.py flags: `--batch-size`, `--lr`, `--epochs`, `--max-steps`, `--seq-len`, `--grad-clip`, `--warmup-ratio`, `--gradient-accumulation-steps`, `--save-every`, `--eval-every`, `--checkpoint-dir`, `--resume`, `--wandb`, `--seed`, `--dtype`
- Model architecture flags: `--dim`, `--num-heads`, `--num-layers`, `--use-tnt`, `--tnt-stage`, `--use-attn-res`, etc.

## LoRA Script (`scripts/lora.py`)

### LoRA Implementation

A `LoRALinear` module that wraps `nn.Linear`:

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base  # frozen via base.freeze()
        in_dim = base.weight.shape[1]
        out_dim = base.weight.shape[0]
        # Kaiming-style init for A, zero init for B (standard LoRA; starts as identity)
        self.lora_A = mx.random.normal((in_dim, rank)) * (1.0 / math.sqrt(rank))
        self.lora_B = mx.zeros((rank, out_dim))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x):
        base_out = self.base(x)  # frozen forward
        lora_input = self.dropout(x) if self.dropout else x
        lora_out = (lora_input @ self.lora_A) @ self.lora_B * self.scale
        return base_out + lora_out
```

- A initialized with Kaiming-style scaling (`1/sqrt(rank)`), B initialized to zero (so LoRA starts as identity)
- Base weights frozen via `base.freeze()`
- Only A and B matrices are trainable (bare `mx.array` attributes on `nn.Module` are picked up by `model.parameters()`)
- Dropout applied to LoRA path only during training

### Target Layer Configuration

CLI flag: `--lora-targets` accepting comma-separated presets.

| Preset | Layers targeted |
|--------|----------------|
| `attn` (default) | Q, K, V, O projections in attention |
| `ffn` | gate, up, down projections in FFN |
| `memory` | Memory MLP layers (neural long-term memory) |
| `attn,ffn` | Both attention and FFN |
| `all` | Every `nn.Linear` except embed and head (to preserve weight tying) |

Layer matching by name pattern (using actual Titans layer names):
- `attn`: matches `*.proj_q`, `*.proj_k`, `*.proj_v`, `*.proj_out`
- `ffn`: matches `*.gate_proj`, `*.up_proj`, `*.down_proj`
- `memory`: matches `*.memory.*.layers.*` (memory MLP linear layers)
- `all`: matches any `nn.Linear` except `embed.*` and `head.*`

**Excluded from all presets**: `embed` and `head` layers are never wrapped — they share tied weights and wrapping either would break the tie.

### Default Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank | 8 | Sufficient for 512-dim model; rank 16 is overkill at this scale |
| Alpha | 16 | alpha = 2 * rank is a common starting point |
| Dropout | 0.05 | Light regularization |
| LR | 1e-4 | LoRA adapters start near zero, need stronger signal than full SFT |
| Weight decay | 0.01 | Lower than full SFT — fewer params, less need for regularization |

### Parameter Freezing

1. Load base model from `--init-weights`
2. Freeze all base model parameters
3. Replace target `nn.Linear` layers with `LoRALinear` wrappers
4. Only LoRA A/B matrices are trainable
5. Log trainable vs total parameter count

### Adapter Saving

Adapters saved as a separate small file:

- **`adapters.safetensors`** — only the LoRA A and B matrices, keyed by their full parameter path (e.g., `blocks.0.attention.proj_q.lora_A`)
- **`adapters.meta.json`** — JSON metadata (deliberately JSON rather than `.meta.npz` for human readability and easier tooling integration):

```json
{
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "lora_targets": "attn",
    "base_checkpoint": "checkpoints/best_model",
    "chat_template": "chatml",
    "model_type": "mac",
    "dim": 512,
    "num_heads": 8,
    "num_layers": 12,
    "vocab_size": 32000,
    "use_tnt": false,
    "tnt_stage": 1,
    "use_attn_res": false,
    "tokenizer": "gpt2",
    "dtype": "float16"
}
```

### Adapter Loading (utility function)

A `load_lora_model()` function that:
1. Reads `adapters.meta.json` to get base model config
2. Creates and loads the base model
3. Wraps target layers with `LoRALinear`
4. Loads adapter weights from `adapters.safetensors`

Also a `merge_lora_weights()` function that folds A @ B * scale into the base weight permanently, producing a standard model checkpoint for deployment without any LoRA overhead.

### CLI

```bash
uv run python scripts/lora.py \
    --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer gpt2 \
    --lora-targets attn \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lr 1e-4
```

Key flags (in addition to all SFT flags):
- `--lora-rank INT` — LoRA rank (default: 8)
- `--lora-alpha FLOAT` — LoRA scaling alpha (default: 16)
- `--lora-dropout FLOAT` — dropout on LoRA path (default: 0.05)
- `--lora-targets STR` — comma-separated target presets (default: `"attn"`)
- `--merge-and-save PATH` — after training, merge adapters into base and save full model
- `--dtype STR` — compute dtype for LoRA A/B matrices (default: `"float16"`); must match base model dtype

## What's NOT in Scope

- Changes to `inference.py` for chat template support (separate task)
- Changes to `pretrain.py` (no shared utilities extraction)
- Evaluation benchmarks beyond validation loss/perplexity
- Multi-GPU / distributed training
- RLHF / DPO / preference tuning
