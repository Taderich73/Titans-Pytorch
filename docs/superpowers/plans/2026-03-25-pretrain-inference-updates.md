# pretrain.py & inference.py Post-SFT/LoRA Updates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `scripts/pretrain.py` to save `chat_template` metadata, and update `scripts/inference.py` to support LoRA adapter loading and chat template formatting for SFT/LoRA fine-tuned models.

**Architecture:** pretrain.py gets a one-line metadata addition. inference.py gets three features: (1) `chat_template` field extraction from metadata + auto-detect chat mode, (2) LoRA adapter loading via copied LoRA utilities, (3) ChatML formatting in interactive and prompt modes. All LoRA utilities are copied from `scripts/lora.py` to keep inference.py self-contained.

**Tech Stack:** MLX, safetensors, JSON

**Spec:** `docs/superpowers/specs/2026-03-25-pretrain-inference-updates-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `scripts/pretrain.py` | Add `chat_template: "none"` to checkpoint metadata |
| Modify | `scripts/inference.py` | LoRA loading, chat template support, CLI flags |
| Create | `tests/test_inference_updates.py` | Tests for chat formatting + LoRA loading in inference |
| Modify | `README.md` | Update inference section with chat + LoRA examples |

---

### Task 1: pretrain.py — Add chat_template to metadata

**Files:**
- Modify: `scripts/pretrain.py:690-717`

- [ ] **Step 1: Add `chat_template` to metadata dict**

In `scripts/pretrain.py`, in the `save_checkpoint()` function, add `"chat_template": "none"` to the metadata dict (after line 716, before the closing `}`):

```python
        "tokenizer_name": config.tokenizer,
        "chat_template": "none",
    }
```

- [ ] **Step 2: Verify pretrain.py still runs**

Run: `uv run python scripts/pretrain.py --help`
Expected: Shows help, no errors

- [ ] **Step 3: Commit**

```bash
git add scripts/pretrain.py
git commit -m "feat(pretrain): add chat_template field to checkpoint metadata"
```

---

### Task 2: inference.py — Chat template support

**Files:**
- Modify: `scripts/inference.py`
- Create: `tests/test_inference_updates.py`

- [ ] **Step 1: Write tests for chat formatting in inference**

Create `tests/test_inference_updates.py`:

```python
"""Tests for inference.py chat template and LoRA adapter support."""

from __future__ import annotations


class TestFormatChatMLInference:
    """Tests for ChatML formatting in inference context."""

    def test_format_prompt_as_chat(self):
        """Wrap a raw prompt as a ChatML user message with assistant prefix."""
        from scripts.inference import format_prompt_for_chat

        result = format_prompt_for_chat("Hello")
        expected = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert result == expected

    def test_format_prompt_preserves_content(self):
        """Multi-line prompts are preserved."""
        from scripts.inference import format_prompt_for_chat

        result = format_prompt_for_chat("Line 1\nLine 2")
        assert "Line 1\nLine 2" in result
        assert result.startswith("<|im_start|>user\n")
        assert result.endswith("<|im_start|>assistant\n")

    def test_strip_chat_delimiters(self):
        """Strip ChatML delimiters from generated output."""
        from scripts.inference import strip_chat_delimiters

        text = "Hello there<|im_end|>\n<|im_start|>user"
        result = strip_chat_delimiters(text)
        assert "<|im_end|>" not in result
        assert "<|im_start|>" not in result

    def test_should_use_chat_auto_detect(self):
        """Auto-detect chat mode from chat_template metadata."""
        from scripts.inference import should_use_chat

        # SFT model with chatml
        assert should_use_chat("chatml", None) is True
        assert should_use_chat("auto", None) is True
        # Pretrain model
        assert should_use_chat("none", None) is False
        assert should_use_chat(None, None) is False
        # CLI overrides
        assert should_use_chat("chatml", True) is True
        assert should_use_chat("chatml", False) is False
        assert should_use_chat("none", True) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_inference_updates.py -v 2>&1 | head -20`
Expected: FAIL (ImportError — functions don't exist yet)

- [ ] **Step 3: Add chat formatting functions to inference.py**

Add these functions to `scripts/inference.py` (after the tokenizer section, before model loading):

```python
# =============================================================================
# Chat Template Support
# =============================================================================

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_prompt_for_chat(prompt: str) -> str:
    """Wrap a raw prompt as a ChatML user message with assistant generation prefix.

    Returns a string ready to be tokenized and fed to the model:
        <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    """
    return (
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def strip_chat_delimiters(text: str) -> str:
    """Remove ChatML special tokens from generated text for display."""
    return text.replace(IM_START, "").replace(IM_END, "").strip()


def should_use_chat(
    chat_template: str | None,
    cli_override: bool | None,
) -> bool:
    """Determine whether to use chat formatting.

    Args:
        chat_template: Value from checkpoint metadata ("chatml", "auto", "none", or None).
        cli_override: True (--chat), False (--no-chat), or None (auto-detect).

    Returns:
        Whether to apply ChatML formatting.
    """
    if cli_override is not None:
        return cli_override
    if chat_template is None or chat_template == "none":
        return False
    return True


def ensure_chat_tokens(tokenizer) -> None:
    """Add ChatML special tokens to the tokenizer if not already present."""
    if not HAS_TRANSFORMERS or isinstance(tokenizer, SimpleTokenizer):
        return
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    to_add = []
    if IM_START not in existing:
        to_add.append(IM_START)
    if IM_END not in existing:
        to_add.append(IM_END)
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})


def get_im_end_token_id(tokenizer) -> int | None:
    """Resolve the <|im_end|> token ID, or None if not in vocabulary."""
    try:
        ids = tokenizer.encode(IM_END, add_special_tokens=False)
        if ids:
            return ids[0]
    except Exception:
        pass
    return None
```

- [ ] **Step 4: Update `load_model()` to extract `chat_template` from metadata**

In `load_model()`, after extracting `tokenizer_name` (line 159-161), add:

```python
        chat_template = str(meta.get("chat_template", ["none"])[0])
        if chat_template == "None":
            chat_template = "none"
```

In the `else` branch (no metadata file, line 194), add:

```python
        chat_template = "none"
```

Update the return type annotation:

```python
def load_model(
    checkpoint_path: Path,
    quantize: int | None = None,
) -> tuple[nn.Module, TitansConfig, str, str | None, str]:
```

Update the return statement (line 269):

```python
    return model, config, model_type, tokenizer_name, chat_template
```

- [ ] **Step 5: Add `--chat` / `--no-chat` CLI flags**

In `main()` argparse section (after the memory persistence arguments, before `args = parser.parse_args()`):

```python
    # Chat template arguments
    parser.add_argument(
        "--chat",
        action="store_true",
        default=None,
        help="Force chat template formatting (auto-detected from checkpoint)",
    )
    parser.add_argument(
        "--no-chat",
        dest="chat",
        action="store_false",
        help="Disable chat template formatting",
    )
```

- [ ] **Step 6: Update `main()` model loading call site**

Change line 608:

```python
    model, config, model_type, saved_tokenizer, chat_template = load_model(
        Path(args.checkpoint), args.quantize
    )
```

After tokenizer loading (line 614), add:

```python
    # Determine chat mode
    use_chat = should_use_chat(chat_template, args.chat)
    if use_chat:
        ensure_chat_tokens(tokenizer)
        im_end_id = get_im_end_token_id(tokenizer)
        logger.info(f"Chat mode enabled (template: {chat_template})")
    else:
        im_end_id = None
```

- [ ] **Step 7: Update interactive mode to use chat formatting**

In the interactive mode section, wrap the prompt when `use_chat` is True. In the streaming branch (around line 683), before `generate_streaming()`, add prompt wrapping:

```python
                if use_chat:
                    formatted_prompt = format_prompt_for_chat(prompt)
                else:
                    formatted_prompt = prompt
```

Use `formatted_prompt` instead of `prompt` for tokenization and generation. For the EOS token, pass `im_end_id` alongside the tokenizer's `eos_token_id`:

```python
                    eos_token_id=im_end_id or getattr(tokenizer, "eos_token_id", None),
```

In the output display, strip delimiters when in chat mode:

```python
                    if use_chat:
                        generated_text = strip_chat_delimiters(generated_text)
```

Apply the same pattern to the non-streaming interactive branch and the single-shot `--prompt` mode.

- [ ] **Step 8: Run tests**

Run: `uv run python -m pytest tests/test_inference_updates.py -v`
Expected: All PASS

- [ ] **Step 9: Smoke test**

Run: `uv run python scripts/inference.py --help`
Expected: Shows `--chat` and `--no-chat` flags, no errors

- [ ] **Step 10: Commit**

```bash
git add scripts/inference.py tests/test_inference_updates.py
git commit -m "feat(inference): add chat template support with auto-detection"
```

---

### Task 3: inference.py — LoRA adapter loading

**Files:**
- Modify: `scripts/inference.py`
- Modify: `tests/test_inference_updates.py`

- [ ] **Step 1: Add LoRA adapter loading test**

Append to `tests/test_inference_updates.py`:

```python
import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC


class TestLoRAInference:
    """Tests for LoRA adapter loading in inference."""

    def test_load_lora_model(self):
        """Load a model with LoRA adapters from saved files."""
        from scripts.lora import wrap_lora_layers, save_adapters, _find_lora_modules
        from scripts.inference import load_lora_model

        # Create and save a base model + adapters
        config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, num_persistent_tokens=2, num_memory_layers=1,
            use_conv=False, use_rope=False, dropout=0.0,
        )
        base_model = TitansMAC(config)
        mx.eval(base_model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save base checkpoint
            base_path = Path(tmpdir) / "base"
            base_model.save_weights(str(base_path.with_suffix(".safetensors")))
            meta = {
                "model_type": "mac", "dim": 32, "num_heads": 2,
                "num_layers": 1, "vocab_size": 50, "chunk_size": 16,
                "window_size": 16, "num_persistent_tokens": 2,
                "num_memory_layers": 2,
            }
            np.savez(
                str(base_path.with_suffix(".meta.npz")),
                **{k: np.array([v]) for k, v in meta.items()},
            )

            # Wrap and save adapters
            wrap_lora_layers(base_model, "attn", rank=4, alpha=8.0, dropout=0.0)
            for _, mod in _find_lora_modules(base_model):
                mod.lora_B = mx.ones_like(mod.lora_B) * 0.1
            mx.eval(base_model.parameters())

            adapter_path = Path(tmpdir) / "adapters"
            save_adapters(base_model, adapter_path, meta={
                "rank": 4, "alpha": 8.0, "dropout": 0.0,
                "lora_targets": "attn",
                "base_checkpoint": str(base_path),
                "model_type": "mac", "dim": 32, "num_heads": 2,
                "num_layers": 1, "vocab_size": 50, "chunk_size": 16,
                "window_size": 16, "num_persistent_tokens": 2,
                "num_memory_layers": 2,
                "chat_template": "chatml",
                "tokenizer": "gpt2",
            })

            # Load via inference function
            model, cfg, mtype, tok, chat_tmpl = load_lora_model(
                adapter_path, checkpoint_override=str(base_path),
            )
            mx.eval(model.parameters())

            assert mtype == "mac"
            assert chat_tmpl == "chatml"

            # Verify model runs
            x = mx.random.randint(0, 50, (1, 16))
            logits, _ = model(x)
            mx.eval(logits)
            assert logits.shape == (1, 16, 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_inference_updates.py::TestLoRAInference -v`
Expected: FAIL (ImportError — `load_lora_model` doesn't exist)

- [ ] **Step 3: Copy LoRA utilities into inference.py**

Copy these from `scripts/lora.py` into `scripts/inference.py` (after the chat template section, before model loading):

- `LoRALinear` class (lora.py lines 84-114)
- `_recursive_find_linear()` (lora.py lines 121-150)
- `_ATTN_NAMES`, `_FFN_NAMES` constants (lora.py lines 158-159)
- `wrap_lora_layers()` (lora.py lines 162-226)
- `_find_lora_modules()` (lora.py lines 233-264)
- `load_adapters()` (lora.py lines 291-315)

Add `import json` to the imports at the top of inference.py.

- [ ] **Step 4: Implement `load_lora_model()`**

Add to `scripts/inference.py`:

```python
def load_lora_model(
    adapters_path: Path,
    checkpoint_override: str | None = None,
    quantize: int | None = None,
) -> tuple[nn.Module, TitansConfig, str, str | None, str]:
    """Load a model with LoRA adapters applied.

    Reads adapter metadata to reconstruct model config, loads base
    checkpoint, wraps target layers, then loads adapter weights.

    Args:
        adapters_path: Path to adapters (stem, without extension).
        checkpoint_override: Override base checkpoint path from metadata.
        quantize: Quantization bits (None, 4, or 8).

    Returns:
        Tuple of (model, config, model_type, tokenizer_name, chat_template).
    """
    adapters_path = Path(adapters_path)
    meta_path = adapters_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())

    # Extract config from adapter metadata
    model_type = meta.get("model_type", "mac")
    tokenizer_name = meta.get("tokenizer", None)
    chat_template = meta.get("chat_template", "none")

    config = TitansConfig(
        dim=int(meta.get("dim", 512)),
        num_heads=int(meta.get("num_heads", 8)),
        num_layers=int(meta.get("num_layers", 12)),
        vocab_size=int(meta.get("vocab_size", 32000)),
        chunk_size=int(meta.get("chunk_size", 512)),
        window_size=int(meta.get("window_size", 512)),
        num_persistent_tokens=int(meta.get("num_persistent_tokens", 16)),
        num_memory_layers=int(meta.get("num_memory_layers", 2)),
        dropout=0.0,
        use_conv=False,
        use_tnt=bool(meta.get("use_tnt", False)),
        global_chunk_size=int(meta.get("global_chunk_size", 2048)),
        local_chunk_sizes=meta.get("local_chunk_sizes", [8, 16]),
        local_shard_length=int(meta.get("local_shard_length", 2048)),
        tnt_stage=int(meta.get("tnt_stage", 1)),
        use_attn_res=bool(meta.get("use_attn_res", False)),
        num_attnres_blocks=int(meta.get("num_attnres_blocks", 8)),
        attnres_warmup_steps=int(meta.get("attnres_warmup_steps", 0)),
    )

    model = create_model(model_type, config)

    # Load base weights
    base_path = Path(checkpoint_override or meta.get("base_checkpoint", ""))
    weights_path = base_path.with_suffix(".safetensors")
    if weights_path.exists():
        model.load_weights(str(weights_path))
    elif base_path.suffix == ".safetensors" and base_path.exists():
        model.load_weights(str(base_path))
    else:
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_path}. "
            "Use --checkpoint to specify the base model path."
        )

    # Wrap with LoRA and load adapter weights
    rank = int(meta.get("rank", 8))
    alpha = float(meta.get("alpha", 16.0))
    targets = meta.get("lora_targets", "attn")
    wrap_lora_layers(model, targets, rank, alpha, dropout=0.0)
    load_adapters(model, adapters_path)

    # Quantize if requested
    if quantize:
        model = quantize_model(model, quantize)

    mx.eval(model.parameters())

    flags = ["LoRA"]
    if config.use_tnt:
        flags.append("TNT")
    if config.use_attn_res:
        flags.append("AttnRes")
    logger.info(
        f"Loaded {model_type.upper()} [{'+'.join(flags)}] from {adapters_path}"
    )

    return model, config, model_type, tokenizer_name, chat_template
```

- [ ] **Step 5: Add `--adapters` CLI flag and update main()**

Add to argparse (after `--checkpoint`):

```python
    parser.add_argument(
        "--adapters",
        type=str,
        default=None,
        help="Path to LoRA adapters (loads adapter metadata, wraps model)",
    )
```

Make `--checkpoint` not required (since `--adapters` can provide the base path from metadata):

```python
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
```

Update model loading in `main()` to branch on `--adapters`:

```python
    # Load model
    if args.adapters:
        model, config, model_type, saved_tokenizer, chat_template = load_lora_model(
            Path(args.adapters),
            checkpoint_override=args.checkpoint,
            quantize=args.quantize,
        )
    elif args.checkpoint:
        model, config, model_type, saved_tokenizer, chat_template = load_model(
            Path(args.checkpoint), args.quantize
        )
    else:
        parser.error("Either --checkpoint or --adapters is required")
```

- [ ] **Step 6: Run all tests**

Run: `uv run python -m pytest tests/test_inference_updates.py -v`
Expected: All PASS

- [ ] **Step 7: Smoke test**

Run: `uv run python scripts/inference.py --help`
Expected: Shows `--adapters`, `--chat`, `--no-chat` flags

- [ ] **Step 8: Commit**

```bash
git add scripts/inference.py tests/test_inference_updates.py
git commit -m "feat(inference): add LoRA adapter loading support"
```

---

### Task 4: README.md — Update inference section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update inference section**

In `README.md`, update the Inference section to include chat mode and LoRA adapter examples:

```markdown
## Inference

```bash
# Generate from pretrained model
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --prompt "Once upon a time" \
    --max-tokens 100

# Interactive mode
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --interactive

# Chat mode with SFT model (auto-detected from checkpoint)
uv run python scripts/inference.py \
    --checkpoint checkpoints/sft_model.safetensors \
    --interactive --stream

# Force chat mode on/off
uv run python scripts/inference.py \
    --checkpoint checkpoints/model.safetensors \
    --interactive --chat
uv run python scripts/inference.py \
    --checkpoint checkpoints/sft_model.safetensors \
    --interactive --no-chat

# Inference with LoRA adapters
uv run python scripts/inference.py \
    --adapters checkpoints/final_adapters \
    --interactive --stream

# LoRA with different base checkpoint
uv run python scripts/inference.py \
    --adapters checkpoints/final_adapters \
    --checkpoint checkpoints/different_base.safetensors \
    --interactive
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update inference section with chat mode and LoRA examples"
```

---

### Task 5: Final verification

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests PASS, no regressions

- [ ] **Step 2: Run linter**

Run: `uv run ruff check scripts/inference.py scripts/pretrain.py tests/test_inference_updates.py`
Expected: No errors (or fix any)

- [ ] **Step 3: Verify all three scripts show help**

Run: `uv run python scripts/pretrain.py --help && uv run python scripts/inference.py --help`
Expected: Both work, inference shows new flags

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: lint fixes for inference and pretrain updates"
```
