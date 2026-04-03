# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for inference.py chat template support and LoRA adapter loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from scripts.inference import (
    IM_END,
    IM_START,
    _find_lora_modules,
    format_prompt_for_chat,
    load_lora_model,
    save_adapters,
    should_use_chat,
    strip_chat_delimiters,
    wrap_lora_layers,
)
from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC

# ---------------------------------------------------------------------------
# TestFormatChatMLInference
# ---------------------------------------------------------------------------


class TestFormatChatMLInference:
    """Tests for chat template formatting functions in inference.py."""

    def test_format_prompt_as_chat(self) -> None:
        """Verify wrapping produces correct ChatML structure."""
        result = format_prompt_for_chat("Hello!")
        expected = f"{IM_START}user\nHello!{IM_END}\n{IM_START}assistant\n"
        assert result == expected

    def test_format_prompt_preserves_content(self) -> None:
        """Multi-line content is preserved inside ChatML wrapper."""
        prompt = "Line 1\nLine 2\nLine 3"
        result = format_prompt_for_chat(prompt)
        assert prompt in result
        assert result.startswith(IM_START)
        assert result.endswith(f"{IM_START}assistant\n")

    def test_strip_chat_delimiters(self) -> None:
        """IM_START and IM_END tokens are removed."""
        text = f"{IM_START}assistant\nHello world{IM_END}"
        cleaned = strip_chat_delimiters(text)
        assert IM_START not in cleaned
        assert IM_END not in cleaned
        assert "Hello world" in cleaned

    def test_should_use_chat_auto_detect(self) -> None:
        """Tests all combinations of template + cli_override."""
        # cli_override takes precedence
        assert should_use_chat("chatml", True) is True
        assert should_use_chat("chatml", False) is False
        assert should_use_chat("none", True) is True
        assert should_use_chat("none", False) is False
        assert should_use_chat(None, True) is True
        assert should_use_chat(None, False) is False

        # Auto mode (cli_override=None)
        assert should_use_chat("chatml", None) is True
        assert should_use_chat("none", None) is False
        assert should_use_chat(None, None) is False
        assert should_use_chat("custom", None) is True


# ---------------------------------------------------------------------------
# TestLoRAInference
# ---------------------------------------------------------------------------


class TestLoRAInference:
    """Tests for LoRA adapter loading via inference.py."""

    def test_load_lora_model(self) -> None:
        """Create base model, save, wrap with LoRA, save adapters, load via load_lora_model."""
        mx.random.seed(42)

        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=32,
            window_size=32,
            num_persistent_tokens=4,
            num_memory_layers=1,
            dropout=0.0,
            use_conv=False,
        )

        model = TitansMAC(config)
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base_model"

            # Save base weights
            from mlx.utils import tree_flatten

            weights = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(str(base_path.with_suffix(".safetensors")), weights)

            # Save base metadata (npz format)
            meta_npz_path = base_path.with_suffix(".meta.npz")
            np.savez(
                str(meta_npz_path),
                model_type=np.array(["mac"]),
                dim=np.array([64]),
                num_heads=np.array([4]),
                num_layers=np.array([2]),
                vocab_size=np.array([256]),
                chunk_size=np.array([32]),
                window_size=np.array([32]),
                num_persistent_tokens=np.array([4]),
                num_memory_layers=np.array([1]),
                tokenizer_name=np.array(["None"]),
                chat_template=np.array(["none"]),
            )

            # Wrap with LoRA and train a tiny bit to get non-zero adapter weights
            rank = 4
            alpha = 8.0
            wrapped = wrap_lora_layers(model, "attn", rank, alpha, dropout=0.0)
            assert len(wrapped) > 0

            # Poke lora_A/B so they are non-zero
            for _, lora_mod in _find_lora_modules(model):
                lora_mod.lora_B = mx.ones_like(lora_mod.lora_B) * 0.01
            mx.eval(model.parameters())

            # Save adapters
            adapters_path = Path(tmpdir) / "adapters"
            adapter_meta = {
                "model_type": "mac",
                "dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "vocab_size": 256,
                "chunk_size": 32,
                "window_size": 32,
                "num_persistent_tokens": 4,
                "num_memory_layers": 1,
                "tokenizer_name": "None",
                "chat_template": "none",
                "base_checkpoint": str(base_path),
                "lora_targets": "attn",
                "lora_rank": rank,
                "lora_alpha": alpha,
                "use_tnt": False,
                "use_attn_res": False,
            }
            save_adapters(model, adapters_path, adapter_meta)

            # Now load via load_lora_model
            loaded_model, loaded_config, loaded_type, loaded_tok, loaded_chat = (
                load_lora_model(adapters_path)
            )

            assert loaded_type == "mac"
            assert loaded_config.dim == 64
            assert loaded_chat == "none"

            # Verify forward pass works
            x = mx.array([[1, 2, 3, 4, 5]])
            logits, _ = loaded_model(x)
            mx.eval(logits)
            assert logits.shape == (1, 5, 256)

            # Verify LoRA modules are present
            lora_mods = _find_lora_modules(loaded_model)
            assert len(lora_mods) > 0
