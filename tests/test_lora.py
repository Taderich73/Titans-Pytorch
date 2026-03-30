# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for LoRA module, layer wrapping, and adapter save/load/merge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import numpy as np
from mlx.utils import tree_flatten

from scripts.lora import (
    LoRALinear,
    _find_lora_modules,
    load_adapters,
    merge_lora_weights,
    save_adapters,
    set_lora_enabled,
    wrap_lora_layers,
)
from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC

# ---------------------------------------------------------------------------
# TestLoRALinear
# ---------------------------------------------------------------------------


class TestLoRALinear:
    """Tests for the LoRALinear wrapper module."""

    def test_identity_at_init(self) -> None:
        """LoRA output equals base output at init (B is zeros)."""
        mx.random.seed(42)
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

        x = mx.random.normal((2, 10, 32))
        base_out = base(x)
        lora_out = lora(x)
        mx.eval(base_out, lora_out)

        np.testing.assert_allclose(np.array(lora_out), np.array(base_out), atol=1e-6)

    def test_output_shape(self) -> None:
        """Output shape matches base layer."""
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

        x = mx.random.normal((2, 10, 32))
        out = lora(x)
        mx.eval(out)

        assert out.shape == (2, 10, 64)

    def test_trainable_params(self) -> None:
        """After base.freeze(), only lora_A and lora_B are trainable."""
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

        trainable = lora.trainable_parameters()
        param_dict = dict(tree_flatten(trainable))

        assert "lora_A" in param_dict
        assert "lora_B" in param_dict
        # Base weight should NOT appear (it's frozen)
        assert "base.weight" not in param_dict

    def test_nonzero_after_training_step(self) -> None:
        """After one gradient step, lora_B is no longer all zeros."""
        mx.random.seed(42)
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

        optimizer = mlx.optimizers.Adam(learning_rate=1e-3)

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        x = mx.random.normal((2, 10, 32))
        loss_and_grad = nn.value_and_grad(lora, loss_fn)
        loss, grads = loss_and_grad(lora, x)
        optimizer.update(lora, grads)
        mx.eval(lora.parameters())

        assert not mx.allclose(lora.lora_B, mx.zeros_like(lora.lora_B)).item()


# ---------------------------------------------------------------------------
# TestWrapModel
# ---------------------------------------------------------------------------


def _make_small_config() -> TitansConfig:
    return TitansConfig(
        dim=32,
        num_heads=2,
        num_layers=1,
        vocab_size=50,
        chunk_size=16,
        num_persistent_tokens=2,
        num_memory_layers=1,
        use_conv=False,
        use_rope=False,
        dropout=0.0,
    )


class TestWrapModel:
    """Tests for wrap_lora_layers on a full TitansMAC model."""

    def test_wrap_attn_layers(self) -> None:
        """targets='attn' wraps proj_q/k/v/out but not gate_proj."""
        mx.random.seed(42)
        config = _make_small_config()
        model = TitansMAC(config)

        wrapped = wrap_lora_layers(model, targets="attn", rank=4, alpha=8.0)

        # All attn projections should be wrapped
        for name in ["proj_q", "proj_k", "proj_v", "proj_out"]:
            assert any(name in p and "attention" in p for p in wrapped), (
                f"{name} not wrapped"
            )

        # FFN layers should NOT be wrapped
        assert not any("gate_proj" in p for p in wrapped)

    def test_wrap_ffn_layers(self) -> None:
        """targets='ffn' wraps gate/up/down_proj but not proj_q."""
        mx.random.seed(42)
        config = _make_small_config()
        model = TitansMAC(config)

        wrapped = wrap_lora_layers(model, targets="ffn", rank=4, alpha=8.0)

        for name in ["gate_proj", "up_proj", "down_proj"]:
            assert any(name in p for p in wrapped), f"{name} not wrapped"

        assert not any("proj_q" in p for p in wrapped)

    def test_embed_head_never_wrapped(self) -> None:
        """targets='all' never wraps layers with 'embed' or 'head' in path."""
        mx.random.seed(42)
        config = _make_small_config()
        model = TitansMAC(config)

        wrapped = wrap_lora_layers(model, targets="all", rank=4, alpha=8.0)

        assert not any("embed" in p for p in wrapped)
        assert not any("head" in p for p in wrapped)
        # But some layers should have been wrapped
        assert len(wrapped) > 0

    def test_forward_after_wrap(self) -> None:
        """Model forward pass works after wrapping."""
        mx.random.seed(42)
        config = _make_small_config()
        model = TitansMAC(config)

        wrap_lora_layers(model, targets="attn,ffn", rank=4, alpha=8.0)

        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        logits, _ = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (1, 16, 50)


# ---------------------------------------------------------------------------
# TestAdapterIO
# ---------------------------------------------------------------------------


class TestAdapterIO:
    """Tests for save_adapters, load_adapters, and merge_lora_weights."""

    def test_save_and_load_adapters(self) -> None:
        """Save adapters from model1, load into model2, verify outputs match."""
        mx.random.seed(42)
        config = _make_small_config()
        model1 = TitansMAC(config)
        # Build model2 with identical base weights
        model2 = TitansMAC(config)
        model2.load_weights(list(tree_flatten(model1.parameters())))

        wrap_lora_layers(model1, targets="attn", rank=4, alpha=8.0)
        wrap_lora_layers(model2, targets="attn", rank=4, alpha=8.0)

        # Perturb lora_B in model1 so adapters are non-trivial
        for _, lora_mod in _find_lora_modules(model1):
            lora_mod.lora_B = mx.random.normal(lora_mod.lora_B.shape) * 0.01
        mx.eval(model1.parameters())

        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            save_adapters(model1, adapter_path, meta={"rank": 4})
            load_adapters(model2, adapter_path)

        out1, _ = model1(input_ids)
        out2, _ = model2(input_ids)
        mx.eval(out1, out2)

        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-5)

    def test_merge_lora_weights(self) -> None:
        """After merging, output matches pre-merge LoRA output."""
        mx.random.seed(42)
        config = _make_small_config()
        model = TitansMAC(config)
        wrap_lora_layers(model, targets="attn", rank=4, alpha=8.0)

        # Perturb lora_B so merge is non-trivial
        for _, lora_mod in _find_lora_modules(model):
            lora_mod.lora_B = mx.random.normal(lora_mod.lora_B.shape) * 0.01
        mx.eval(model.parameters())

        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        out_before, _ = model(input_ids)
        mx.eval(out_before)

        merge_lora_weights(model)

        out_after, _ = model(input_ids)
        mx.eval(out_after)

        np.testing.assert_allclose(np.array(out_before), np.array(out_after), atol=1e-5)

    def test_adapter_meta_json(self) -> None:
        """Saved metadata JSON contains expected fields."""
        mx.random.seed(42)
        config = _make_small_config()
        model = TitansMAC(config)
        wrap_lora_layers(model, targets="attn", rank=4, alpha=8.0)

        meta = {
            "rank": 4,
            "alpha": 8.0,
            "lora_targets": "attn",
            "chat_template": "chatml",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            save_adapters(model, adapter_path, meta=meta)

            meta_path = adapter_path.with_suffix(".meta.json")
            assert meta_path.exists()
            loaded_meta = json.loads(meta_path.read_text())

        assert loaded_meta["rank"] == 4
        assert loaded_meta["alpha"] == 8.0
        assert loaded_meta["lora_targets"] == "attn"
        assert loaded_meta["chat_template"] == "chatml"


class TestLoRAEnabled:
    """Tests for LoRA enabled/disabled toggle."""

    def test_disabled_equals_base(self) -> None:
        """When enabled=False, output equals base layer only."""
        mx.random.seed(42)
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

        # Make lora_B nonzero so LoRA would change the output
        lora.lora_B = mx.ones_like(lora.lora_B) * 0.1

        x = mx.random.normal((2, 10, 32))

        # Enabled: output differs from base
        enabled_out = lora(x)
        base_out = base(x)
        mx.eval(enabled_out, base_out)
        assert not np.allclose(np.array(enabled_out), np.array(base_out), atol=1e-6)

        # Disabled: output equals base
        lora.enabled = False
        disabled_out = lora(x)
        mx.eval(disabled_out)
        np.testing.assert_allclose(
            np.array(disabled_out), np.array(base_out), atol=1e-6
        )

    def test_enabled_default_true(self) -> None:
        """LoRALinear.enabled defaults to True."""
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        assert lora.enabled is True


class TestSetLoRAEnabled:
    """Tests for the set_lora_enabled utility."""

    def test_toggle_on_model(self) -> None:
        """set_lora_enabled toggles all LoRA layers in a model."""
        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)
        wrap_lora_layers(model, "attn", rank=4, alpha=8.0)

        # All should be enabled
        for _path, lora_mod in _find_lora_modules(model):
            assert lora_mod.enabled is True

        set_lora_enabled(model, False)
        for _path, lora_mod in _find_lora_modules(model):
            assert lora_mod.enabled is False

        set_lora_enabled(model, True)
        for _path, lora_mod in _find_lora_modules(model):
            assert lora_mod.enabled is True
