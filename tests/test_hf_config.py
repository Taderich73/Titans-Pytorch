"""Tests for TitansMACConfig <-> TitansConfig bridging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from titans.config import TitansConfig

pytest.importorskip("transformers")

from titans.hf.configuration import TitansMACConfig


class TestTitansMACConfig:
    """TitansMACConfig construction and conversion tests."""

    def test_default_construction(self):
        """Config can be created with defaults."""
        config = TitansMACConfig()
        assert config.model_type == "titans-mac"
        assert config.dim == 512
        assert config.num_heads == 8
        assert config.vocab_size == 50257

    def test_custom_construction(self):
        """Config accepts custom values."""
        config = TitansMACConfig(dim=1024, num_heads=16, num_layers=20)
        assert config.dim == 1024
        assert config.num_heads == 16
        assert config.num_layers == 20

    def test_to_titans_config(self):
        """Converts to native TitansConfig with matching fields."""
        hf_config = TitansMACConfig(dim=256, num_heads=4, num_layers=6)
        titans_config = hf_config.to_titans_config()
        assert isinstance(titans_config, TitansConfig)
        assert titans_config.dim == 256
        assert titans_config.num_heads == 4
        assert titans_config.num_layers == 6

    def test_from_titans_config(self):
        """Creates HF config from native TitansConfig."""
        titans_config = TitansConfig(
            dim=128, num_heads=2, num_layers=4, vocab_size=1000
        )
        hf_config = TitansMACConfig.from_titans_config(titans_config)
        assert hf_config.dim == 128
        assert hf_config.num_heads == 2
        assert hf_config.vocab_size == 1000
        assert hf_config.model_type == "titans-mac"

    def test_roundtrip_titans_config(self):
        """TitansConfig -> HF -> TitansConfig preserves all fields."""
        original = TitansConfig(
            dim=512,
            num_heads=8,
            num_layers=12,
            vocab_size=32000,
            chunk_size=256,
            use_tnt=True,
            use_attn_res=True,
            use_mca=True,
            memory_objective="huber",
            adaptive_window=True,
        )
        hf_config = TitansMACConfig.from_titans_config(original)
        restored = hf_config.to_titans_config()
        assert original.to_dict() == restored.to_dict()

    def test_save_load_json_roundtrip(self):
        """Config survives save_pretrained / from_pretrained cycle."""
        config = TitansMACConfig(dim=768, num_heads=12, use_tnt=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)

            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists()

            data = json.loads(config_path.read_text())
            assert data["model_type"] == "titans-mac"
            assert data["dim"] == 768
            assert data["architectures"] == ["TitansMACForCausalLM"]

            loaded = TitansMACConfig.from_pretrained(tmpdir)
            assert loaded.dim == 768
            assert loaded.num_heads == 12
            assert loaded.use_tnt is True

    def test_local_chunk_sizes_list_preserved(self):
        """List fields (local_chunk_sizes) survive JSON roundtrip."""
        config = TitansMACConfig(local_chunk_sizes=[8, 16, 32])
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = TitansMACConfig.from_pretrained(tmpdir)
            assert loaded.local_chunk_sizes == [8, 16, 32]


def test_titans_mac_config_roundtrips_auto_checkpoint_fields():
    """auto_checkpoint and checkpoint_config must survive the
    TitansConfig -> TitansMACConfig -> TitansConfig round-trip."""
    from titans.checkpointing.types import MemoryCheckpointConfig

    cp_cfg = MemoryCheckpointConfig(
        checkpoint_dir="./some/path",
        ring_size=8,
        after_capture_count=3,
        cooldown_chunks=5,
    )
    native = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=1000,
        auto_checkpoint=True,
        checkpoint_config=cp_cfg,
    )

    hf_cfg = TitansMACConfig.from_titans_config(native)
    round_tripped = hf_cfg.to_titans_config()

    assert round_tripped.auto_checkpoint is True
    assert round_tripped.checkpoint_config is not None
    assert round_tripped.checkpoint_config.checkpoint_dir == "./some/path"
    assert round_tripped.checkpoint_config.ring_size == 8
    assert round_tripped.checkpoint_config.after_capture_count == 3
    assert round_tripped.checkpoint_config.cooldown_chunks == 5


def test_titans_mac_config_auto_checkpoint_survives_save_pretrained():
    """Auto-checkpoint fields must survive save_pretrained / from_pretrained."""
    from titans.checkpointing.types import MemoryCheckpointConfig

    cp_cfg = MemoryCheckpointConfig(checkpoint_dir="./cp", ring_size=4)
    hf_cfg = TitansMACConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=1000,
        auto_checkpoint=True,
        checkpoint_config=cp_cfg,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_cfg.save_pretrained(tmpdir)
        loaded = TitansMACConfig.from_pretrained(tmpdir)

    assert loaded.auto_checkpoint is True
    assert loaded.checkpoint_config is not None
    native = loaded.to_titans_config()
    assert native.checkpoint_config is not None
    assert native.checkpoint_config.checkpoint_dir == "./cp"
    assert native.checkpoint_config.ring_size == 4


def test_safe_register_passes_exist_ok_true():
    """_safe_register must pass ``exist_ok=True`` to both upstream register
    calls so duplicate registration is tolerated by transformers directly
    (no substring matching on error messages)."""
    from unittest.mock import MagicMock, patch

    from titans.hf import _safe_register
    from titans.hf.configuration import TitansMACConfig
    from titans.hf.modeling import TitansMACForCausalLM

    mock_cfg = MagicMock()
    mock_model = MagicMock()
    with (
        patch("transformers.AutoConfig.register", mock_cfg),
        patch("transformers.AutoModelForCausalLM.register", mock_model),
    ):
        _safe_register("titans-mac", TitansMACConfig, TitansMACForCausalLM)

    mock_cfg.assert_called_once_with("titans-mac", TitansMACConfig, exist_ok=True)
    mock_model.assert_called_once_with(
        TitansMACConfig, TitansMACForCausalLM, exist_ok=True
    )


def test_safe_register_propagates_unrelated_valueerror():
    """_safe_register must not silently swallow ValueErrors.

    Regression guard: if someone ever reintroduces a bare ``except ValueError``
    around the register calls, this test fails. The helper now relies on
    ``exist_ok=True`` at the upstream level and has no try/except of its own,
    so unrelated errors must propagate untouched.
    """
    import pytest
    from unittest.mock import patch
    from titans.hf import _safe_register
    from titans.hf.configuration import TitansMACConfig
    from titans.hf.modeling import TitansMACForCausalLM

    def raise_other(*args, **kwargs):
        raise ValueError("Some unrelated validation failure")

    with patch("transformers.AutoConfig.register", side_effect=raise_other):
        with pytest.raises(ValueError, match="unrelated"):
            _safe_register("titans-mac", TitansMACConfig, TitansMACForCausalLM)


class TestNewFieldsRoundTrip:
    """New Plan 5 config fields survive HF serialization.

    Covers ``mac_per_position_memory_query`` (Task 5) and
    ``num_memory_inner_steps`` (Task 9). Guards the round-trip in both
    directions so future config additions cannot silently drop these
    fields.
    """

    def test_per_position_query_roundtrip(self):
        """``mac_per_position_memory_query`` survives TitansConfig <-> HF."""
        tc = TitansConfig(mac_per_position_memory_query=False)
        hf = TitansMACConfig.from_titans_config(tc)
        tc2 = hf.to_titans_config()
        assert tc2.mac_per_position_memory_query is False

    def test_inner_steps_roundtrip(self):
        """``num_memory_inner_steps`` survives TitansConfig <-> HF."""
        tc = TitansConfig(num_memory_inner_steps=16)
        hf = TitansMACConfig.from_titans_config(tc)
        tc2 = hf.to_titans_config()
        assert tc2.num_memory_inner_steps == 16

    def test_old_hf_config_loads_with_defaults(self):
        """Omitting the new fields yields the paper-aligned defaults."""
        hf = TitansMACConfig(dim=64, num_heads=4)
        tc = hf.to_titans_config()
        assert tc.mac_per_position_memory_query is True  # new default
        assert tc.num_memory_inner_steps == 1

    def test_new_fields_survive_save_pretrained(self):
        """Full save_pretrained / from_pretrained round-trip preserves
        both new fields, not just in-memory conversion."""
        tc = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            mac_per_position_memory_query=False,
            num_memory_inner_steps=8,
        )
        hf = TitansMACConfig.from_titans_config(tc)
        with tempfile.TemporaryDirectory() as tmpdir:
            hf.save_pretrained(tmpdir)
            loaded = TitansMACConfig.from_pretrained(tmpdir)

        # Raw HF attributes
        assert loaded.mac_per_position_memory_query is False
        assert loaded.num_memory_inner_steps == 8

        # And they come back correctly through to_titans_config
        restored = loaded.to_titans_config()
        assert restored.mac_per_position_memory_query is False
        assert restored.num_memory_inner_steps == 8


def test_titans_hf_double_import_is_idempotent():
    """Re-importing titans.hf in the same process must not raise.

    Regression guard: ``_safe_register`` passes ``exist_ok=True`` to
    transformers' register APIs, so duplicate registration during a
    module reload is tolerated at the upstream level.
    """
    import importlib
    import titans.hf

    importlib.reload(titans.hf)
