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
        titans_config = TitansConfig(dim=128, num_heads=2, num_layers=4, vocab_size=1000)
        hf_config = TitansMACConfig.from_titans_config(titans_config)
        assert hf_config.dim == 128
        assert hf_config.num_heads == 2
        assert hf_config.vocab_size == 1000
        assert hf_config.model_type == "titans-mac"

    def test_roundtrip_titans_config(self):
        """TitansConfig -> HF -> TitansConfig preserves all fields."""
        original = TitansConfig(
            dim=512, num_heads=8, num_layers=12, vocab_size=32000,
            chunk_size=256, use_tnt=True, use_attn_res=True, use_mca=True,
            memory_objective="huber", adaptive_window=True,
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
