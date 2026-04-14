"""Tests for checkpoint conversion to HF format."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from titans.checkpoint import save_checkpoint
from titans.config import TitansConfig
from titans.models import TitansMAC

pytest.importorskip("transformers")

from titans.hf.configuration import TitansMACConfig
from titans.hf.modeling import TitansMACForCausalLM

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture
def native_checkpoint(tmp_path):
    """Create a native .pt checkpoint for conversion testing."""
    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        window_size=32,
        num_memory_layers=2,
        num_persistent_tokens=4,
    )
    model = TitansMAC(config)
    ckpt_stem = tmp_path / "native_ckpt"
    save_checkpoint(
        model.state_dict(),
        ckpt_stem,
        format="pt",
        metadata={"config": config.to_dict(), "step": 100},
    )
    return tmp_path / "native_ckpt.pt", config, model


class TestKeyRemapping:
    """State dict key remapping from native -> HF format."""

    def test_remap_adds_model_prefix(self, native_checkpoint):
        ckpt_path, config, native_model = native_checkpoint

        from convert_to_hf import remap_state_dict_keys

        from titans.checkpoint import load_checkpoint

        ckpt = load_checkpoint(ckpt_path)
        remapped = remap_state_dict_keys(ckpt["model"])

        for key in remapped:
            assert key.startswith("model."), f"Key {key} missing model. prefix"

    def test_all_keys_remapped(self, native_checkpoint):
        ckpt_path, config, native_model = native_checkpoint

        from convert_to_hf import remap_state_dict_keys

        from titans.checkpoint import load_checkpoint

        ckpt = load_checkpoint(ckpt_path)
        original_keys = set(ckpt["model"].keys())
        remapped = remap_state_dict_keys(ckpt["model"])
        remapped_stems = {k.removeprefix("model.") for k in remapped}

        assert original_keys == remapped_stems


class TestEndToEndConversion:
    """Full conversion pipeline: native -> HF -> load -> verify logits."""

    def test_converted_model_produces_same_logits(self, native_checkpoint):
        ckpt_path, config, native_model = native_checkpoint
        native_model.eval()

        input_ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            native_logits, _, _ = native_model(input_ids)

        from convert_to_hf import convert_checkpoint

        with tempfile.TemporaryDirectory() as hf_dir:
            convert_checkpoint(
                checkpoint_path=str(ckpt_path),
                output_dir=hf_dir,
                model_type="mac",
                tokenizer_name=None,
            )

            assert (Path(hf_dir) / "config.json").exists()
            assert (Path(hf_dir) / "model.safetensors").exists()

            loaded = TitansMACForCausalLM.from_pretrained(hf_dir)
            loaded.eval()
            with torch.no_grad():
                hf_logits = loaded(input_ids).logits

        assert torch.allclose(native_logits, hf_logits, atol=1e-5)

    def test_config_json_has_required_fields(self, native_checkpoint):
        ckpt_path, config, _ = native_checkpoint

        from convert_to_hf import convert_checkpoint

        with tempfile.TemporaryDirectory() as hf_dir:
            convert_checkpoint(
                checkpoint_path=str(ckpt_path),
                output_dir=hf_dir,
                model_type="mac",
                tokenizer_name=None,
            )
            data = json.loads((Path(hf_dir) / "config.json").read_text())
            assert data["model_type"] == "titans-mac"
            assert "architectures" in data
            assert data["dim"] == 64
