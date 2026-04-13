"""Tests for TitansMACForCausalLM HuggingFace model wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from titans.config import TitansConfig

pytest.importorskip("transformers")

from transformers.modeling_outputs import CausalLMOutputWithPast

from titans.hf.configuration import TitansMACConfig
from titans.hf.modeling import TitansMACForCausalLM


@pytest.fixture
def small_hf_config():
    """Small config for fast HF model tests."""
    return TitansMACConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        window_size=32,
        max_seq_len=256,
        num_memory_layers=2,
        num_persistent_tokens=4,
    )


class TestModelConstruction:
    """Model can be built from config."""

    def test_construction(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        assert model.vocab_size == 256
        assert model.config.model_type == "titans-mac"

    def test_param_count_nonzero(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0


class TestForward:
    """forward() returns correct output type and shapes."""

    def test_forward_without_labels(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        model.eval()
        input_ids = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            outputs = model(input_ids)
        assert isinstance(outputs, CausalLMOutputWithPast)
        assert outputs.logits.shape == (2, 16, 256)
        assert outputs.loss is None
        assert outputs.past_key_values is not None

    def test_forward_with_labels(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        input_ids = torch.randint(0, 256, (2, 16))
        labels = input_ids.clone()
        outputs = model(input_ids, labels=labels)
        assert outputs.loss is not None
        assert outputs.loss.ndim == 0  # scalar

    def test_forward_with_memory_states(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        model.eval()
        input_ids = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            out1 = model(input_ids)
            states = out1.past_key_values
            out2 = model(input_ids, memory_states=states)
        assert out2.logits.shape == (2, 16, 256)


class TestWeightTying:
    """Embedding and head weights are tied."""

    def test_weights_tied(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        assert model.get_input_embeddings().weight.data_ptr() == \
               model.get_output_embeddings().weight.data_ptr()

    def test_weights_tied_after_load(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = TitansMACForCausalLM.from_pretrained(tmpdir)
        assert loaded.get_input_embeddings().weight.data_ptr() == \
               loaded.get_output_embeddings().weight.data_ptr()


class TestSaveLoad:
    """save_pretrained / from_pretrained roundtrip."""

    def test_roundtrip_weights_match(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        model.eval()
        input_ids = torch.randint(0, 256, (1, 16))

        with torch.no_grad():
            original_logits = model(input_ids).logits

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            assert (Path(tmpdir) / "config.json").exists()
            assert (Path(tmpdir) / "model.safetensors").exists()

            loaded = TitansMACForCausalLM.from_pretrained(tmpdir)
            loaded.eval()
            with torch.no_grad():
                loaded_logits = loaded(input_ids).logits

        assert torch.allclose(original_logits, loaded_logits, atol=1e-6)


class TestGenerate:
    """Custom chunked generation with memory state management."""

    def test_generate_produces_tokens(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        model.eval()
        input_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=10)
        assert generated.shape == (1, 18)  # 8 prompt + 10 generated
        assert (generated[:, :8] == input_ids).all()  # prompt preserved

    def test_generate_greedy(self, small_hf_config):
        model = TitansMACForCausalLM(small_hf_config)
        model.eval()
        input_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            gen1 = model.generate(input_ids, max_new_tokens=5, do_sample=False)
            gen2 = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        assert (gen1 == gen2).all()  # deterministic

    def test_generate_multi_chunk_prefill(self, small_hf_config):
        """Prompt longer than chunk_size is handled via chunked prefill."""
        model = TitansMACForCausalLM(small_hf_config)
        model.eval()
        # chunk_size=32, prompt=48 -> requires 2 chunks for prefill
        input_ids = torch.randint(0, 256, (1, 48))
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=5)
        assert generated.shape == (1, 53)

    def test_generate_with_memory_states(self, small_hf_config):
        """Can pass initial memory states to generate."""
        model = TitansMACForCausalLM(small_hf_config)
        model.eval()
        input_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out = model(input_ids)
            states = out.past_key_values
            generated = model.generate(
                input_ids, max_new_tokens=5, memory_states=states,
            )
        assert generated.shape == (1, 13)


class TestAutoRegistration:
    """Auto-registration with transformers AutoModel."""

    def test_auto_config_registered(self):
        from transformers import AutoConfig
        config = AutoConfig.for_model("titans-mac")
        assert isinstance(config, TitansMACConfig)

    def test_auto_model_registered(self, small_hf_config):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(small_hf_config)
        assert isinstance(model, TitansMACForCausalLM)
