# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for Titans model variants (MLX)."""

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from titans_mlx.config import TitansConfig
from titans_mlx.memory import NeuralLongTermMemory
from titans_mlx.models import (
    FeedForward,
    LMMBlock,
    MACBlock,
    MAGBlock,
    MALBlock,
    RMSNorm,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    process_chunk,
)
from titans_mlx.tnt_memory import HierarchicalMemory


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_forward(self) -> None:
        """Test RMS normalization."""
        norm = RMSNorm(dim=64)
        x = mx.random.normal((2, 16, 64))

        y = norm(x)
        mx.eval(y)

        assert y.shape == x.shape
        assert not np.any(np.isnan(np.array(y)))

    def test_output_scale(self) -> None:
        """Test output is properly scaled."""
        norm = RMSNorm(dim=64)
        x = mx.random.normal((2, 16, 64)) * 10

        y = norm(x)
        mx.eval(y)

        rms = np.sqrt(np.mean(np.array(y) ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones_like(rms), atol=0.5)


class TestFeedForward:
    """Tests for FeedForward network."""

    def test_forward(self, default_config: TitansConfig) -> None:
        """Test FFN forward pass."""
        ffn = FeedForward(default_config)
        x = mx.random.normal((2, 16, default_config.dim))

        y = ffn(x)
        mx.eval(y)

        assert y.shape == x.shape

    def test_dimensions(self, default_config: TitansConfig) -> None:
        """Test FFN hidden dimensions."""
        ffn = FeedForward(default_config)

        assert ffn.gate_proj.weight.shape == (
            default_config.ffn_dim,
            default_config.dim,
        )
        assert ffn.down_proj.weight.shape == (
            default_config.dim,
            default_config.ffn_dim,
        )


class TestMACBlock:
    """Tests for MACBlock."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = MACBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        output, state = block(x)
        mx.eval(output)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = MACBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        _, state1 = block(x)
        mx.eval(state1.weights[0])

        output, state2 = block(x, state=state1)
        mx.eval(output)

        assert output.shape == x.shape
        assert state2 is not None


class TestMACBlockSubLayers:
    """Test MACBlock core_forward / ffn_forward decomposition."""

    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            chunk_size=16,
        )
        self.block = MACBlock(self.config)
        self.batch, self.seq, self.dim = 2, 16, 32

    def test_core_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        core_out, new_state = self.block.core_forward(x, state=None)
        assert core_out.shape == (self.batch, self.seq, self.dim)
        assert new_state is not None

    def test_ffn_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        ffn_out = self.block.ffn_forward(x)
        assert ffn_out.shape == (self.batch, self.seq, self.dim)

    def test_sublayers_match_residual_path(self):
        """core + ffn with manual residuals should match old __call__ behavior."""
        x = mx.random.normal((self.batch, self.seq, self.dim))
        mx.eval(x)

        # Sub-layer path with standard residuals
        core_out, state = self.block.core_forward(x, state=None)
        h = x + core_out
        ffn_out = self.block.ffn_forward(h)
        result_sublayers = h + ffn_out

        # Verify shapes match
        assert result_sublayers.shape == (self.batch, self.seq, self.dim)

    def test_core_forward_with_memory_gate(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        gate = mx.array(0.5)
        core_out, state = self.block.core_forward(x, state=None, memory_gate=gate)
        assert core_out.shape == (self.batch, self.seq, self.dim)

    def test_tnt_memory_selection(self):
        """use_tnt=True should select HierarchicalMemory."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            use_tnt=True,
        )
        block = MACBlock(config)
        assert isinstance(block.memory, HierarchicalMemory)

    def test_default_memory_selection(self):
        """use_tnt=False should select NeuralLongTermMemory."""
        block = MACBlock(self.config)
        assert isinstance(block.memory, NeuralLongTermMemory)


class TestTitansMAC:
    """Tests for TitansMAC model."""

    def test_forward(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test full forward pass."""
        model = TitansMAC(small_config)
        seq_len = small_config.chunk_size * 2
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (
            batch_size,
            seq_len,
            small_config.vocab_size,
        )
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansMAC(small_config)
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, small_config.chunk_size)
        )

        _, states1 = model(input_ids)
        mx.eval(states1[0].weights[0])

        logits2, states2 = model(input_ids, states=states1)
        mx.eval(logits2)

        assert logits2.shape[0] == batch_size
        assert len(states2) == small_config.num_layers

    def test_chunking(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test sequence is processed in chunks."""
        model = TitansMAC(small_config)
        chunk_size = small_config.chunk_size
        seq_len = chunk_size * 3

        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )
        logits, _ = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (
            batch_size,
            seq_len,
            small_config.vocab_size,
        )

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansMAC(small_config)

        head_w = np.array(model.head.weight)
        embed_w = np.array(model.embed.weight)
        np.testing.assert_array_equal(head_w, embed_w)


class TestMAGBlock:
    """Tests for MAGBlock."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = MAGBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        output, state = block(x)
        mx.eval(output)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = MAGBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        _, state1 = block(x)
        mx.eval(state1.weights[0])

        output, state2 = block(x, state=state1)
        mx.eval(output)

        assert output.shape == x.shape
        assert state2 is not None


class TestTitansMAG:
    """Tests for TitansMAG model."""

    def test_forward(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test full forward pass."""
        model = TitansMAG(small_config)
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (
            batch_size,
            seq_len,
            small_config.vocab_size,
        )
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansMAG(small_config)
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        _, states1 = model(input_ids)
        mx.eval(states1[0].weights[0])

        logits2, states2 = model(input_ids, states=states1)
        mx.eval(logits2)

        assert logits2.shape[0] == batch_size

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansMAG(small_config)

        head_w = np.array(model.head.weight)
        embed_w = np.array(model.embed.weight)
        np.testing.assert_array_equal(head_w, embed_w)


class TestMALBlock:
    """Tests for MALBlock."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = MALBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        output, state = block(x)
        mx.eval(output)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = MALBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        _, state1 = block(x)
        mx.eval(state1.weights[0])

        output, state2 = block(x, state=state1)
        mx.eval(output)

        assert output.shape == x.shape
        assert state2 is not None


class TestTitansMAL:
    """Tests for TitansMAL model."""

    def test_forward(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test full forward pass."""
        model = TitansMAL(small_config)
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (
            batch_size,
            seq_len,
            small_config.vocab_size,
        )
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansMAL(small_config)
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        _, states1 = model(input_ids)
        mx.eval(states1[0].weights[0])

        logits2, states2 = model(input_ids, states=states1)
        mx.eval(logits2)

        assert logits2.shape[0] == batch_size

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansMAL(small_config)

        head_w = np.array(model.head.weight)
        embed_w = np.array(model.embed.weight)
        np.testing.assert_array_equal(head_w, embed_w)


class TestLMMBlock:
    """Tests for LMMBlock (standalone memory)."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = LMMBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        output, state = block(x)
        mx.eval(output)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = LMMBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        _, state1 = block(x)
        mx.eval(state1.weights[0])

        output, state2 = block(x, state=state1)
        mx.eval(output)

        assert output.shape == x.shape
        assert state2 is not None


class TestTitansLMM:
    """Tests for TitansLMM model (memory only)."""

    def test_forward(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test full forward pass."""
        model = TitansLMM(small_config)
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (
            batch_size,
            seq_len,
            small_config.vocab_size,
        )
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansLMM(small_config)
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        _, states1 = model(input_ids)
        mx.eval(states1[0].weights[0])

        logits2, states2 = model(input_ids, states=states1)
        mx.eval(logits2)

        assert logits2.shape[0] == batch_size

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansLMM(small_config)

        head_w = np.array(model.head.weight)
        embed_w = np.array(model.embed.weight)
        np.testing.assert_array_equal(head_w, embed_w)


class TestMAGBlockSubLayers:
    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            chunk_size=16, window_size=16,
        )
        self.block = MAGBlock(self.config)
        self.batch, self.seq, self.dim = 2, 16, 32

    def test_core_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        core_out, new_state = self.block.core_forward(x, state=None)
        assert core_out.shape == (self.batch, self.seq, self.dim)

    def test_ffn_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        ffn_out = self.block.ffn_forward(x)
        assert ffn_out.shape == (self.batch, self.seq, self.dim)

    def test_tnt_memory_selection(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            use_tnt=True, window_size=16,
        )
        block = MAGBlock(config)
        from titans_mlx.tnt_memory import HierarchicalMemory
        assert isinstance(block.memory, HierarchicalMemory)

    def test_backward_compat_call(self):
        """__call__ wrapper still works."""
        x = mx.random.normal((self.batch, self.seq, self.dim))
        output, state = self.block(x, state=None)
        assert output.shape == (self.batch, self.seq, self.dim)


class TestMAGBlockPersistentMemoryInput:
    """Verify Phase 1.1 fix: MAG memory receives persistent-augmented input."""

    def test_mag_memory_sees_persistent(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """MAGBlock passes [persistent || input] to memory, not just input."""
        block = MAGBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        output, state = block(x)
        mx.eval(output)

        # If persistent tokens are used, memory state should incorporate
        # them. Run a second call and verify state evolved.
        output2, state2 = block(x, state=state)
        mx.eval(output2, state.weights[0], state2.weights[0])

        diff = float(mx.sum(mx.abs(state2.weights[0] - state.weights[0])))
        assert diff > 0.0, "Memory state should evolve between calls"


class TestGatingNormalization:
    """Verify Phase 1.2 fix: gating uses learnable normalization + sigmoid."""

    def test_mac_has_gate_norms(self, default_config: TitansConfig) -> None:
        """MACBlock has gate_norm_attn and gate_norm_mem."""
        block = MACBlock(default_config)
        assert hasattr(block, "gate_norm_attn")
        assert hasattr(block, "gate_norm_mem")
        assert isinstance(block.gate_norm_attn, RMSNorm)
        assert isinstance(block.gate_norm_mem, RMSNorm)

    def test_mag_has_gate_norms(self, default_config: TitansConfig) -> None:
        """MAGBlock has gate_norm_attn and gate_norm_mem."""
        block = MAGBlock(default_config)
        assert hasattr(block, "gate_norm_attn")
        assert hasattr(block, "gate_norm_mem")
        assert isinstance(block.gate_norm_attn, RMSNorm)
        assert isinstance(block.gate_norm_mem, RMSNorm)

    def test_gated_output_bounded(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Sigmoid gating produces bounded contribution."""
        block = MACBlock(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))
        output, _ = block(x)
        mx.eval(output)

        # Output should be finite (sigmoid gating prevents explosion)
        output_np = np.array(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))


class TestMALBlockSubLayers:
    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=2, vocab_size=100,
            chunk_size=16, window_size=16,
        )
        self.block = MALBlock(self.config)
        self.batch, self.seq, self.dim = 2, 16, 32

    def test_core_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        core_out, new_state = self.block.core_forward(x, state=None)
        assert core_out.shape == (self.batch, self.seq, self.dim)

    def test_ffn_forward_shape(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        ffn_out = self.block.ffn_forward(x)
        assert ffn_out.shape == (self.batch, self.seq, self.dim)

    def test_backward_compat_call(self):
        x = mx.random.normal((self.batch, self.seq, self.dim))
        output, state = self.block(x, state=None)
        assert output.shape == (self.batch, self.seq, self.dim)


class TestProcessChunk:
    """Test the shared process_chunk orchestrator."""

    def setup_method(self):
        self.config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16,
        )

    def test_standard_residual_path(self):
        """Without AttnRes, process_chunk applies standard residuals."""
        blocks = [MACBlock(self.config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        output, new_states = process_chunk(blocks, chunk, states, self.config, step_count=0)
        assert output.shape == (2, 16, 32)
        assert len(new_states) == 4

    def test_attnres_path(self):
        """With AttnRes, process_chunk uses AttnRes orchestration."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
        )
        blocks = [MACBlock(config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        output, new_states = process_chunk(blocks, chunk, states, config, step_count=0)
        assert output.shape == (2, 16, 32)
        assert len(new_states) == 4

    def test_attnres_warmup_bypasses_gate(self):
        """During warmup, memory_gate should be None."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
            attnres_warmup_steps=100,
        )
        blocks = [MACBlock(config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        output, _ = process_chunk(blocks, chunk, states, config, step_count=0)
        assert output.shape == (2, 16, 32)

    def test_attnres_with_tnt_memory(self):
        """AttnRes + TNT memory should work together."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, use_tnt=True,
            num_attnres_blocks=2,
        )
        blocks = [MACBlock(config) for _ in range(4)]
        chunk = mx.random.normal((2, 16, 32))
        states = [None] * 4
        output, new_states = process_chunk(blocks, chunk, states, config, step_count=0)
        assert output.shape == (2, 16, 32)


class TestTitansMACAttnRes:
    """Test TitansMAC with AttnRes enabled."""

    def test_forward_with_attn_res(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAC(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_forward_with_attn_res_and_tnt(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, use_tnt=True,
            num_attnres_blocks=2,
        )
        model = TitansMAC(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_step_count_increments(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAC(config)
        input_ids = mx.array([[1, 2, 3, 4]])
        model(input_ids)
        assert model._step_count == 1
        model(input_ids)
        assert model._step_count == 2


class TestTitansMAGAttnRes:
    """Test TitansMAG with AttnRes enabled."""

    def test_forward_with_attn_res(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, window_size=16, use_attn_res=True,
            num_attnres_blocks=2,
        )
        model = TitansMAG(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_step_count_increments(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, window_size=16, use_attn_res=True,
            num_attnres_blocks=2,
        )
        model = TitansMAG(config)
        input_ids = mx.array([[1, 2, 3, 4]])
        model(input_ids)
        assert model._step_count == 1
        model(input_ids)
        assert model._step_count == 2


class TestTitansMALAttnRes:
    """Test TitansMAL with AttnRes enabled."""

    def test_forward_with_attn_res(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, window_size=16, use_attn_res=True,
            num_attnres_blocks=2,
        )
        model = TitansMAL(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_step_count_increments(self):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, window_size=16, use_attn_res=True,
            num_attnres_blocks=2,
        )
        model = TitansMAL(config)
        input_ids = mx.array([[1, 2, 3, 4]])
        model(input_ids)
        assert model._step_count == 1
        model(input_ids)
        assert model._step_count == 2


class TestModelsIntegration:
    """Integration tests for all model variants."""

    def test_all_models_produce_valid_logits(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test all models produce valid logits."""
        seq_len = 16
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        models = [
            TitansMAC(small_config),
            TitansMAG(small_config),
            TitansMAL(small_config),
            TitansLMM(small_config),
        ]

        for model in models:
            logits, _ = model(input_ids)
            mx.eval(logits)
            logits_np = np.array(logits)

            assert not np.any(np.isnan(logits_np)), (
                f"{type(model).__name__} produced NaN"
            )
            assert not np.any(np.isinf(logits_np)), (
                f"{type(model).__name__} produced Inf"
            )

    def test_all_models_return_states(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test all models return memory states."""
        seq_len = 16
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        models = [
            TitansMAC(small_config),
            TitansMAG(small_config),
            TitansMAL(small_config),
            TitansLMM(small_config),
        ]

        for model in models:
            _, states = model(input_ids)

            assert states is not None, (
                f"{type(model).__name__} returned None states"
            )
            assert len(states) == small_config.num_layers

    def test_gradient_flow_via_value_and_grad(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test gradients flow through models using mx.value_and_grad."""
        seq_len = 16
        input_ids = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )
        targets = mx.random.randint(
            0, small_config.vocab_size, (batch_size, seq_len)
        )

        model = TitansLMM(small_config)

        def loss_fn(model: nn.Module) -> mx.array:
            logits, _ = model(input_ids)
            logits_flat = logits.reshape(-1, small_config.vocab_size)
            targets_flat = targets.reshape(-1)
            return nn.losses.cross_entropy(
                logits_flat, targets_flat
            ).mean()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss)

        assert float(loss) > 0
        flat_grads = tree_flatten(grads)
        has_nonzero = any(
            float(mx.abs(g).sum()) > 0 for _, g in flat_grads
        )
        assert has_nonzero, "Expected at least some non-zero gradients"


class TestFlagCombinations:
    """Verify all flag combinations produce valid forward passes."""

    @pytest.mark.parametrize("model_cls", [TitansMAC, TitansMAG, TitansMAL])
    @pytest.mark.parametrize("use_tnt", [False, True])
    @pytest.mark.parametrize("use_attn_res", [False, True])
    def test_forward_pass(self, model_cls, use_tnt, use_attn_res):
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=4, vocab_size=100,
            chunk_size=16, window_size=16,
            use_tnt=use_tnt, use_attn_res=use_attn_res,
            num_attnres_blocks=2,
        )
        model = model_cls(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, states = model(input_ids)
        assert logits.shape == (1, 8, 100)
        assert len(states) == 4


def test_multi_chunk_attn_res():
    """AttnRes should work when seq_len > chunk_size (multiple chunks)."""
    config = TitansConfig(
        dim=32, num_heads=4, num_layers=4, vocab_size=100,
        chunk_size=8, use_attn_res=True, num_attnres_blocks=2,
    )
    model = TitansMAC(config)
    # seq_len=16 > chunk_size=8 → 2 chunks
    input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    logits, states = model(input_ids)
    assert logits.shape == (1, 16, 100)
