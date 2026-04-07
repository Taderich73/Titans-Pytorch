"""Tests for Titans model architectures."""

import pytest
import torch

from titans.config import TitansConfig
from titans.models import (
    FeedForward,
    LMMBlock,
    MACBlock,
    MALBlock,
    RMSNorm,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
)


class TestRMSNorm:
    def test_output_shape(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_preserves_dtype(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32


class TestFeedForward:
    def test_output_shape(self, default_config, device):
        ffn = FeedForward(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        out = ffn(x)
        assert out.shape == x.shape


class TestTitansMAC:
    def test_forward_shape(self, default_config, device):
        model = TitansMAC(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_multi_chunk(self, default_config, device):
        """Sequence longer than chunk_size triggers chunked processing."""
        model = TitansMAC(default_config).to(device)
        seq_len = default_config.chunk_size * 2 + 5
        x = torch.randint(0, default_config.vocab_size, (2, seq_len), device=device)
        logits, states = model(x)
        assert logits.shape == (2, seq_len, default_config.vocab_size)

    def test_state_carryover(self, default_config, device):
        """Memory state from first call can be passed to second."""
        model = TitansMAC(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1 = model(x)
        logits2, states2 = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansMAC(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        """Full training step should produce gradients."""
        model = TitansMAC(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None

    def test_tnt_gate_projections_receive_gradients(self, device):
        """Data-dependent gates in both global and local memory must learn.

        End-to-end regression test using TitansMAC with TNT hierarchical
        memory. After a forward + LM cross-entropy + backward, every
        gate_decay_proj / gate_lr_proj / gate_momentum_proj / gate_delta_proj
        in both global_memory and every local_memory across every block must
        have a nonzero gradient on its bias. Prevents the previously silent
        failure where all four gate families were frozen at init for entire
        training runs.

        Note on huber_delta_init: delta's gradient path only carries signal
        when some errors exceed the current delta (Huber else-branch). With
        the default init (delta ~= 5.0) and a randomly-initialized small
        model the errors never reach the knee, so delta has a graph edge
        but zero gradient. Setting huber_delta_init=-10 forces delta ~= 4.5e-4
        so the test exercises the real gradient path. This is a test-only
        choice; production runs can use any delta init they want and delta
        will start learning once errors grow past it.
        """
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=32,
            num_memory_layers=2,
            num_persistent_tokens=4,
            use_tnt=True,
            use_attn_res=True,
            use_mca=True,
            memory_objective="huber",
            huber_delta_init=-10.0,
            adaptive_window=True,
            local_shard_length=128,
        )
        model = TitansMAC(config).to(device)
        model.train()

        ids = torch.randint(0, config.vocab_size, (2, 64), device=device)
        logits, _ = model(ids, states=None)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), ids.reshape(-1)
        )
        loss.backward()

        gate_names = (
            "gate_decay_proj",
            "gate_lr_proj",
            "gate_momentum_proj",
            "gate_delta_proj",
        )

        for block_idx, block in enumerate(model.blocks):
            # Global memory gates
            global_nltm = block.memory.global_memory.memory
            for name in gate_names:
                proj = getattr(global_nltm, name, None)
                assert proj is not None, (
                    f"block[{block_idx}].global.{name} missing"
                )
                assert proj.bias.grad is not None, (
                    f"block[{block_idx}].global.{name}.bias.grad is None"
                )
                assert proj.bias.grad.abs().max() > 0, (
                    f"block[{block_idx}].global.{name}.bias.grad is all zero"
                )

            # Local memory gates (one NLTM per local memory)
            for local_idx, local_mem in enumerate(block.memory.local_memories):
                local_nltm = local_mem.memory
                for name in gate_names:
                    proj = getattr(local_nltm, name, None)
                    assert proj is not None, (
                        f"block[{block_idx}].local[{local_idx}].{name} missing"
                    )
                    assert proj.bias.grad is not None, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is None"
                    )
                    assert proj.bias.grad.abs().max() > 0, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is all zero"
                    )


class TestTitansMAG:
    def test_forward_shape(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_multi_chunk(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        seq_len = default_config.chunk_size * 2 + 5
        x = torch.randint(0, default_config.vocab_size, (2, seq_len), device=device)
        logits, states = model(x)
        assert logits.shape == (2, seq_len, default_config.vocab_size)

    def test_state_carryover(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1 = model(x)
        logits2, states2 = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None


class TestTitansMAL:
    def test_forward_shape(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_multi_chunk(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        seq_len = default_config.chunk_size * 2 + 5
        x = torch.randint(0, default_config.vocab_size, (2, seq_len), device=device)
        logits, states = model(x)
        assert logits.shape == (2, seq_len, default_config.vocab_size)

    def test_state_carryover(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1 = model(x)
        logits2, states2 = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None


class TestTitansLMM:
    def test_forward_shape(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_state_carryover(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1 = model(x)
        logits2, states2 = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None


class TestTNTIntegration:
    @pytest.fixture
    def tnt_config(self):
        return TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=1, num_persistent_tokens=4,
            use_tnt=True, global_chunk_size=64,
            local_chunk_sizes=[8], local_shard_length=64,
            use_qk_projection=True,
        )

    def test_mac_with_tnt(self, tnt_config, device):
        model = TitansMAC(tnt_config).to(device)
        x = torch.randint(0, tnt_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, tnt_config.vocab_size)

    def test_mag_with_tnt(self, tnt_config, device):
        model = TitansMAG(tnt_config).to(device)
        x = torch.randint(0, tnt_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, tnt_config.vocab_size)

    def test_mal_with_tnt(self, tnt_config, device):
        model = TitansMAL(tnt_config).to(device)
        x = torch.randint(0, tnt_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, tnt_config.vocab_size)


class TestMCAIntegration:
    def test_mac_with_mca(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=2, num_persistent_tokens=4,
            use_mca=True, mca_insertion_layers=[0],
        )
        model = TitansMAC(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_mag_with_mca(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=2, num_persistent_tokens=4,
            use_mca=True, mca_insertion_layers=[0],
        )
        model = TitansMAG(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, config.vocab_size)


class TestAttnResIntegration:
    def test_mac_with_attn_res(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=4, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=2, num_persistent_tokens=4,
            use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAC(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_mag_with_attn_res(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=4, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=2, num_persistent_tokens=4,
            use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAG(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_mal_with_attn_res(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=4, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=2, num_persistent_tokens=4,
            use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAL(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_attn_res_backward(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=4, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=2, num_persistent_tokens=4,
            use_attn_res=True, num_attnres_blocks=2,
        )
        model = TitansMAC(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None

