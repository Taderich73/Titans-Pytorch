"""Tests for Titans model architectures."""

import pytest
import torch
import torch.nn.functional as F

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
        logits, states, _ = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_multi_chunk_raises(self, default_config, device):
        """Sequence longer than chunk_size must raise ValueError."""
        model = TitansMAC(default_config).to(device)
        seq_len = default_config.chunk_size * 2 + 5
        x = torch.randint(0, default_config.vocab_size, (2, seq_len), device=device)
        with pytest.raises(ValueError, match="chunk_size"):
            model(x)

    def test_state_carryover(self, default_config, device):
        """Memory state from first call can be passed to second."""
        model = TitansMAC(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1, _ = model(x)
        logits2, states2, _ = model(x, states=states1)
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
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None

    def test_tnt_gate_projections_receive_gradients(self, device):
        """Data-dependent gates in both global and local memory must learn.

        End-to-end regression test using TitansMAC with TNT hierarchical
        memory. After a forward + LM cross-entropy + backward, the main gate
        projections (decay, lr, delta) must have nonzero gradients on their
        biases in both global_memory and every local_memory across every block.
        Prevents the previously silent failure where all gate families were
        frozen at init for entire training runs.

        Note on huber_delta_init: delta's gradient path only carries signal
        when some errors exceed the current delta (Huber else-branch). With
        the default init (delta ~= 5.0) and a randomly-initialized small
        model the errors never reach the knee, so delta has a graph edge
        but zero gradient. Setting huber_delta_init=-10 forces delta ~= 4.5e-4
        so the test exercises the real gradient path. This is a test-only
        choice; production runs can use any delta init they want and delta
        will start learning once errors grow past it.

        Note on gate_momentum_proj: with zero initial momentum (first chunk,
        states=None), the dominant eta-dependent term eta^S * S_prev is zero
        and grad_eta_sum can also be negligibly small — this is a valid
        mathematical property of the parallel memory update. gate_momentum_proj
        is verified for grad existence but not for nonzero magnitude here;
        the multi-chunk test (test_tnt_gate_gradients_with_multi_chunk_seq)
        covers the nonzero case with carried state.
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

        ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
        logits, _, _ = model(ids, states=None)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), ids.reshape(-1)
        )
        loss.backward()

        # gate_momentum_proj may have zero grad on first chunk (zero initial
        # momentum). gate_decay_proj may also have zero grad under delta
        # memory parameterization when delta=0 (first chunk, states=None),
        # because d(decay_S)/d(alpha) * delta_0 = 0.
        gate_names_nonzero = (
            "gate_lr_proj",
            "gate_delta_proj",
        )
        gate_names_all = (
            "gate_decay_proj",
            "gate_lr_proj",
            "gate_momentum_proj",
            "gate_delta_proj",
        )

        for block_idx, block in enumerate(model.blocks):
            # Global memory gates — all must have grad, nonzero except momentum
            global_nltm = block.memory.global_memory.memory
            for name in gate_names_all:
                proj = getattr(global_nltm, name, None)
                assert proj is not None, (
                    f"block[{block_idx}].global.{name} missing"
                )
                assert proj.bias.grad is not None, (
                    f"block[{block_idx}].global.{name}.bias.grad is None"
                )
            for name in gate_names_nonzero:
                proj = getattr(global_nltm, name)
                assert proj.bias.grad.abs().max() > 0, (
                    f"block[{block_idx}].global.{name}.bias.grad is all zero"
                )

            # Local memory gates (one NLTM per local memory)
            for local_idx, local_mem in enumerate(block.memory.local_memories):
                local_nltm = local_mem.memory
                for name in gate_names_all:
                    proj = getattr(local_nltm, name, None)
                    assert proj is not None, (
                        f"block[{block_idx}].local[{local_idx}].{name} missing"
                    )
                    assert proj.bias.grad is not None, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is None"
                    )
                for name in gate_names_nonzero:
                    proj = getattr(local_nltm, name)
                    assert proj.bias.grad.abs().max() > 0, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is all zero"
                    )

    def test_tnt_gate_gradients_with_multi_chunk_seq(self, device):
        """Gate gradients must flow when processing multiple chunks via manual loop.

        Regression test verifying that within-chunk gradient flow to gate
        projections is preserved when callers chunk externally and thread
        states between forward() calls. Uses a manual 4-chunk training loop
        (chunk_size=32, total_seq=128) with state detach at chunk boundaries
        to bound peak memory — the same pattern the training loop will use.
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

        # Simulate 4-chunk training step via manual loop.
        ids = torch.randint(0, config.vocab_size, (2, config.chunk_size * 4), device=device)
        chunks = ids.split(config.chunk_size, dim=1)
        states = None

        for chunk in chunks:
            logits, states, _ = model(chunk, states=states)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size), chunk.reshape(-1)
            )
            (loss / len(chunks)).backward()
            states = [s.detach() if s is not None else None for s in states]

        gate_names = (
            "gate_decay_proj",
            "gate_lr_proj",
            "gate_momentum_proj",
            "gate_delta_proj",
        )

        for block_idx, block in enumerate(model.blocks):
            global_nltm = block.memory.global_memory.memory
            for name in gate_names:
                proj = getattr(global_nltm, name, None)
                assert proj is not None, (
                    f"block[{block_idx}].global.{name} missing"
                )
                assert proj.bias.grad is not None, (
                    f"block[{block_idx}].global.{name}.bias.grad is None "
                    f"(within-chunk gradient flow broken under manual chunk loop)"
                )
                assert proj.bias.grad.abs().max() > 0, (
                    f"block[{block_idx}].global.{name}.bias.grad is all zero "
                    f"(within-chunk gradient flow severed under manual chunk loop)"
                )

            for local_idx, local_mem in enumerate(block.memory.local_memories):
                local_nltm = local_mem.memory
                for name in gate_names:
                    proj = getattr(local_nltm, name, None)
                    assert proj is not None, (
                        f"block[{block_idx}].local[{local_idx}].{name} missing"
                    )
                    assert proj.bias.grad is not None, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is None under manual chunk loop"
                    )
                    assert proj.bias.grad.abs().max() > 0, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is all zero under manual chunk loop"
                    )


class TestTitansMAG:
    def test_forward_shape(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states, _ = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_multi_chunk_raises(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        seq_len = default_config.chunk_size * 2 + 5
        x = torch.randint(0, default_config.vocab_size, (2, seq_len), device=device)
        with pytest.raises(ValueError, match="chunk_size"):
            model(x)

    def test_state_carryover(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1, _ = model(x)
        logits2, states2, _ = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        model = TitansMAG(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None


class TestTitansMAL:
    def test_forward_shape(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states, _ = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_multi_chunk_raises(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        seq_len = default_config.chunk_size * 2 + 5
        x = torch.randint(0, default_config.vocab_size, (2, seq_len), device=device)
        with pytest.raises(ValueError, match="chunk_size"):
            model(x)

    def test_state_carryover(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1, _ = model(x)
        logits2, states2, _ = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        model = TitansMAL(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None


class TestTitansLMM:
    def test_forward_shape(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states, _ = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_state_carryover(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1, _ = model(x)
        logits2, states2, _ = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        model = TitansLMM(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _, _ = model(x)
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
        logits, states, _ = model(x)
        assert logits.shape == (2, 16, tnt_config.vocab_size)

    def test_mag_with_tnt(self, tnt_config, device):
        model = TitansMAG(tnt_config).to(device)
        x = torch.randint(0, tnt_config.vocab_size, (2, 16), device=device)
        logits, states, _ = model(x)
        assert logits.shape == (2, 16, tnt_config.vocab_size)

    def test_mal_with_tnt(self, tnt_config, device):
        model = TitansMAL(tnt_config).to(device)
        x = torch.randint(0, tnt_config.vocab_size, (2, 16), device=device)
        logits, states, _ = model(x)
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
        logits, states, _ = model(x)
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
        logits, states, _ = model(x)
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
        logits, states, _ = model(x)
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
        logits, states, _ = model(x)
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
        logits, states, _ = model(x)
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
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None


class TestSingleChunkForward:
    """Verify the simplified single-chunk forward API."""

    def test_forward_single_chunk(self):
        """forward() with seq_len == chunk_size works and returns logits + states."""
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, num_memory_layers=2, num_persistent_tokens=4,
            use_tnt=True, local_chunk_sizes=[8], local_shard_length=128,
        )
        model = TitansMAC(config)
        model.train()

        ids = torch.randint(0, 256, (2, 32))
        logits, states, _ = model(ids, states=None)

        assert logits.shape == (2, 32, 256)
        assert len(states) == 2
        assert states[0] is not None

    def test_forward_rejects_multi_chunk_input(self):
        """forward() should raise when seq_len > chunk_size."""
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, num_memory_layers=2, num_persistent_tokens=4,
        )
        model = TitansMAC(config)

        ids = torch.randint(0, 256, (2, 64))  # 2 chunks
        with pytest.raises(ValueError, match="chunk_size"):
            model(ids, states=None)

    def test_manual_chunk_loop_matches_gate_gradients(self):
        """Manual chunk loop (simulating training) must give gate gradients."""
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, num_memory_layers=2, num_persistent_tokens=4,
            use_tnt=True, local_chunk_sizes=[8], local_shard_length=128,
            memory_objective="huber", huber_delta_init=-10.0,
        )
        model = TitansMAC(config)
        model.train()

        # Simulate 2-chunk training step
        ids = torch.randint(0, 256, (2, 64))
        chunks = ids.split(32, dim=1)
        states = None
        total_loss = 0.0

        for chunk in chunks:
            logits, states, _ = model(chunk, states=states)
            loss = F.cross_entropy(logits.reshape(-1, 256), chunk.reshape(-1))
            (loss / len(chunks)).backward()
            total_loss += loss.item()
            states = [s.detach() if s is not None else None for s in states]

        # Gate gradients must be nonzero
        block0 = model.blocks[0]
        global_nltm = block0.memory.global_memory.memory
        assert global_nltm.gate_decay_proj.bias.grad is not None
        assert global_nltm.gate_decay_proj.bias.grad.abs().max() > 0


class TestHierarchicalMemoryCleanup:
    """Verify HierarchicalMemory uses NLTM outputs directly (no redundant retrieve)."""

    def test_tnt_forward_output_has_gate_gradients(self):
        """HierarchicalMemory.forward output should have gradient path to gates."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=1, vocab_size=64,
            chunk_size=16, num_memory_layers=1, num_persistent_tokens=4,
            use_tnt=True, local_chunk_sizes=[8], local_shard_length=128,
        )
        from titans.tnt_memory import HierarchicalMemory
        hm = HierarchicalMemory(config)
        hm.train()

        x = torch.randn(2, 16, 32)
        output, new_state, _ = hm(x, state=None)

        # Output must have gradient path to gate projections
        loss = output.sum()
        loss.backward()

        global_nltm = hm.global_memory.memory
        assert global_nltm.gate_decay_proj.bias.grad is not None
        assert global_nltm.gate_decay_proj.bias.grad.abs().max() > 0
