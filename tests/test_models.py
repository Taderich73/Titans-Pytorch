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

    def test_tnt_gate_gradients_with_multi_chunk_seq(self, device):
        """Gate gradients must still flow when seq_len > chunk_size.

        Regression test for a follow-up bug to the gate-fix work: with
        seq_len > chunk_size, TitansMAC.forward processes the input as
        multiple sequential chunks (process_chunk loop), threading
        new_states from chunk N to chunk N+1. With the gate fix in place,
        new_states carries its autograd graph, so without an explicit
        chunk-boundary detach the entire multi-chunk graph is held alive
        for the whole forward pass — causing OOM on production-size models.

        The chunk-boundary detach in TitansMAC.forward severs the
        cross-chunk gradient flow but preserves the within-chunk gradient
        flow that the gate fix relies on. This test verifies that gates
        in EVERY block still receive nonzero gradients when the model is
        run with a multi-chunk sequence (chunk_size=32, seq_len=128 -> 4
        chunks). Without within-chunk gradient flow, gates would be dead
        again — defeating the purpose of the original gate fix.
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

        # seq_len = 4 * chunk_size, exercises the multi-chunk branch with
        # 4 chunks of state threading.
        seq_len = config.chunk_size * 4
        ids = torch.randint(0, config.vocab_size, (2, seq_len), device=device)
        logits, _ = model(ids, states=None)
        assert logits.shape == (2, seq_len, config.vocab_size)

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
            global_nltm = block.memory.global_memory.memory
            for name in gate_names:
                proj = getattr(global_nltm, name, None)
                assert proj is not None, (
                    f"block[{block_idx}].global.{name} missing"
                )
                assert proj.bias.grad is not None, (
                    f"block[{block_idx}].global.{name}.bias.grad is None "
                    f"(within-chunk gradient flow broken under multi-chunk)"
                )
                assert proj.bias.grad.abs().max() > 0, (
                    f"block[{block_idx}].global.{name}.bias.grad is all zero "
                    f"(within-chunk gradient flow severed under multi-chunk)"
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
                        f"bias.grad is None under multi-chunk"
                    )
                    assert proj.bias.grad.abs().max() > 0, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is all zero under multi-chunk"
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


class TestChunkCheckpointing:
    """Tests for the torch.utils.checkpoint wrapper around process_chunk."""

    def test_flatten_unflatten_tnt_state_roundtrip(self, device):
        """Flattening a TNT state to tuples and back must preserve tensors.

        The flatten/unflatten helpers are the boundary between the pytree-
        friendly inputs torch.utils.checkpoint expects and the dataclass
        types TitansMAC uses internally. A round-trip must preserve every
        tensor's data_ptr (we want zero-copy views) and every non-tensor
        field exactly.
        """
        from titans.memory import MemoryState, TNTMemoryState
        from titans.models import _flatten_states_to_tuples, _unflatten_tuples_to_states

        global_state = MemoryState(
            weights=[torch.randn(64, 64, device=device), torch.randn(64, 64, device=device)],
            momentum=[torch.zeros(64, 64, device=device), torch.zeros(64, 64, device=device)],
        )
        local_state = MemoryState(
            weights=[torch.randn(64, 64, device=device)],
            momentum=[torch.zeros(64, 64, device=device)],
        )
        local_init = [torch.randn(64, 64, device=device)]
        qk_proj = torch.zeros(64, 64, device=device)

        state = TNTMemoryState(
            global_state=global_state,
            local_states=[local_state, local_state],
            local_inits=[local_init, local_init],
            qk_projections=[qk_proj, qk_proj],
            local_step_counters=[37, 101],
        )

        tuples = _flatten_states_to_tuples([state, None, state])
        restored = _unflatten_tuples_to_states(tuples)

        assert len(restored) == 3
        assert restored[1] is None
        for r in (restored[0], restored[2]):
            assert r is not None
            assert torch.equal(r.global_state.weights[0], global_state.weights[0])
            assert torch.equal(r.global_state.weights[1], global_state.weights[1])
            assert torch.equal(r.global_state.momentum[0], global_state.momentum[0])
            assert torch.equal(r.local_states[0].weights[0], local_state.weights[0])
            assert len(r.local_states) == 2
            assert len(r.local_inits) == 2
            assert len(r.local_inits[0]) == 1
            assert torch.equal(r.local_inits[0][0], local_init[0])
            assert len(r.qk_projections) == 2
            assert r.local_step_counters == [37, 101]

    def test_flatten_unflatten_plain_memory_state_roundtrip(self, device):
        """Round-trip for the non-TNT (plain NeuralLongTermMemory) state.

        Plain MemoryState is what TitansConfig(use_tnt=False) produces.
        The flatten helper must handle both dataclass types so the
        checkpointing path works for non-TNT configs too.
        """
        from titans.memory import MemoryState
        from titans.models import _flatten_states_to_tuples, _unflatten_tuples_to_states

        state = MemoryState(
            weights=[torch.randn(32, 32, device=device)],
            momentum=[torch.zeros(32, 32, device=device)],
        )
        tuples = _flatten_states_to_tuples([state, None])
        restored = _unflatten_tuples_to_states(tuples)
        assert len(restored) == 2
        assert restored[1] is None
        assert torch.equal(restored[0].weights[0], state.weights[0])
        assert torch.equal(restored[0].momentum[0], state.momentum[0])

    def test_flatten_preserves_autograd_graph(self, device):
        """Flatten must NOT call .detach() on any tensor.

        Regression guard: the whole point of the checkpointing work is
        to keep the autograd graph intact for the duration of the
        checkpointed forward. If the flatten helper accidentally
        detaches, gates lose their gradient path.
        """
        from titans.memory import MemoryState, TNTMemoryState
        from titans.models import _flatten_states_to_tuples, _unflatten_tuples_to_states

        w = torch.randn(8, 8, device=device, requires_grad=True)
        # Build a graph-bearing tensor (not a leaf) to simulate new_state.
        w_with_graph = w * 2.0
        assert w_with_graph.grad_fn is not None

        state = TNTMemoryState(
            global_state=MemoryState(weights=[w_with_graph], momentum=[w_with_graph]),
            local_states=[MemoryState(weights=[w_with_graph], momentum=[w_with_graph])],
            local_inits=[[w_with_graph]],
            qk_projections=[w_with_graph],
            local_step_counters=[0],
        )

        tuples = _flatten_states_to_tuples([state])
        restored = _unflatten_tuples_to_states(tuples)
        assert restored[0].global_state.weights[0].grad_fn is not None
        assert restored[0].local_states[0].weights[0].grad_fn is not None
        assert restored[0].local_inits[0][0].grad_fn is not None
        assert restored[0].qk_projections[0].grad_fn is not None

    def test_checkpointed_chunk_matches_non_checkpointed(self, device):
        """Checkpointed process_chunk must produce numerically identical
        outputs and state tensors to non-checkpointed process_chunk.

        This is the correctness contract: activation checkpointing is a
        memory/compute trade, not a numerics change. Bitwise equality
        is not guaranteed in bf16 (autocast rounding may reorder ops
        across the recompute boundary), so we use a tight tolerance at
        fp32 instead.
        """
        from titans.memory import TNTMemoryState
        from titans.models import (
            _flatten_states_to_tuples,
            _run_process_chunk_checkpointed,
            process_chunk,
        )

        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=32,
            num_memory_layers=2,
            num_persistent_tokens=4,
            use_tnt=True,
            use_attn_res=False,
            memory_objective="l2",
        )
        model = TitansMAC(config).to(device)
        model.eval()
        # eval() disables dropout so the two paths are deterministic.

        B, S = 2, config.chunk_size
        x_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
        x = model.embed(x_ids)
        init_states = [None] * config.num_layers

        # Non-checkpointed reference.
        with torch.no_grad():
            ref_chunk, ref_states = process_chunk(
                model.blocks, x, init_states, config, 0
            )

        # Checkpointed path (still runs under no_grad to match the reference;
        # the grad-flow check is a separate test below).
        with torch.no_grad():
            ck_chunk, ck_states = _run_process_chunk_checkpointed(
                chunk=x,
                state_tuples=_flatten_states_to_tuples(init_states),
                blocks=model.blocks,
                config=config,
                step_count=0,
            )

        assert torch.allclose(ref_chunk, ck_chunk, rtol=1e-5, atol=1e-6)
        assert len(ref_states) == len(ck_states)
        for r, c in zip(ref_states, ck_states):
            assert isinstance(r, TNTMemoryState) and isinstance(c, TNTMemoryState)
            assert torch.allclose(
                r.global_state.weights[0], c.global_state.weights[0],
                rtol=1e-5, atol=1e-6,
            )
            assert torch.allclose(
                r.local_states[0].weights[0], c.local_states[0].weights[0],
                rtol=1e-5, atol=1e-6,
            )

    def test_checkpointed_chunk_propagates_gradients(self, device):
        """Backward through the checkpointed runner must populate grads.

        Verifies that the checkpoint recompute path actually wires up
        gradients. Without this, the outer TitansMAC.forward would
        silently produce logits that no parameter can learn from.
        """
        from titans.models import (
            _flatten_states_to_tuples,
            _run_process_chunk_checkpointed,
        )

        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=32,
            num_memory_layers=2,
            num_persistent_tokens=4,
            use_tnt=True,
            use_attn_res=False,
            memory_objective="huber",
            huber_delta_init=-10.0,
        )
        model = TitansMAC(config).to(device)
        model.train()

        B, S = 2, config.chunk_size
        x_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
        x = model.embed(x_ids)

        chunk_out, _ = _run_process_chunk_checkpointed(
            chunk=x,
            state_tuples=_flatten_states_to_tuples([None] * config.num_layers),
            blocks=model.blocks,
            config=config,
            step_count=0,
        )
        loss = chunk_out.sum()
        loss.backward()

        # At least one meaningful param should receive gradient.
        assert model.embed.weight.grad is not None
        assert model.embed.weight.grad.abs().max() > 0
        # The gate projections (the whole reason we care about the graph)
        # should receive gradients via the retrieve-from-new-state path.
        # Note: gate_momentum_proj is excluded here because with zero initial
        # momentum (first chunk, init_states=[None]*num_layers), the dominant
        # eta-dependent term eta^S * S_prev is zero and grad_eta_sum can also
        # be negligibly small — this is a valid mathematical property of the
        # parallel memory update, NOT a checkpointing regression. The
        # non-checkpointed process_chunk path exhibits the same behavior.
        block0_global_nltm = model.blocks[0].memory.global_memory.memory
        for name in (
            "gate_decay_proj",
            "gate_lr_proj",
            "gate_delta_proj",
        ):
            proj = getattr(block0_global_nltm, name)
            assert proj.bias.grad is not None, f"block[0].global.{name}.bias.grad is None"
            assert proj.bias.grad.abs().max() > 0, (
                f"block[0].global.{name}.bias.grad is all zero under checkpointed runner"
            )

    def test_titans_mac_multi_chunk_checkpointed_parity(self, device):
        """Full TitansMAC.forward must match between checkpointed and
        non-checkpointed runs on a multi-chunk sequence.

        Seeds are pinned and both models are constructed from the same
        state dict so the comparison is well-defined. Uses eval() to
        eliminate dropout nondeterminism. Forward outputs must match
        to float32 tolerance.
        """
        torch.manual_seed(0)
        config_ref = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=32,
            num_memory_layers=2,
            num_persistent_tokens=4,
            use_tnt=True,
            use_attn_res=False,
            memory_objective="l2",
            use_chunk_checkpointing=False,
        )
        torch.manual_seed(0)
        config_ck = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=32,
            num_memory_layers=2,
            num_persistent_tokens=4,
            use_tnt=True,
            use_attn_res=False,
            memory_objective="l2",
            use_chunk_checkpointing=True,
        )

        torch.manual_seed(0)
        ref_model = TitansMAC(config_ref).to(device).eval()
        torch.manual_seed(0)
        ck_model = TitansMAC(config_ck).to(device).eval()
        ck_model.load_state_dict(ref_model.state_dict())

        seq_len = config_ref.chunk_size * 4  # 4 chunks
        ids = torch.randint(0, config_ref.vocab_size, (2, seq_len), device=device)

        with torch.no_grad():
            ref_logits, _ = ref_model(ids, states=None)
            ck_logits, _ = ck_model(ids, states=None)

        assert ref_logits.shape == ck_logits.shape == (2, seq_len, config_ref.vocab_size)
        assert torch.allclose(ref_logits, ck_logits, rtol=1e-5, atol=1e-6), (
            f"max_abs_diff={(ref_logits - ck_logits).abs().max().item():.3e}"
        )

    def test_titans_mac_multi_chunk_checkpointed_gate_gradients(self, device):
        """Gate gradients must still flow through the checkpointed
        multi-chunk path. This is the production regression test that
        `test_tnt_gate_gradients_with_multi_chunk_seq` could not catch
        because it did not exercise use_chunk_checkpointing=True.
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
            use_chunk_checkpointing=True,
        )
        model = TitansMAC(config).to(device).train()

        seq_len = config.chunk_size * 4
        ids = torch.randint(0, config.vocab_size, (2, seq_len), device=device)
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
            global_nltm = block.memory.global_memory.memory
            for name in gate_names:
                proj = getattr(global_nltm, name)
                assert proj.bias.grad is not None, (
                    f"block[{block_idx}].global.{name}.bias.grad is None "
                    f"under use_chunk_checkpointing=True"
                )
                assert proj.bias.grad.abs().max() > 0, (
                    f"block[{block_idx}].global.{name}.bias.grad is all zero "
                    f"under use_chunk_checkpointing=True"
                )
            for local_idx, local_mem in enumerate(block.memory.local_memories):
                local_nltm = local_mem.memory
                for name in gate_names:
                    proj = getattr(local_nltm, name)
                    assert proj.bias.grad is not None, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is None under use_chunk_checkpointing=True"
                    )
                    assert proj.bias.grad.abs().max() > 0, (
                        f"block[{block_idx}].local[{local_idx}].{name}."
                        f"bias.grad is all zero under use_chunk_checkpointing=True"
                    )
