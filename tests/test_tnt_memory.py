"""Tests for TNT Hierarchical Memory."""

import pytest
import torch

from titans.config import TitansConfig


class TestQKProjection:
    def test_update_carry_shape(self, device):
        from titans.qk_projection import QKProjection

        proj = QKProjection(dim=64).to(device)
        keys = torch.randn(2, 8, 64, device=device)
        carry = torch.zeros(64, 64, device=device)
        new_carry = proj.update_carry(keys, carry)
        assert new_carry.shape == (64, 64)

    def test_forward_shapes(self, device):
        from titans.qk_projection import QKProjection

        proj = QKProjection(dim=64).to(device)
        queries = torch.randn(2, 8, 64, device=device)
        keys = torch.randn(2, 8, 64, device=device)
        carry = torch.zeros(64, 64, device=device)
        projected_q, new_carry = proj(queries, keys, carry)
        assert projected_q.shape == (2, 8, 64)
        assert new_carry.shape == (64, 64)

    def test_carry_accumulation(self, device):
        from titans.qk_projection import QKProjection

        proj = QKProjection(dim=64).to(device)
        carry = torch.zeros(64, 64, device=device)
        keys = torch.randn(2, 8, 64, device=device)
        new_carry = proj.update_carry(keys, carry)
        assert new_carry.abs().sum() > 0


class TestTNTMemoryState:
    def test_detach(self, device):
        from titans.memory import MemoryState, TNTMemoryState

        gstate = MemoryState(
            weights=[torch.randn(64, 64, device=device, requires_grad=True)],
            momentum=[torch.zeros(64, 64, device=device)],
        )
        state = TNTMemoryState(
            global_state=gstate, local_states=[], local_inits=[],
            qk_projections=[], local_step_counters=[],
        )
        detached = state.detach()
        assert not detached.global_state.weights[0].requires_grad


class TestHierarchicalMemory:
    @pytest.fixture
    def tnt_config(self):
        return TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=1, num_persistent_tokens=4,
            use_tnt=True, global_chunk_size=64,
            local_chunk_sizes=[8, 16], local_shard_length=64,
            use_qk_projection=True,
        )

    def test_init_state(self, tnt_config, device):
        from titans.tnt_memory import HierarchicalMemory

        mem = HierarchicalMemory(tnt_config).to(device)
        state = mem.init_state(batch_size=2)
        assert len(state.local_states) == 2
        assert len(state.qk_projections) == 2
        assert state.local_step_counters == [0, 0]

    def test_forward_shape(self, tnt_config, device):
        from titans.tnt_memory import HierarchicalMemory

        mem = HierarchicalMemory(tnt_config).to(device)
        x = torch.randn(2, 8, tnt_config.dim, device=device)
        output, new_state = mem(x)
        assert output.shape == (2, 8, tnt_config.dim)
        assert new_state.local_step_counters == [8, 8]

    def test_state_updates(self, tnt_config, device):
        from titans.tnt_memory import HierarchicalMemory

        mem = HierarchicalMemory(tnt_config).to(device)
        x = torch.randn(2, 8, tnt_config.dim, device=device)
        state = mem.init_state(2)
        _, new_state = mem(x, state=state)
        assert not torch.allclose(
            state.global_state.weights[0], new_state.global_state.weights[0],
        )

    def test_retrieve_shape(self, tnt_config, device):
        from titans.tnt_memory import HierarchicalMemory

        mem = HierarchicalMemory(tnt_config).to(device)
        state = mem.init_state(2)
        queries = torch.randn(2, 4, tnt_config.dim, device=device)
        out = mem.retrieve(queries, state)
        assert out.shape == (2, 4, tnt_config.dim)

    def test_backward(self, tnt_config, device):
        from titans.tnt_memory import HierarchicalMemory

        mem = HierarchicalMemory(tnt_config).to(device)
        x = torch.randn(2, 8, tnt_config.dim, device=device, requires_grad=True)
        output, _ = mem(x)
        output.sum().backward()
        assert x.grad is not None


class TestLocalMemoryReset:
    """Regression tests for LocalMemory.maybe_reset batch handling."""

    def test_maybe_reset_accepts_batch_size(self, device):
        """maybe_reset must accept a batch_size parameter."""
        from titans.tnt_memory import LocalMemory

        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=2)
        local = LocalMemory(config, chunk_size=8, shard_length=16).to(device)

        batch_size = 4
        state = local.init_state(batch_size=batch_size)

        # Trigger a reset (step_counter is a multiple of shard_length and > 0)
        new_state, new_counter = local.maybe_reset(
            state, step_counter=16, batch_size=batch_size,
        )

        assert new_counter == 0

    def test_maybe_reset_no_reset_when_not_at_boundary(self, device):
        """maybe_reset must return the same state when not at a shard boundary."""
        from titans.tnt_memory import LocalMemory

        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=2)
        local = LocalMemory(config, chunk_size=8, shard_length=16).to(device)

        state = local.init_state(batch_size=2)
        returned_state, counter = local.maybe_reset(state, step_counter=8, batch_size=2)

        assert counter == 8
        assert returned_state is state

    def test_hierarchical_memory_forward_batch_gt_one_across_shard(self, device):
        """End-to-end: HierarchicalMemory must not crash when reset
        triggers mid-sequence with batch_size > 1."""
        from titans.tnt_memory import HierarchicalMemory

        config = TitansConfig(
            dim=64, num_heads=4, num_memory_layers=2,
            local_chunk_sizes=[8], local_shard_length=8,
        )
        hm = HierarchicalMemory(config).to(device)

        batch_size = 3
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.dim, device=device)

        # First call: counter goes from 0 -> seq_len (16), crossing shard boundary
        out1, state1 = hm(x)
        # Second call: counter is 16, divisible by shard_length=8, will reset
        out2, state2 = hm(x, state=state1)

        assert out2.shape == (batch_size, seq_len, config.dim)
