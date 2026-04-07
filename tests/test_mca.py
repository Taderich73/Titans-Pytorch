"""Tests for Memory Cross-Attention."""

import pytest
import torch

from titans.config import TitansConfig


@pytest.fixture
def mca_config():
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        use_mca=True,
        mca_insertion_layers=[0],
        mca_num_heads=4,
        mca_gate_type="scalar",
        mca_gate_bias_init=-3.0,
    )


class TestMemoryCrossAttention:
    def test_output_shape(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device)
        memory_weights = torch.randn(mca_config.dim, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.shape == x.shape

    def test_gate_init_near_zero(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device)
        memory_weights = torch.randn(mca_config.dim, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.abs().mean() < 1.0

    def test_vector_gate(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            use_mca=True, mca_insertion_layers=[0],
            mca_gate_type="vector",
        )
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(config).to(device)
        x = torch.randn(2, 8, config.dim, device=device)
        memory_weights = torch.randn(config.dim, config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.shape == x.shape

    def test_deep_memory_weights(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device)
        num_rows = mca_config.memory_hidden_dim
        memory_weights = torch.randn(num_rows, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.shape == x.shape

    def test_backward(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device, requires_grad=True)
        memory_weights = torch.randn(mca_config.dim, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        out.sum().backward()
        assert x.grad is not None


class TestMCAForwardValidation:
    def test_mca_forward_raises_valueerror_on_non_2d_weight(self, device):
        """_mca_forward must raise ValueError (not AssertionError) for
        a memory weight tensor that is not 2D — robust under python -O."""
        from titans.config import TitansConfig
        from titans.memory import MemoryState
        from titans.models import _mca_forward

        config = TitansConfig(dim=32, num_heads=4, use_mca=True)
        del config  # config is not actually needed by _mca_forward; kept for clarity

        # Build a fake mem_state with a 3D weight tensor
        bad_weight = torch.zeros(2, 4, 4, device=device)
        mem_state = MemoryState(weights=[bad_weight], momentum=[bad_weight])

        # A minimal stub block — _mca_forward only touches `block.mca`
        class _Stub:
            def mca(self, h, W):
                return h
        block = _Stub()

        h = torch.randn(2, 8, 32, device=device)
        with pytest.raises(ValueError, match="2D weight matrix"):
            _mca_forward(block, h, mem_state)
