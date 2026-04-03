"""Tests for Neural Long-term Memory module."""

import torch

from titans.config import TitansConfig
from titans.memory import MemoryMLP, MemoryState, NeuralLongTermMemory


class TestMemoryState:
    def test_detach(self, device):
        w = torch.randn(4, 4, device=device, requires_grad=True)
        m = torch.randn(4, 4, device=device, requires_grad=True)
        state = MemoryState(weights=[w], momentum=[m])
        detached = state.detach()
        assert not detached.weights[0].requires_grad
        assert not detached.momentum[0].requires_grad

    def test_clone(self, device):
        w = torch.randn(4, 4, device=device)
        state = MemoryState(weights=[w], momentum=[torch.zeros_like(w)])
        cloned = state.clone()
        w.fill_(0)
        assert cloned.weights[0].abs().sum() > 0


class TestMemoryMLP:
    def test_forward_shape_linear(self, device):
        config = TitansConfig(dim=64, num_memory_layers=1)
        mlp = MemoryMLP(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = mlp(x)
        assert out.shape == (2, 8, 64)

    def test_forward_shape_deep(self, default_config, device):
        mlp = MemoryMLP(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        out = mlp(x)
        assert out.shape == (2, 8, default_config.dim)

    def test_forward_with_weights(self, device):
        config = TitansConfig(dim=32, num_memory_layers=1)
        mlp = MemoryMLP(config).to(device)
        x = torch.randn(2, 4, 32, device=device)
        weights = mlp.get_weights()
        out1 = mlp(x)
        out2 = mlp.forward_with_weights(x, weights)
        torch.testing.assert_close(out1, out2)


class TestNeuralLongTermMemory:
    def test_init_state(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        state = mem.init_state(batch_size=2)
        assert len(state.weights) == default_config.num_memory_layers
        assert state.weights[0].device.type == device.type
        assert state.momentum[0].abs().sum() == 0

    def test_forward_shape(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        output, new_state = mem(x)
        assert output.shape == x.shape
        assert new_state is not None

    def test_state_changes_after_forward(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        state = mem.init_state(batch_size=2)
        _, new_state = mem(x, state=state)
        assert not torch.allclose(state.weights[0], new_state.weights[0])

    def test_linear_memory_forward(self, linear_memory_config, device):
        """Single-layer memory uses parallel update path."""
        mem = NeuralLongTermMemory(linear_memory_config).to(device)
        x = torch.randn(2, 8, linear_memory_config.dim, device=device)
        output, new_state = mem(x)
        assert output.shape == x.shape
        assert len(new_state.weights) == 1

    def test_forward_no_state(self, default_config, device):
        """Forward without explicit state should auto-initialize."""
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        output, state = mem(x, state=None)
        assert output.shape == x.shape
        assert state is not None

    def test_state_detached(self, default_config, device):
        """Returned state should be detached from computation graph."""
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        _, state = mem(x)
        for w in state.weights:
            assert not w.requires_grad
        for m in state.momentum:
            assert not m.requires_grad

    def test_with_conv(self, device):
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, use_conv=True)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state = mem(x)
        assert output.shape == x.shape

    def test_without_conv(self, device):
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, use_conv=False)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state = mem(x)
        assert output.shape == x.shape

    def test_huber_objective(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_memory_layers=1, memory_objective="huber"
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state = mem(x)
        assert output.shape == x.shape

    def test_retrieve(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        state = mem.init_state(batch_size=2)
        queries = torch.randn(2, 4, default_config.dim, device=device)
        retrieved = mem.retrieve(queries, state)
        assert retrieved.shape == queries.shape

    def test_gradients_flow_through_projections(self, default_config, device):
        """Main training graph: gradients should flow through proj_q/out.

        Note: proj_k feeds the analytical memory update path (detached state),
        so it does not receive autograd gradients from the output. The retrieval
        path (proj_q -> forward_with_weights -> proj_out) is the differentiable
        path that carries gradients back to the input.
        """
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device, requires_grad=True)
        output, _ = mem(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert mem.proj_q.weight.grad is not None
        assert mem.proj_out.weight.grad is not None
