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

    def test_state_graph_under_default_flag(self, default_config, device):
        """Default behavior: returned state carries the autograd graph.

        With detach_memory_state_in_forward=False (the fix), new_state is
        returned with requires_grad=True so data-dependent gate projections
        (alpha, theta, eta, delta) can receive gradients via downstream
        consumers of new_state (e.g. HierarchicalMemory.retrieve on the
        new state). Cross-batch gradient leakage is prevented at the
        training loop boundary, not inside this function.
        """
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        _, state = mem(x)
        assert default_config.detach_memory_state_in_forward is False
        for w in state.weights:
            assert w.requires_grad
        # Momentum isn't always a graph node (starts from zeros in init_state
        # and updates through the analytical path); assert it's at least not
        # explicitly detached into a leaf that blocks backward.
        for m in state.momentum:
            assert m.grad_fn is not None or not m.requires_grad

    def test_state_detached_under_legacy_flag(self, device):
        """Legacy flag: returned state is fully detached for backward compat."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=32,
            num_memory_layers=2,
            num_persistent_tokens=4,
            detach_memory_state_in_forward=True,
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, config.dim, device=device)
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

        The retrieval path (proj_q -> forward_with_weights -> proj_out) is the
        direct differentiable path that carries gradients back to the input.
        With detach_memory_state_in_forward defaulting to False, proj_k and
        proj_v also receive gradients indirectly through the state-update math
        (k, v -> _parallel_memory_update_linear -> new_state -> retrieve ->
        output).
        """
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device, requires_grad=True)
        output, _ = mem(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert mem.proj_q.weight.grad is not None
        assert mem.proj_out.weight.grad is not None

    def test_gradients_flow_through_data_dependent_gates(self, device):
        """Gate projections (alpha, theta, eta, delta) must receive gradients.

        Regression test for a structural bug where new_state was detached
        unconditionally inside NeuralLongTermMemory.forward, severing the only
        path from gate projections (gate_decay_proj, gate_lr_proj,
        gate_momentum_proj, gate_delta_proj) to the loss. The fix makes the
        detach conditional on config.detach_memory_state_in_forward (default
        False), letting gate projections learn via autograd while cross-batch
        gradient flow is still prevented by training-loop detach at batch
        boundaries.

        Two things matter for this test:

        1. `output` from NLTM.forward only depends on the incoming state, not
           the new state. The gates only affect new_state, so the test must
           use new_state in some downstream computation (here via retrieve)
           to get a gradient path. This mirrors how HierarchicalMemory.forward
           consumes the output of global_memory: it discards the returned
           `output` and uses the returned `new_state` in its own retrieve
           call.

        2. `gate_delta_proj` (Huber knee) is only active when some errors
           exceed the current delta. With the default init (delta ~= 5.0) and
           a randomly-initialized small model, errors never reach the knee
           and delta has zero gradient even though the graph edge exists.
           Setting huber_delta_init=-10 gives delta ~= 4.5e-4 so the Huber
           else-branch activates and delta receives real gradient signal.
        """
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            memory_objective="huber",
            huber_delta_init=-10.0,
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 16, config.dim, device=device, requires_grad=True)
        _, new_state = mem(x)

        # Simulate HierarchicalMemory.forward: retrieve via the NEW state so
        # the gates (which only affect new_state) have a path to the loss.
        retrieved = mem.retrieve(x, new_state)
        loss = retrieved.sum()
        loss.backward()

        for name in (
            "gate_decay_proj",
            "gate_lr_proj",
            "gate_momentum_proj",
            "gate_delta_proj",
        ):
            proj = getattr(mem, name)
            assert proj.bias.grad is not None, f"{name}.bias.grad is None"
            assert proj.bias.grad.abs().max() > 0, (
                f"{name}.bias.grad is all zero (expected nonzero)"
            )
            assert proj.weight.grad is not None, f"{name}.weight.grad is None"
            assert proj.weight.grad.abs().max() > 0, (
                f"{name}.weight.grad is all zero (expected nonzero)"
            )

class TestNLTMRetrievalOrder:
    """Verify NeuralLongTermMemory retrieves from the UPDATED state (Eq. 3-4)."""

    def test_output_depends_on_gate_decay(self):
        """NLTM output must change when gate_decay_proj bias changes.

        If retrieval uses the old (input) state, the output is independent of
        alpha and this test fails. If retrieval uses new_state (as Eq. 3-4
        requires), the output depends on alpha via new_state.weights.
        """
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=1, vocab_size=64,
            chunk_size=16, num_memory_layers=1, num_persistent_tokens=4,
        )
        mem = NeuralLongTermMemory(config)
        mem.eval()

        x = torch.randn(1, 16, 32)
        state = mem.init_state(1)

        # Forward with original bias
        out1, _ = mem(x, state=state, return_state=True)

        # Change gate_decay_proj bias and forward again with same input + state
        with torch.no_grad():
            mem.gate_decay_proj.bias.fill_(5.0)  # very different from init
        out2, _ = mem(x, state=state, return_state=True)

        # If retrieval uses updated state, outputs must differ
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "NLTM output is independent of gate_decay_proj — retrieval likely "
            "uses old state instead of updated state (violates Eq. 3-4)"
        )

    def test_gate_projections_receive_gradients_through_output(self):
        """Gate projections must have nonzero gradients from loss through output.

        This tests the direct gradient path: loss -> output -> proj_out ->
        retrieved -> forward_with_weights(new_state.weights) -> alpha ->
        gate_decay_proj.bias.
        """
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=1, vocab_size=64,
            chunk_size=16, num_memory_layers=1, num_persistent_tokens=4,
            memory_objective="huber", huber_delta_init=-10.0,
        )
        mem = NeuralLongTermMemory(config)
        mem.train()

        x = torch.randn(2, 16, 32)
        state = mem.init_state(2)

        output, new_state = mem(x, state=state, return_state=True)
        loss = output.sum()
        loss.backward()

        gate_names = ["gate_decay_proj", "gate_lr_proj", "gate_momentum_proj",
                      "gate_delta_proj"]
        for name in gate_names:
            proj = getattr(mem, name, None)
            if proj is None:
                continue
            assert proj.bias.grad is not None, f"{name}.bias.grad is None"
            assert proj.bias.grad.abs().max() > 0, f"{name}.bias.grad is zero"


class TestNeuralLongTermMemoryLegacy:
    def test_legacy_detach_flag_freezes_gates(self, device):
        """Legacy flag preserves pre-fix broken behavior for backward compat.

        When detach_memory_state_in_forward=True, new_state is detached inside
        forward, severing gate projections from the loss even when downstream
        code uses new_state in a retrieve call. Retained so checkpoints from
        broken runs can be reloaded and scored with matching semantics.
        """
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            memory_objective="huber",
            huber_delta_init=-10.0,
            detach_memory_state_in_forward=True,
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 16, config.dim, device=device, requires_grad=True)
        _, new_state = mem(x)

        # Downstream retrieve on the (now-detached) new_state can't reach the
        # gates, so even this more forgiving path has no gradient signal.
        retrieved = mem.retrieve(x, new_state)
        loss = retrieved.sum()
        loss.backward()

        for name in (
            "gate_decay_proj",
            "gate_lr_proj",
            "gate_momentum_proj",
            "gate_delta_proj",
        ):
            proj = getattr(mem, name)
            assert proj.bias.grad is None, (
                f"{name}.bias.grad should be None under legacy flag"
            )
            assert proj.weight.grad is None, (
                f"{name}.weight.grad should be None under legacy flag"
            )

        # Sanity: main retrieval-path gradients still flow under legacy flag.
        assert mem.proj_q.weight.grad is not None
        assert mem.proj_out.weight.grad is not None
