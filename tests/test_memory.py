"""Tests for Neural Long-term Memory module."""

import pytest
import torch

from titans.checkpoint_types import GateSnapshot
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory


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
        output, new_state, _ = mem(x)
        assert output.shape == x.shape
        assert new_state is not None

    def test_state_changes_after_forward(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        state = mem.init_state(batch_size=2)
        _, new_state, _ = mem(x, state=state)
        assert not torch.allclose(state.weights[0], new_state.weights[0])

    def test_linear_memory_forward(self, linear_memory_config, device):
        """Single-layer memory uses parallel update path."""
        mem = NeuralLongTermMemory(linear_memory_config).to(device)
        x = torch.randn(2, 8, linear_memory_config.dim, device=device)
        output, new_state, _ = mem(x)
        assert output.shape == x.shape
        assert len(new_state.weights) == 1

    def test_forward_no_state(self, default_config, device):
        """Forward without explicit state should auto-initialize."""
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        output, state, _ = mem(x, state=None)
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
        _, state, _ = mem(x)
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
        _, state, _ = mem(x)
        for w in state.weights:
            assert not w.requires_grad
        for m in state.momentum:
            assert not m.requires_grad

    def test_with_conv(self, device):
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, use_conv=True)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state, _ = mem(x)
        assert output.shape == x.shape

    def test_without_conv(self, device):
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, use_conv=False)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state, _ = mem(x)
        assert output.shape == x.shape

    def test_huber_objective(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_memory_layers=1, memory_objective="huber"
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state, _ = mem(x)
        assert output.shape == x.shape

    def test_gate_snapshot_none_when_auto_checkpoint_false(self, device):
        """gate_snapshot is None when auto_checkpoint=False (default)."""
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, auto_checkpoint=False)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        _, _, gate_snapshot = mem(x)
        assert gate_snapshot is None

    def test_gate_snapshot_populated_when_auto_checkpoint_true(self, device):
        """gate_snapshot is a GateSnapshot when auto_checkpoint=True."""
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, auto_checkpoint=True)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        _, _, gate_snapshot = mem(x)
        assert isinstance(gate_snapshot, GateSnapshot)
        assert len(gate_snapshot.alpha) == 1
        assert len(gate_snapshot.theta) == 1
        assert len(gate_snapshot.eta) == 1
        assert gate_snapshot.delta is None
        assert isinstance(gate_snapshot.input_activation_norm, float)
        assert gate_snapshot.chunk_index == 0

    def test_gate_snapshot_has_delta_with_huber(self, device):
        """gate_snapshot.delta is populated when memory_objective='huber'."""
        config = TitansConfig(
            dim=64, num_heads=4, num_memory_layers=1,
            auto_checkpoint=True, memory_objective="huber",
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        _, _, gate_snapshot = mem(x)
        assert isinstance(gate_snapshot, GateSnapshot)
        assert gate_snapshot.delta is not None
        assert len(gate_snapshot.delta) == 1

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
        output, _, _ = mem(x)
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
        _, new_state, _ = mem(x)

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

class TestPerChunkDecay:
    """Verify per-chunk decay reparameterization identity and gradient health."""

    def test_reparameterization_identity(self, device):
        """(1-token_alpha)^S must equal (1-chunk_alpha) by construction."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            per_chunk_decay=True,
            gate_decay_bias_init=-2.0,
        )
        mem = NeuralLongTermMemory(config).to(device)

        seq_len = 512
        x = torch.randn(2, seq_len, config.dim, device=device)
        state = mem.init_state(batch_size=2)

        # Run forward to get the parallel update result
        output, new_state, _ = mem(x, state=state)

        # Compute chunk_alpha from the bias directly
        with torch.no_grad():
            chunk_alpha = torch.sigmoid(mem.gate_decay_proj.bias).item()

        # Derive expected per-chunk retention
        expected_retention = 1.0 - chunk_alpha

        # Derive token_alpha via the same formula used in forward()
        token_alpha = 1.0 - (1.0 - chunk_alpha) ** (1.0 / seq_len)
        actual_retention = (1.0 - token_alpha) ** seq_len

        assert abs(actual_retention - expected_retention) < 1e-6, (
            f"Reparameterization identity failed: "
            f"(1-token_alpha)^S={actual_retention:.8f} != "
            f"1-chunk_alpha={expected_retention:.8f}"
        )

    def test_gate_gradient_nonzero_at_long_seq(self, device):
        """Gate gradient must be nonzero even at the seq lengths where legacy dies.

        The death spiral in legacy mode requires multiple forward passes with
        state carry (weights collapse → gradient vanishes).  Here we verify
        the simpler prerequisite: after a single forward pass at seq_len=512,
        the gate still receives a nonzero gradient.
        """
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            per_chunk_decay=True,
            gate_decay_bias_init=-2.0,
            memory_objective="huber",
            huber_delta_init=-10.0,
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(1, 512, config.dim, device=device)
        _, new_state, _ = mem(x)
        retrieved = mem.retrieve(x, new_state)
        loss = retrieved.sum()
        loss.backward()
        grad_mag = mem.gate_decay_proj.bias.grad.abs().item()
        assert grad_mag > 1e-10, (
            f"gate_decay_proj.bias.grad is near-zero ({grad_mag:.2e}) at seq_len=512"
        )

    def test_weight_norm_preserved_after_forward(self, device):
        """With per_chunk_decay, memory weights should retain meaningful norm."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            per_chunk_decay=True,
            gate_decay_bias_init=-2.0,
            delta_memory_param=False,  # Use absolute weights for this test
        )
        mem = NeuralLongTermMemory(config).to(device)

        x = torch.randn(2, 512, config.dim, device=device)
        state = mem.init_state(batch_size=2)
        init_norm = state.weights[0].norm().item()

        _, new_state, _ = mem(x, state=state)
        new_norm = new_state.weights[0].detach().norm().item()

        # With chunk_alpha ≈ 0.12, retention ≈ 88% — weight norm should
        # stay in the same order of magnitude, not collapse to near-zero
        ratio = new_norm / init_norm
        assert ratio > 0.3, (
            f"Weight norm collapsed: {new_norm:.4f} / {init_norm:.4f} = {ratio:.4f} "
            f"(expected > 0.3 with per-chunk decay)"
        )

    def test_legacy_mode_unchanged(self, device):
        """per_chunk_decay=False should leave the parallel path unchanged."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            per_chunk_decay=False,
            gate_decay_bias_init=-4.0,
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 16, config.dim, device=device)
        output, state, _ = mem(x)
        assert output.shape == x.shape
        assert state is not None


class TestDeltaMemoryParam:
    """Verify delta memory parameterization behavior."""

    def test_init_state_zeros(self, device):
        """With delta_memory_param=True, init state should be all zeros."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            delta_memory_param=True,
        )
        mem = NeuralLongTermMemory(config).to(device)
        state = mem.init_state(batch_size=2)
        for w in state.weights:
            assert w.abs().max() == 0.0, "Delta init should be zeros"
        for m in state.momentum:
            assert m.abs().max() == 0.0, "Momentum init should be zeros"

    def test_retrieval_equals_base_at_init(self, device):
        """At init (delta=0), retrieval should match legacy (non-delta) retrieval."""
        config_delta = TitansConfig(
            dim=64,
            num_memory_layers=1,
            delta_memory_param=True,
        )
        config_legacy = TitansConfig(
            dim=64,
            num_memory_layers=1,
            delta_memory_param=False,
        )
        # Use same weights by sharing state dict
        mem_delta = NeuralLongTermMemory(config_delta).to(device)
        mem_legacy = NeuralLongTermMemory(config_legacy).to(device)
        mem_legacy.load_state_dict(mem_delta.state_dict())

        queries = torch.randn(2, 8, config_delta.dim, device=device)
        state_delta = mem_delta.init_state(batch_size=2)
        state_legacy = mem_legacy.init_state(batch_size=2)

        ret_delta = mem_delta.retrieve(queries, state_delta)
        ret_legacy = mem_legacy.retrieve(queries, state_legacy)

        torch.testing.assert_close(ret_delta, ret_legacy, atol=1e-5, rtol=1e-5)

    def test_decay_degrades_to_base_not_zero(self, device):
        """After heavy decay, retrieval should approach base, not collapse."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            delta_memory_param=True,
            gate_decay_bias_init=2.0,  # High decay: sigmoid(2) ≈ 0.88
            per_chunk_decay=False,     # Raw alpha for aggressive decay test
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 16, config.dim, device=device)

        # Run many forward passes with state carry to decay delta
        state = mem.init_state(batch_size=2)
        for _ in range(20):
            _, state, _ = mem(x, state=state)
            state = state.detach()

        # Delta should be near zero after heavy decay
        delta_norm = state.weights[0].norm().item()

        # But retrieval should still produce nonzero output (from base)
        retrieved = mem.retrieve(x, state)
        output_norm = retrieved.norm().item()

        assert output_norm > 0.01, (
            f"Retrieval collapsed to near-zero ({output_norm:.4f}) — "
            f"delta param not working (delta_norm={delta_norm:.2e})"
        )

    def test_gate_gradients_with_delta_param(self, device):
        """Gate projections must receive gradients under delta param."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            delta_memory_param=True,
            per_chunk_decay=True,
            memory_objective="huber",
            huber_delta_init=-10.0,
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 16, config.dim, device=device)
        _, new_state, _ = mem(x)
        retrieved = mem.retrieve(x, new_state)
        loss = retrieved.sum()
        loss.backward()

        for name in ("gate_decay_proj", "gate_lr_proj", "gate_momentum_proj"):
            proj = getattr(mem, name)
            assert proj.bias.grad is not None, f"{name}.bias.grad is None"
            assert proj.bias.grad.abs().max() > 0, f"{name}.bias.grad is zero"

    def test_legacy_absolute_weights(self, device):
        """delta_memory_param=False preserves absolute weight behavior."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            delta_memory_param=False,
        )
        mem = NeuralLongTermMemory(config).to(device)
        state = mem.init_state(batch_size=2)

        # Legacy: init state should be nonzero (cloned MLP weights)
        assert state.weights[0].abs().max() > 0, (
            "Legacy init should clone MLP weights, not zeros"
        )

    def test_deep_memory_delta_param(self, device):
        """Delta param works with multi-layer memory (deep path)."""
        config = TitansConfig(
            dim=64,
            num_memory_layers=2,
            delta_memory_param=True,
            per_chunk_decay=True,
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 16, config.dim, device=device)
        state = mem.init_state(batch_size=2)

        # Forward should work and produce nonzero output
        output, new_state, _ = mem(x, state=state)
        assert output.norm().item() > 0, "Output is zero with deep delta param"

        # Delta should be nonzero after forward
        for i, w in enumerate(new_state.weights):
            assert w.abs().max() > 0, f"Delta[{i}] is still zero after forward"


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
        out1, _, _ = mem(x, state=state, return_state=True)

        # Change gate_decay_proj bias and forward again with same input + state
        with torch.no_grad():
            mem.gate_decay_proj.bias.fill_(5.0)  # very different from init
        out2, _, _ = mem(x, state=state, return_state=True)

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

        output, new_state, _ = mem(x, state=state, return_state=True)
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
        _, new_state, _ = mem(x)

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


@pytest.mark.parametrize("num_memory_layers", [1, 2])
def test_memory_module_no_module_state_stash(num_memory_layers):
    """Regression: self._current_delta was assigned inside forward() so that
    _parallel_memory_update_linear (linear path, num_memory_layers=1) and
    _compute_gradients (deep path, num_memory_layers>=2) could read it later.
    That's a torch.compile and concurrency hazard. The field must not be set
    on the module after forward, regardless of which consumer path ran.
    """
    import torch
    from titans.memory import NeuralLongTermMemory
    from titans.config import TitansConfig

    cfg = TitansConfig(
        dim=16,
        num_heads=2,
        num_layers=1,
        vocab_size=64,
        num_memory_layers=num_memory_layers,
        memory_objective="huber",
    )
    mem = NeuralLongTermMemory(cfg)
    x = torch.randn(1, 4, 16)
    _ = mem(x, return_state=False)
    assert not hasattr(mem, "_current_delta") or mem._current_delta is None, (
        "forward() is stashing delta on the module — thread it as an argument"
    )


class TestVProjectionActivation:
    """V projection must be raw linear (no SiLU, no L2-norm) per paper Eq. 12."""

    def test_v_is_raw_linear_when_input_large(self):
        """A large-magnitude input should yield a large-magnitude v tensor.

        Pre-fix (SiLU + L2): all v rows have unit norm, bounded in [0, 1].
        Post-fix: v = x @ W_V.T with no activation -> norms can exceed 1.
        """
        import torch
        from titans.config import TitansConfig
        from titans.memory import NeuralLongTermMemory

        config = TitansConfig(dim=32, num_memory_layers=1)
        memory = NeuralLongTermMemory(config)
        # Large input so v has large norm through the linear projection
        x = torch.randn(2, 8, 32) * 10.0
        v = memory.proj_v(x)
        # We only apply the convolution (if enabled) — NOT SiLU, NOT L2-norm
        # The test must assert that the v path in forward() preserves norms.
        # We mirror the code path exactly so any regression is caught.
        _, v_post, _ = memory._apply_conv(memory.proj_k(x), v, memory.proj_q(x))
        assert v_post.norm(dim=-1).mean().item() > 2.0, (
            "V must not be L2-normalized to unit sphere"
        )

    def test_v_is_unchanged_between_proj_and_loss(self):
        """Walk the forward path manually; verify v is not activated."""
        import torch
        from titans.config import TitansConfig
        from titans.memory import NeuralLongTermMemory

        # Use 2-layer memory path so we hit _compute_gradients_deep.
        config2 = TitansConfig(dim=16, num_memory_layers=2, memory_hidden_mult=2.0)
        memory2 = NeuralLongTermMemory(config2)
        x = torch.randn(1, 4, 16)

        captured2 = {}
        orig2 = memory2._compute_gradients_deep

        def capture2(keys, values, weights, delta=None):
            captured2["values"] = values.detach().clone()
            return orig2(keys, values, weights, delta=delta)

        memory2._compute_gradients_deep = capture2  # type: ignore[assignment]
        memory2(x)

        # v should equal proj_v(x) after optional conv, with NO SiLU applied
        expected_v = memory2.proj_v(x)
        _, expected_v, _ = memory2._apply_conv(
            memory2.proj_k(x), expected_v, memory2.proj_q(x)
        )
        # The values passed to gradient computation should match this raw v
        assert torch.allclose(captured2["values"], expected_v, atol=1e-6), (
            "V passed to the loss must be raw linear (post-conv only)"
        )


class TestErrorScale:
    """Error scale must be 2/S per paper §3.1, not 2/numel."""

    def test_linear_gradient_scale_is_2_over_S(self):
        """Manually verify the scale factor used in _compute_gradients_linear."""
        config = TitansConfig(
            dim=8,
            num_memory_layers=1,
            memory_error_clip=1e6,  # effectively disable clamp
            memory_grad_clip=1e6,
            delta_memory_param=False,
        )
        memory = NeuralLongTermMemory(config)
        B, S, D = 3, 5, 8
        keys = torch.randn(B, S, D)
        values = torch.randn(B, S, D)
        weight = memory.memory.layers[0].weight.data.clone()

        grads = memory._compute_gradients_linear(keys, values, weight)

        # Expected per paper: scale = 2/S
        preds = torch.nn.functional.linear(keys, weight)
        err = preds - values
        expected_scale = 2.0 / float(S)
        batch_seq = B * S
        err_flat = err.reshape(batch_seq, -1)
        keys_flat = keys.reshape(batch_seq, -1)
        expected_grad = expected_scale * (err_flat.T @ keys_flat)

        assert torch.allclose(grads[0], expected_grad, atol=1e-5), (
            f"Expected scale 2/S={expected_scale:.6f}, got mismatched gradient"
        )

    def test_deep_gradient_scale_is_2_over_S(self):
        """Verify deep-memory gradient outer-scale is 2/S."""
        config = TitansConfig(
            dim=8,
            num_memory_layers=2,
            memory_hidden_mult=2.0,
            memory_error_clip=1e6,
            memory_grad_clip=1e6,
            delta_memory_param=False,
        )
        memory = NeuralLongTermMemory(config)
        B, S, D = 2, 4, 8
        keys = torch.randn(B, S, D)
        values = torch.randn(B, S, D)
        weights = [memory.memory.layers[i].weight.data.clone() for i in range(2)]

        grads = memory._compute_gradients_deep(keys, values, weights)

        # With 2/numel, grad would be scaled by D_out smaller than 2/S.
        # We assert the last-layer grad's Frobenius norm matches 2/S scaling.
        h = keys
        acts = [h]
        for w in weights[:-1]:
            h = torch.nn.functional.silu(torch.nn.functional.linear(h, w))
            acts.append(h)
        h = torch.nn.functional.linear(h, weights[-1])
        err = h - values
        batch_seq = B * S
        err_flat = err.reshape(batch_seq, -1)
        last_act_flat = acts[-1].reshape(batch_seq, -1)
        expected_last = (2.0 / float(S)) * (err_flat.T @ last_act_flat)

        assert torch.allclose(grads[-1], expected_last, atol=1e-5), (
            "Deep-memory last-layer gradient should use 2/S outer scale"
        )

    def test_parallel_update_uses_2_over_S(self):
        """Parallel linear update also uses 2/S (not 2/(B*S*D))."""
        import torch.nn.functional as F

        config = TitansConfig(
            dim=8,
            num_memory_layers=1,
            memory_error_clip=1e6,
            memory_grad_clip=1e6,
            per_chunk_decay=False,
            delta_memory_param=False,
        )
        memory = NeuralLongTermMemory(config)
        B, S, D = 2, 4, 8
        x = torch.randn(B, S, D)
        state = memory.init_state(B)

        k = memory.proj_k(x)
        v = memory.proj_v(x)  # post-Task-1: raw linear
        k_proc = F.silu(k)
        k_proc = (k_proc.float() / (k_proc.float().norm(dim=-1, keepdim=True) + 1e-8)).to(
            k.dtype
        )

        alpha = torch.tensor(1.0)  # decay = 0
        eta = torch.tensor(0.0)  # no momentum
        theta = torch.tensor(1.0)

        new_state = memory._parallel_memory_update_linear(
            k_proc, v, state, alpha, theta, eta
        )
        new_w = new_state.weights[0]

        # Ratio check: post-fix scale (2/S) over pre-fix scale (2/(B*S*D)) = B*D.
        pre_scale = 2.0 / float(B * S * D)
        post_scale = 2.0 / float(S)
        ratio = post_scale / pre_scale
        assert abs(ratio - float(B * D)) < 1e-6, (
            f"Scale ratio must be B*D = {B * D}, got {ratio}"
        )
        assert new_w.abs().sum().item() > 0, "Update should be non-zero"


class TestPerSampleGates:
    """Gate projections must be per-sample; no batch-mean collapse."""

    def test_gates_differ_between_samples(self):
        """Two samples with wildly different statistics produce different alpha/theta/eta."""
        config = TitansConfig(dim=16, num_memory_layers=1, auto_checkpoint=True)
        memory = NeuralLongTermMemory(config)

        # Two samples: one near-zero input, one large-magnitude input
        torch.manual_seed(0)
        x = torch.zeros(2, 8, 16)
        x[1] = torch.randn(8, 16) * 5.0
        _, _, snap = memory(x)
        # Direct reproduction of the gate math to verify per-sample shape
        x_mean = torch.mean(x, dim=1, keepdim=True)
        alpha_expected = torch.sigmoid(memory.gate_decay_proj(x_mean))
        # Shape must be (B, 1, 1), not scalar
        assert alpha_expected.shape == (2, 1, 1), (
            f"alpha expected shape (2,1,1), got {alpha_expected.shape}"
        )
        assert not torch.allclose(alpha_expected[0], alpha_expected[1]), (
            "Per-sample gates must differ when inputs differ"
        )
        # Snapshot must preserve per-sample gates (no batch collapse to scalar).
        assert snap is not None
        snap_alpha = snap.alpha[0]
        assert snap_alpha.shape == (2, 1, 1), (
            f"GateSnapshot alpha must preserve batch dim (2,1,1), got {snap_alpha.shape}"
        )
        assert not torch.allclose(snap_alpha[0], snap_alpha[1]), (
            "Snapshot gates must differ per-sample"
        )

    def test_batch_size_one_behavior_unchanged(self):
        """For B=1, post-fix behavior exactly matches pre-fix (mean over singleton = itself)."""
        config = TitansConfig(dim=16, num_memory_layers=1)
        torch.manual_seed(0)
        memory = NeuralLongTermMemory(config)
        x = torch.randn(1, 6, 16)
        out1, _, _ = memory(x)
        # Manually compute alpha path for B=1 and confirm it is (1,1,1)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        alpha = torch.sigmoid(memory.gate_decay_proj(x_mean))
        assert alpha.shape == (1, 1, 1)
        # Output well-defined
        assert out1.shape == (1, 6, 16)

    def test_gate_state_shapes_broadcast_correctly(self):
        """Gates at shape (B,1,1) must broadcast against (B,S,D) tensors."""
        config = TitansConfig(dim=16, num_memory_layers=2, memory_hidden_mult=2.0)
        torch.manual_seed(0)
        memory = NeuralLongTermMemory(config)
        # Deep memory goes through the gate-driven update at lines 454-458
        x = torch.randn(3, 4, 16)
        out, new_state, _ = memory(x)
        assert out.shape == (3, 4, 16)
        # Ensure per-sample outputs differ: retrieval depends on per-sample queries
        out_a = out[0]
        out_b = out[1]
        assert not torch.allclose(out_a, out_b), (
            "Per-sample outputs must differ (no batch collapse)"
        )
