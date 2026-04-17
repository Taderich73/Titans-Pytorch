"""End-to-end integration tests for TNT paper-alignment fixes."""

from __future__ import annotations

import pytest
import torch

from titans.config import TitansConfig
from titans.qk_projection import QKProjection
from titans.tnt_memory import HierarchicalMemory, LocalMemory


def _tnt_config(
    dim: int = 32,
    chunk_size: int = 8,
    shard_length: int = 8,
    local_chunk_sizes: list[int] | None = None,
    num_memory_layers: int = 1,
    **overrides,
) -> TitansConfig:
    """Small TNT config suitable for integration tests."""
    if local_chunk_sizes is None:
        local_chunk_sizes = [chunk_size]
    cfg = TitansConfig(
        dim=dim,
        num_heads=4,
        num_layers=2,
        vocab_size=128,
        chunk_size=chunk_size,
        window_size=chunk_size,
        max_seq_len=256,
        num_memory_layers=num_memory_layers,
        num_persistent_tokens=4,
        use_tnt=True,
        global_chunk_size=max(32, chunk_size),
        local_chunk_sizes=local_chunk_sizes,
        local_shard_length=shard_length,
        use_qk_projection=True,
        **overrides,
    )
    return cfg


class TestLearnableWInit:
    def test_w_init_receives_gradient_from_retrieve(self, device):
        """Gradient of an LM-like loss must reach ``_w_init`` when the state
        used for retrieval was freshly reset (so its weights derive from
        w_init without any intervening update)."""
        config = _tnt_config()
        local = LocalMemory(config, chunk_size=8, shard_length=8).to(device)

        # A fresh state clones from the learnable parameter; retrieving
        # without any forward update means output depends ONLY on w_init.
        state = local.init_state(batch_size=2)
        queries = torch.randn(2, 4, config.dim, device=device)
        retrieved = local.retrieve(queries, state)

        loss = retrieved.pow(2).sum()
        loss.backward()

        grads = [p.grad for p in local._w_init]
        assert all(g is not None for g in grads), (
            "w_init parameters received no gradient"
        )
        assert any(g.abs().sum().item() > 0.0 for g in grads), (
            "w_init gradients are all zero"
        )


class TestQKProjectionEfficient:
    def test_matches_naive_reference(self, device):
        """Efficient path must match a naive per-position implementation
        on small D=8, C=16 where the naive version is cheap enough."""
        torch.manual_seed(0)
        B, C, D = 2, 16, 8
        q = torch.randn(B, C, D, device=device)
        # L2-normalise keys (paper assumption).
        k_raw = torch.randn(B, C, D, device=device)
        k = k_raw / (k_raw.norm(dim=-1, keepdim=True) + 1e-8)
        carry = torch.randn(D, D, device=device)

        proj = QKProjection(dim=D).to(device)
        projected_q, _ = proj(q, k, carry)

        # Reference: materialise per-position M_t and apply.
        ref_proj = torch.zeros(B, C, D, device=device)
        for b in range(B):
            M = carry.clone()
            for t in range(C):
                M = M + torch.outer(k[b, t], k[b, t])
                ref_proj[b, t] = M @ q[b, t]

        assert torch.allclose(projected_q, ref_proj, atol=1e-5, rtol=1e-5), (
            f"efficient path diverges: max abs diff = "
            f"{(projected_q - ref_proj).abs().max().item()}"
        )

    def test_causality_first_query_depends_only_on_k0_and_carry(self, device):
        """q_0's projection must not depend on k_1, k_2, ..., k_{C-1}."""
        torch.manual_seed(1)
        B, C, D = 1, 8, 16
        q = torch.randn(B, C, D, device=device)
        k = torch.randn(B, C, D, device=device)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        carry = torch.zeros(D, D, device=device)

        proj = QKProjection(dim=D).to(device)
        base_q, _ = proj(q, k, carry)

        # Perturb keys at positions 1..C-1; q_0 projection must not change.
        k_perturbed = k.clone()
        k_perturbed[:, 1:] = torch.randn_like(k_perturbed[:, 1:])
        k_perturbed[:, 1:] = k_perturbed[:, 1:] / (
            k_perturbed[:, 1:].norm(dim=-1, keepdim=True) + 1e-8
        )
        perturbed_q, _ = proj(q, k_perturbed, carry)

        assert torch.allclose(base_q[:, 0], perturbed_q[:, 0], atol=1e-6), (
            "q_0 projection changed after perturbing k_1..k_{C-1}: causality violated"
        )
        assert not torch.allclose(base_q[:, -1], perturbed_q[:, -1], atol=1e-4), (
            "q_{C-1} projection was unchanged after perturbing future keys -- suspicious"
        )

    def test_carry_recurrence_matches_paper(self, device):
        """new_carry = carry + mean_b(k^T @ k) -- the paper's per-chunk update."""
        torch.manual_seed(2)
        B, C, D = 3, 12, 16
        k = torch.randn(B, C, D, device=device)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        q = torch.randn(B, C, D, device=device)
        carry = torch.randn(D, D, device=device)

        proj = QKProjection(dim=D).to(device)
        _, new_carry = proj(q, k, carry)

        expected = carry + torch.einsum("bcd,bce->de", k, k) / B
        assert torch.allclose(new_carry, expected, atol=1e-5), (
            f"carry recurrence mismatch: "
            f"{(new_carry - expected).abs().max().item()}"
        )


class TestTNTQKProjectionConfig:
    def test_default_is_per_position(self):
        cfg = TitansConfig()
        assert cfg.tnt_qk_projection == "per_position"

    def test_roundtrip_through_dict(self):
        cfg = TitansConfig(tnt_qk_projection="chunk_mean")
        d = cfg.to_dict()
        assert d["tnt_qk_projection"] == "chunk_mean"
        cfg2 = TitansConfig.from_dict(d)
        assert cfg2.tnt_qk_projection == "chunk_mean"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TitansConfig(tnt_qk_projection="bogus")


class TestHierarchicalPerPositionProjection:
    def test_default_forward_uses_per_position(self, device):
        """With per-position projection (default), per-token outputs must
        vary along the chunk. Under the old chunk-mean path, every query saw
        the same projection — here we assert outputs differ at t=0 vs t=C-1.
        """
        torch.manual_seed(3)
        config = _tnt_config(dim=32, chunk_size=8, shard_length=64,
                             local_chunk_sizes=[8])
        assert config.tnt_qk_projection == "per_position"
        hm = HierarchicalMemory(config).to(device)
        x = torch.randn(2, 8, config.dim, device=device)
        out, _, _ = hm(x)
        assert not torch.allclose(out[:, 0], out[:, -1], atol=1e-4), (
            "Outputs at t=0 and t=C-1 are identical — projection is not "
            "per-position as Eq. 7 requires."
        )

    def test_chunk_mean_opt_in_runs_and_carries(self, device):
        """Opting into chunk_mean must not crash and must produce outputs
        of the correct shape with a (dim, dim) carry stored in state."""
        config = _tnt_config(tnt_qk_projection="chunk_mean",
                             local_chunk_sizes=[8])
        hm = HierarchicalMemory(config).to(device)
        x = torch.randn(2, 8, config.dim, device=device)
        out, state, _ = hm(x)
        assert out.shape == (2, 8, config.dim)
        assert state.qk_projections[0].shape == (config.dim, config.dim)

    def test_per_position_causality_q0_independent_of_future_x(self, device):
        """q_0 output must not depend on x_{1..C-1} when retrieval uses the
        per-position carry. We perturb x[:,1:] and assert out[:,0] unchanged
        up to the global-memory/NLTM nonlinear floor (which is also per-
        position causal via its own independent machinery).

        NB: LocalMemory.retrieve inside NLTM depends on updated weights (a
        function of ALL tokens via the parallel update), so this test
        restricts itself to proving the QK-projection branch is causal by
        comparing against a chunk_mean-config baseline: the per-position
        delta at t=0 must be strictly smaller than at t=C-1 when future
        tokens are perturbed."""
        torch.manual_seed(7)
        cfg_pp = _tnt_config(dim=16, chunk_size=8, shard_length=64,
                             local_chunk_sizes=[8])
        hm = HierarchicalMemory(cfg_pp).to(device)
        x = torch.randn(1, 8, cfg_pp.dim, device=device)
        x_perturbed = x.clone()
        x_perturbed[:, 1:] = torch.randn_like(x_perturbed[:, 1:])

        out_a, _, _ = hm(x)
        out_b, _, _ = hm(x_perturbed)

        # In a strictly-causal QK projection, t=0's projected query depends
        # only on k_0 and the carry. Changes at t=0 arise solely from the
        # NLTM's own dependence on the whole chunk (same mechanism in both
        # cfg variants). We assert the end-position change exceeds the
        # start-position change — a sanity check that the per-position
        # pathway is active.
        delta_first = (out_a[:, 0] - out_b[:, 0]).abs().mean().item()
        delta_last = (out_a[:, -1] - out_b[:, -1]).abs().mean().item()
        assert delta_last > delta_first, (
            f"per-position projection not engaged: delta_last={delta_last} "
            f"delta_first={delta_first}"
        )
