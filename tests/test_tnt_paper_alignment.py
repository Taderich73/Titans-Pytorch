"""End-to-end integration tests for TNT paper-alignment fixes."""

from __future__ import annotations

import torch

from titans.config import TitansConfig
from titans.qk_projection import QKProjection
from titans.tnt_memory import LocalMemory


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
