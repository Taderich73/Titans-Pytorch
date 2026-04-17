"""End-to-end integration tests for TNT paper-alignment fixes."""

from __future__ import annotations

import torch

from titans.config import TitansConfig
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
