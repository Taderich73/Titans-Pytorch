"""Unit tests for ``titans.scripts._common.chunked_forward``."""

from __future__ import annotations

import torch

from titans import TitansConfig, TitansMAC


def _tiny_mac(chunk_size: int = 8) -> TitansMAC:
    """Build a tiny TitansMAC suitable for fast CPU tests."""
    cfg = TitansConfig(
        vocab_size=32,
        dim=16,
        num_heads=2,
        num_layers=2,
        chunk_size=chunk_size,
        window_size=chunk_size,
        num_persistent_tokens=2,
        num_memory_layers=1,
    )
    return TitansMAC(cfg)


def test_chunked_forward_yields_per_chunk() -> None:
    """Generator yields one tuple per chunk with expected shapes."""
    from titans.scripts import chunked_forward

    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    input_ids = torch.randint(0, 32, (2, chunk_size * 3))

    chunks = list(chunked_forward(model, input_ids, chunk_size, states=None))
    assert len(chunks) == 3
    for logits, chunk_ids, _states in chunks:
        assert logits.shape == (2, chunk_size, 32)
        assert chunk_ids.shape == (2, chunk_size)


def test_chunked_forward_ragged_last_chunk() -> None:
    """Final chunk is shorter when seq_len is not a multiple of chunk_size."""
    from titans.scripts import chunked_forward

    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    input_ids = torch.randint(0, 32, (1, chunk_size * 2 + 3))

    chunks = list(chunked_forward(model, input_ids, chunk_size, states=None))
    assert len(chunks) == 3
    assert chunks[0][1].shape == (1, chunk_size)
    assert chunks[1][1].shape == (1, chunk_size)
    assert chunks[2][1].shape == (1, 3)


def test_chunked_forward_detaches_between_chunks() -> None:
    """With detach_between=True the post-chunk state tensors are detached."""
    from titans.scripts import chunked_forward

    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    input_ids = torch.randint(0, 32, (1, chunk_size * 2))

    chunks = list(
        chunked_forward(model, input_ids, chunk_size, states=None, detach_between=True)
    )
    # Every chunk's yielded state must be detached.
    for _, _, chunk_states in chunks:
        assert chunk_states is not None
        for s in chunk_states:
            if s is None:
                continue
            for w in s.weights:
                assert not w.requires_grad


def test_chunked_forward_preserves_graph_when_detach_false() -> None:
    """With detach_between=False logits stay connected to autograd."""
    from titans.scripts import chunked_forward

    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size)  # train mode
    input_ids = torch.randint(0, 32, (1, chunk_size * 2))

    chunks = list(
        chunked_forward(model, input_ids, chunk_size, states=None, detach_between=False)
    )
    # First chunk's logits must require grad so loss.backward() is valid.
    logits0, _, _ = chunks[0]
    assert logits0.requires_grad


def test_chunked_forward_single_chunk_when_seq_eq_chunk() -> None:
    """seq_len == chunk_size yields exactly one chunk."""
    from titans.scripts import chunked_forward

    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    input_ids = torch.randint(0, 32, (1, chunk_size))

    chunks = list(chunked_forward(model, input_ids, chunk_size, states=None))
    assert len(chunks) == 1
    assert chunks[0][1].shape == (1, chunk_size)


def test_chunked_forward_threads_states_across_chunks() -> None:
    """Threaded chunked forward matches a single full-sequence forward."""
    from titans.scripts import chunked_forward

    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    input_ids = torch.randint(0, 32, (1, chunk_size * 2))

    # Single-shot reference (one full chunk pair through the model).
    with torch.no_grad():
        logits_ref_1, states_1, _ = model(input_ids[:, :chunk_size], states=None)
        logits_ref_2, states_2, _ = model(input_ids[:, chunk_size:], states=states_1)

        chunks = list(
            chunked_forward(
                model,
                input_ids,
                chunk_size,
                states=None,
                detach_between=False,
            )
        )

    assert len(chunks) == 2
    torch.testing.assert_close(chunks[0][0], logits_ref_1)
    torch.testing.assert_close(chunks[1][0], logits_ref_2)
