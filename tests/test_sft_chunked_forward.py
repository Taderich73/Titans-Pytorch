from __future__ import annotations

"""Unit tests for chunked forward pass in SFT training step."""

import pytest
import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansMAC


def _tiny_mac(chunk_size: int = 8) -> TitansMAC:
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


def _sft_chunked_step(
    model: TitansMAC,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    memory_states: list | None,
    vocab_size: int,
) -> tuple[torch.Tensor, list | None]:
    """Reference implementation of the SFT chunked step.

    Mirrors the logic that sft.py must contain after Task 1.
    """
    chunk_size = model.config.chunk_size
    id_chunks = input_ids.split(chunk_size, dim=1)
    lbl_chunks = labels.split(chunk_size, dim=1)
    msk_chunks = loss_mask.split(chunk_size, dim=1)

    total_loss_num = torch.tensor(0.0, device=input_ids.device)
    total_tokens = torch.tensor(0.0, device=input_ids.device)
    states = memory_states

    for ids_c, lbl_c, msk_c in zip(id_chunks, lbl_chunks, msk_chunks):
        logits, states, _ = model(ids_c, states=states)
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = lbl_c.reshape(-1)
        mask_flat = msk_c.reshape(-1).float()

        per_token = F.cross_entropy(logits_flat, labels_flat, reduction="none")
        total_loss_num = total_loss_num + (per_token * mask_flat).sum()
        total_tokens = total_tokens + mask_flat.sum()

        if states is not None:
            states = [s.detach() if s is not None else None for s in states]

    loss = total_loss_num / total_tokens.clamp(min=1.0)
    return loss, states


@pytest.mark.parametrize("seq_len_mult", [1, 2, 3])
def test_sft_chunked_step_does_not_crash(seq_len_mult: int) -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    seq_len = chunk_size * seq_len_mult + chunk_size // 2
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    labels = torch.randint(0, 32, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)

    loss, states = _sft_chunked_step(
        model, input_ids, labels, loss_mask, None, vocab_size=32
    )

    assert torch.isfinite(loss), "loss must be finite"
    assert states is not None, "states must be returned"
    # TitansMAC returns one state per block (num_layers), not per memory layer.
    assert len(states) == model.config.num_layers


def test_sft_chunked_step_matches_single_shot_when_seq_fits() -> None:
    torch.manual_seed(0)
    chunk_size = 16
    model = _tiny_mac(chunk_size=chunk_size).eval()
    # seq_len == chunk_size means exactly one chunk; chunked should match single-shot.
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, chunk_size))
    labels = torch.randint(0, 32, (batch_size, chunk_size))
    loss_mask = torch.ones(batch_size, chunk_size)

    loss_chunked, _ = _sft_chunked_step(
        model, input_ids, labels, loss_mask, None, vocab_size=32
    )

    # Single-shot reference
    logits, _, _ = model(input_ids, states=None)
    per_token = F.cross_entropy(
        logits.reshape(-1, 32), labels.reshape(-1), reduction="none"
    )
    loss_single = (per_token * loss_mask.reshape(-1)).sum() / loss_mask.sum()

    torch.testing.assert_close(loss_chunked, loss_single, rtol=1e-5, atol=1e-5)


def test_sft_train_uses_chunked_step() -> None:
    """Verify sft.py sources contain a .split(chunk_size) chunked loop."""
    import pathlib

    src = pathlib.Path("scripts/sft.py").read_text()
    # After Task 1, the training loop must iterate chunks.
    # We grep for the canonical token-splitting pattern.
    assert ".split(chunk_size, dim=1)" in src, (
        "sft.py must chunk input_ids along the sequence dimension "
        "(see scripts/pretrain.py:578)"
    )


def test_sft_evaluate_uses_chunked_loop() -> None:
    """Verify sft.py evaluate() chunks the input."""
    import pathlib

    src = pathlib.Path("scripts/sft.py").read_text()

    # Locate the evaluate function body and assert the chunk split appears
    # after the `def evaluate(` line.
    assert "def evaluate(" in src
    eval_start = src.index("def evaluate(")
    eval_end = src.index("def train(", eval_start)
    eval_body = src[eval_start:eval_end]
    assert ".split(chunk_size, dim=1)" in eval_body, (
        "evaluate() must chunk input_ids"
    )


def test_sft_eval_numerical_equivalence_single_chunk() -> None:
    """When seq_len == chunk_size, chunked eval loss must match single-shot."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    chunk_size = 16
    model = _tiny_mac(chunk_size=chunk_size).eval()
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, chunk_size))
    labels = torch.randint(0, 32, (batch_size, chunk_size))
    loss_mask = torch.ones(batch_size, chunk_size)

    # Single-shot reference
    with torch.no_grad():
        logits, _, _ = model(input_ids, states=None)
        per_token = F.cross_entropy(
            logits.reshape(-1, 32), labels.reshape(-1), reduction="none"
        )
        ref_num = (per_token * loss_mask.reshape(-1)).sum()
        ref_tok = loss_mask.sum()

    # Chunked path via the same helper used in Task 1
    with torch.no_grad():
        loss_chunked, _ = _sft_chunked_step(
            model, input_ids, labels, loss_mask, None, vocab_size=32
        )

    torch.testing.assert_close(
        loss_chunked,
        ref_num / ref_tok,
        rtol=1e-5,
        atol=1e-5,
    )
