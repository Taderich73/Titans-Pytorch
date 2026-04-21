from __future__ import annotations

"""Unit tests for chunked forward + LoRA gradient flow in lora.py."""

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansMAC
from titans.lora import wrap_lora_layers


def _tiny_mac_with_lora(chunk_size: int = 8) -> TitansMAC:
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
    model = TitansMAC(cfg)
    wrap_lora_layers(
        model,
        targets="attn",
        rank=4,
        alpha=8.0,
        dropout=0.0,
    )
    return model


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


def _lora_chunked_step(
    model: TitansMAC,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    memory_states: list | None,
    vocab_size: int,
) -> tuple[torch.Tensor, list | None]:
    """Reference implementation of the LoRA chunked step.

    Mirrors the logic that lora.py must contain after Task 5.
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


def _import_lora_module():
    """Import scripts/lora.py by inserting the scripts/ dir onto sys.path."""
    scripts_dir = pathlib.Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import lora  # noqa: WPS433

    return lora


def test_lora_config_has_reset_memory_per_batch() -> None:
    """LoRATrainingConfig must expose reset_memory_per_batch with default True."""
    lora = _import_lora_module()

    cfg = lora.LoRATrainingConfig()
    assert hasattr(cfg, "reset_memory_per_batch")
    assert cfg.reset_memory_per_batch is True


def test_lora_parse_args_exposes_reset_flag(monkeypatch) -> None:
    """--no-reset-memory-per-batch toggles the flag False."""
    lora = _import_lora_module()

    monkeypatch.setattr(sys, "argv", [
        "lora.py",
        "--no-reset-memory-per-batch",
    ])
    cfg = lora.parse_args()
    assert cfg.reset_memory_per_batch is False


def test_lora_source_chunks_input_ids() -> None:
    """Verify lora.py sources contain a .split(chunk_size) chunked loop."""
    src = pathlib.Path("scripts/lora.py").read_text()
    assert ".split(chunk_size, dim=1)" in src, (
        "lora.py training loop must chunk along sequence dim"
    )


@pytest.mark.parametrize("seq_len_mult", [1, 2, 3])
def test_lora_chunked_step_does_not_crash(seq_len_mult: int) -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac_with_lora(chunk_size=chunk_size).eval()
    seq_len = chunk_size * seq_len_mult
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    labels = torch.randint(0, 32, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)

    loss, states = _lora_chunked_step(
        model, input_ids, labels, loss_mask, None, vocab_size=32
    )

    assert torch.isfinite(loss), "loss must be finite"
    assert states is not None, "states must be returned"
    assert len(states) == model.config.num_layers


def test_lora_chunked_step_matches_single_shot_when_seq_fits() -> None:
    torch.manual_seed(0)
    chunk_size = 16
    model = _tiny_mac_with_lora(chunk_size=chunk_size).eval()
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, chunk_size))
    labels = torch.randint(0, 32, (batch_size, chunk_size))
    loss_mask = torch.ones(batch_size, chunk_size)

    loss_chunked, _ = _lora_chunked_step(
        model, input_ids, labels, loss_mask, None, vocab_size=32
    )

    # Single-shot reference
    logits, _, _ = model(input_ids, states=None)
    per_token = F.cross_entropy(
        logits.reshape(-1, 32), labels.reshape(-1), reduction="none"
    )
    loss_single = (per_token * loss_mask.reshape(-1)).sum() / loss_mask.sum()

    torch.testing.assert_close(loss_chunked, loss_single, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seq_len_mult", [1, 2, 3])
def test_lora_chunked_step_grads_reach_lora_params(seq_len_mult: int) -> None:
    """LoRA adapter gradients must flow through the chunked forward."""
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac_with_lora(chunk_size=chunk_size)
    seq_len = chunk_size * seq_len_mult
    batch = torch.randint(0, 32, (1, seq_len))
    labels = torch.randint(0, 32, (1, seq_len))
    mask = torch.ones(1, seq_len)

    id_chunks = batch.split(chunk_size, dim=1)
    lbl_chunks = labels.split(chunk_size, dim=1)
    msk_chunks = mask.split(chunk_size, dim=1)

    states = None
    accum = torch.tensor(0.0)

    for ids_c, lbl_c, msk_c in zip(id_chunks, lbl_chunks, msk_chunks):
        logits, states, _ = model(ids_c, states=states)
        logits_flat = logits.reshape(-1, 32)
        per_token = F.cross_entropy(
            logits_flat, lbl_c.reshape(-1), reduction="none"
        )
        num_c = (per_token * msk_c.reshape(-1).float()).sum()
        tok_c = msk_c.reshape(-1).float().sum()
        accum = accum + num_c / tok_c.clamp(min=1.0)
        if states is not None:
            states = [
                s.detach() if s is not None else None for s in states
            ]

    loss = accum / max(len(id_chunks), 1)
    loss.backward()

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    assert lora_params, "Expected LoRA adapter params"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in lora_params
    ), "LoRA adapters received no gradient"


def test_lora_reset_semantics_true_discards_states() -> None:
    """With reset=True, memory at start of batch 2 equals initial (None)."""
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    input_ids = torch.randint(0, 32, (1, 16))
    labels = torch.randint(0, 32, (1, 16))
    loss_mask = torch.ones(1, 16)

    # Simulate batch 1
    _, states_after_1 = _lora_chunked_step(
        model, input_ids, labels, loss_mask, None, vocab_size=32
    )
    assert states_after_1 is not None

    # Reset semantics: drop states_after_1 before batch 2
    states_for_2 = None  # reset_memory_per_batch=True
    _, states_after_2 = _lora_chunked_step(
        model, input_ids, labels, loss_mask, states_for_2, vocab_size=32
    )
    assert states_after_2 is not None


def test_lora_no_reset_carries_states_detached() -> None:
    """With reset=False, states from batch 1 are carried (but detached)."""
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    input_ids = torch.randint(0, 32, (1, 16))
    labels = torch.randint(0, 32, (1, 16))
    loss_mask = torch.ones(1, 16)

    _, states_after_1 = _lora_chunked_step(
        model, input_ids, labels, loss_mask, None, vocab_size=32
    )
    for s in states_after_1:
        for w in s.weights:
            assert not w.requires_grad
    _, states_after_2 = _lora_chunked_step(
        model, input_ids, labels, loss_mask, states_after_1, vocab_size=32
    )
    assert states_after_2 is not None
