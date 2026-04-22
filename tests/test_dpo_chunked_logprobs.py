from __future__ import annotations

"""Tests for memory-aware chunked log-prob computation in DPO."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansMAC

# Load scripts/dpo.py as a module (scripts/ is not a package).
_DPO_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo.py"
_spec = importlib.util.spec_from_file_location("scripts_dpo_mod", _DPO_PATH)
_dpo_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts_dpo_mod"] = _dpo_mod
_spec.loader.exec_module(_dpo_mod)

compute_log_probs = _dpo_mod.compute_log_probs


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


def test_compute_log_probs_accepts_states_kwarg() -> None:
    """compute_log_probs must accept a states= parameter after Task 6."""
    import inspect

    sig = inspect.signature(compute_log_probs)
    assert "states" in sig.parameters


def test_compute_log_probs_returns_final_states() -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    # Fused (prompt, response) sequence length > chunk_size
    seq_len = chunk_size * 3
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    labels = torch.randint(0, 32, (batch_size, seq_len))
    # Prompt = first chunk (mask=0), response = rest (mask=1)
    loss_mask = torch.zeros(batch_size, seq_len)
    loss_mask[:, chunk_size:] = 1.0

    with torch.no_grad():
        sum_logps, lengths, final_states = compute_log_probs(
            model,
            input_ids,
            labels,
            loss_mask,
            vocab_size=32,
            states=None,
        )

    assert sum_logps.shape == (batch_size,)
    assert lengths.shape == (batch_size,)
    assert torch.isfinite(sum_logps).all()
    assert final_states is not None


def test_compute_log_probs_matches_single_shot_when_seq_fits() -> None:
    """When seq_len <= chunk_size, chunked path must match single-shot."""
    torch.manual_seed(0)
    chunk_size = 16
    model = _tiny_mac(chunk_size=chunk_size).eval()
    batch_size = 2
    seq_len = chunk_size
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    labels = torch.randint(0, 32, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        # New chunked path
        sum_logps_chunked, _, _ = compute_log_probs(
            model,
            input_ids,
            labels,
            loss_mask,
            vocab_size=32,
            states=None,
        )

        # Single-shot reference
        logits, _, _ = model(input_ids, states=None)
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        labels_clamped = labels.clamp(min=0, max=31)
        token_lp = log_probs.gather(-1, labels_clamped.unsqueeze(-1)).squeeze(-1)
        sum_logps_ref = (token_lp * loss_mask).sum(dim=-1)

    torch.testing.assert_close(sum_logps_chunked, sum_logps_ref, rtol=1e-5, atol=1e-5)


def test_compute_log_probs_no_crash_seq_gt_chunk() -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    seq_len = chunk_size * 4 + 3
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    labels = torch.randint(0, 32, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        sum_logps, lengths, _ = compute_log_probs(
            model,
            input_ids,
            labels,
            loss_mask,
            vocab_size=32,
            states=None,
        )

    assert torch.isfinite(sum_logps).all()
    assert (lengths == seq_len).all()


def test_chosen_vs_rejected_differ_after_chunking() -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    prompt_len = chunk_size
    cont_len = chunk_size * 2
    seq_len = prompt_len + cont_len
    prompt = torch.randint(0, 32, (1, prompt_len))
    chosen = torch.randint(0, 32, (1, cont_len))
    rejected = torch.randint(0, 32, (1, cont_len))

    # Build two fused sequences sharing the prompt
    chosen_ids = torch.cat([prompt, chosen], dim=1)
    rejected_ids = torch.cat([prompt, rejected], dim=1)
    # Labels are the same shape as input_ids (per existing dpo.py convention).
    chosen_labels = chosen_ids.clone()
    rejected_labels = rejected_ids.clone()
    mask = torch.zeros(1, seq_len)
    mask[:, prompt_len:] = 1.0

    with torch.no_grad():
        chosen_lp, _, _ = compute_log_probs(
            model,
            chosen_ids,
            chosen_labels,
            mask,
            vocab_size=32,
            states=None,
        )
        rejected_lp, _, _ = compute_log_probs(
            model,
            rejected_ids,
            rejected_labels,
            mask,
            vocab_size=32,
            states=None,
        )

    assert chosen_lp.shape == rejected_lp.shape == (1,)
    assert torch.isfinite(chosen_lp).all() and torch.isfinite(rejected_lp).all()
    # With distinct continuations, log-probs should generically differ.
    assert (chosen_lp - rejected_lp).abs().item() > 0
