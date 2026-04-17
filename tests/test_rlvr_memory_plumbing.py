from __future__ import annotations

"""Tests for memory-aware log-prob paths in RLVR."""

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansMAC

# Load scripts/rlvr.py as a module (scripts/ is not a package).
_RLVR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rlvr.py"
_spec = importlib.util.spec_from_file_location("scripts_rlvr_mod", _RLVR_PATH)
_rlvr_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts_rlvr_mod"] = _rlvr_mod
_spec.loader.exec_module(_rlvr_mod)

compute_token_log_probs = _rlvr_mod.compute_token_log_probs
compute_log_probs_for_generated = _rlvr_mod.compute_log_probs_for_generated
generate_rollouts = _rlvr_mod.generate_rollouts


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


def test_compute_token_log_probs_accepts_states() -> None:
    """compute_token_log_probs must accept a states= parameter after Task 8."""
    sig = inspect.signature(compute_token_log_probs)
    assert "states" in sig.parameters


def test_compute_token_log_probs_returns_states() -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    seq_len = chunk_size * 3
    input_ids = torch.randint(0, 32, (2, seq_len))

    with torch.no_grad():
        out = compute_token_log_probs(model, input_ids, vocab_size=32, states=None)

    assert isinstance(out, tuple)
    token_lp, final_states = out
    assert token_lp.shape == (2, seq_len - 1)
    assert final_states is not None


def test_compute_token_log_probs_matches_single_shot_when_fits() -> None:
    """When seq_len <= chunk_size, chunked path must match single-shot."""
    torch.manual_seed(0)
    chunk_size = 16
    model = _tiny_mac(chunk_size=chunk_size).eval()
    seq_len = chunk_size
    input_ids = torch.randint(0, 32, (2, seq_len))

    with torch.no_grad():
        token_lp_chunked, _ = compute_token_log_probs(
            model, input_ids, vocab_size=32, states=None
        )

        logits, _, _ = model(input_ids, states=None)
        logits = logits.float()
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        targets = input_ids[:, 1:].clamp(min=0, max=31)
        token_lp_ref = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(
        token_lp_chunked, token_lp_ref, rtol=1e-5, atol=1e-5
    )


def test_compute_token_log_probs_no_crash_seq_gt_chunk() -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    seq_len = chunk_size * 4 + 3
    input_ids = torch.randint(0, 32, (2, seq_len))

    with torch.no_grad():
        token_lp, _ = compute_token_log_probs(
            model, input_ids, vocab_size=32, states=None
        )

    assert token_lp.shape == (2, seq_len - 1)
    assert torch.isfinite(token_lp).all()


def test_compute_log_probs_for_generated_passes_states_through() -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    prompt_len = chunk_size
    completion_len = chunk_size * 2
    batch_size = 1
    num_samples = 2
    prompt_ids = torch.randint(0, 32, (batch_size, prompt_len))
    generated_ids = torch.randint(
        0, 32, (batch_size, num_samples, prompt_len + completion_len)
    )
    generated_ids[:, :, :prompt_len] = prompt_ids.unsqueeze(1)

    with torch.no_grad():
        logps = compute_log_probs_for_generated(
            model, prompt_ids, generated_ids, vocab_size=32
        )

    assert logps.shape == (batch_size, num_samples)
    assert torch.isfinite(logps).all()


def test_generate_rollouts_uses_prefill_pattern() -> None:
    """generate_rollouts must implement prefill->buffer->commit over chunks."""
    import pathlib

    src = pathlib.Path(_RLVR_PATH).read_text()
    # After Task 9, generate_rollouts should have a committed_states concept
    # matching inference.py:102-153.
    gen_start = src.index("def generate_rollouts(")
    gen_end = src.index("\ndef ", gen_start + 1)
    body = src[gen_start:gen_end]

    assert "committed_states" in body, (
        "generate_rollouts must commit memory at chunk boundaries "
        "(mirror inference.py:102-153)"
    )
    assert "buffer_start" in body, "Must track decode buffer start position"


def test_generate_rollouts_produces_expected_shape() -> None:
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    batch_size = 1
    prompt_len = chunk_size
    max_new = 4
    num_samples = 2
    prompt = torch.randint(1, 32, (batch_size, prompt_len))

    generated_ids, completion_lp = generate_rollouts(
        model,
        prompt,
        max_new_tokens=max_new,
        temperature=0.7,
        num_samples=num_samples,
        eos_token_id=None,
        pad_token_id=0,
    )

    assert generated_ids.shape[0] == batch_size
    assert generated_ids.shape[1] == num_samples
    assert generated_ids.shape[2] >= prompt_len + max_new
    assert completion_lp.shape == (batch_size, num_samples)


def test_generate_rollouts_long_prompt_does_not_crash() -> None:
    """Prompt longer than chunk_size must chunk during prefill."""
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    batch_size = 1
    prompt_len = chunk_size * 3 + 2  # spans 4 chunks
    prompt = torch.randint(1, 32, (batch_size, prompt_len))

    generated_ids, completion_lp = generate_rollouts(
        model,
        prompt,
        max_new_tokens=4,
        temperature=0.7,
        num_samples=1,
    )

    assert torch.isfinite(completion_lp).all()


def test_rlvr_config_has_reset_memory_per_batch() -> None:
    RLVRConfig = _rlvr_mod.RLVRConfig

    cfg = RLVRConfig()
    assert hasattr(cfg, "reset_memory_per_batch")
    assert cfg.reset_memory_per_batch is True


def test_rlvr_source_gates_memory_save() -> None:
    import pathlib

    src = pathlib.Path(_RLVR_PATH).read_text()
    # After Task 10 the save path references reset_memory_per_batch and
    # calls save_memory_states from titans.memory_dump.
    assert "reset_memory_per_batch" in src, (
        "rlvr.py must consult reset_memory_per_batch for memory save"
    )
    assert "save_memory_states" in src, (
        "rlvr.py must import and call save_memory_states"
    )


def test_rlvr_memory_roundtrip(tmp_path) -> None:
    """Simulate the rlvr.py save/load pattern and verify tensors roundtrip."""
    from titans.memory import MemoryState
    from titans.memory_dump import load_memory_states, save_memory_states

    ckpt_dir = tmp_path / "rlvr"
    ckpt_dir.mkdir()
    memory_states = [
        MemoryState(
            weights=[torch.arange(16).reshape(4, 4).float()],
            momentum=[torch.zeros(4, 4)],
        )
    ]
    global_step = 500
    save_memory_states(
        memory_states, ckpt_dir / f"memory_step_{global_step}.npz"
    )
    loaded = load_memory_states(
        ckpt_dir / f"memory_step_{global_step}.npz",
        device=torch.device("cpu"),
    )
    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0].weights[0], memory_states[0].weights[0])
