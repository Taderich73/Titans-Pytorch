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
    """Rollout helpers must implement prefill->buffer->commit over chunks.

    After Plan 8 Task 13 the prefill/commit machinery lives in the
    ``rollout_samples`` helper (plus the two module-level ``_prefill_prompt``
    and ``_decode_from_committed`` helpers it composes).
    ``generate_rollouts`` is a thin delegator.  The inspection checks the
    combined source spans both the delegator and the helper(s).
    """
    import pathlib

    src = pathlib.Path(_RLVR_PATH).read_text()
    # Scan from the first prefill helper through the legacy delegator.
    gen_start = src.index("def _prefill_prompt(")
    gen_end = src.index("\ndef decode_tokens(")
    body = src[gen_start:gen_end]

    assert "committed_states" in body, (
        "rollout helpers must commit memory at chunk boundaries "
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


def test_generate_rollouts_no_double_commit_of_partial_prompt_tail() -> None:
    """Regression: partial prompt tail must not commit during prefill.

    Before the fix, prefill iterated ``input_ids.split(chunk_size)`` which
    committed the final partial chunk into memory. The re-feed then fed the
    same tokens again, and once decoding produced enough tokens to complete
    the chunk, those prompt-tail tokens were committed a second time.

    We verify that after prefill, ``committed_states`` reflects ONLY the
    full prompt chunks — equal to what we get by prefilling just the
    full-chunk prefix. Any deviation indicates the partial tail leaked into
    memory during prefill.
    """
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()

    batch_size = 1
    # Prompt with a partial tail: 2 full chunks + 3 extra tokens.
    prompt_len = chunk_size * 2 + 3
    prompt = torch.randint(1, 32, (batch_size, prompt_len))
    full_chunk_end = (prompt_len // chunk_size) * chunk_size

    # Reference: commit only the full-chunk prefix.
    with torch.no_grad():
        ref_states = None
        for i in range(prompt_len // chunk_size):
            chunk = prompt[:, i * chunk_size : (i + 1) * chunk_size]
            _, ref_states, _ = model(chunk, states=ref_states)
            ref_states = [
                s.detach() if s is not None else None for s in ref_states
            ]

    # Capture the ``committed_states`` that generate_rollouts builds during
    # prefill by patching model.__call__ and snapshotting state right after
    # the prefill loop finishes. Simplest: monkey-patch the model to record
    # every call's input length, then verify the prefill only invoked the
    # model with full-chunk inputs (plus at most one partial re-feed with
    # states=committed_states ~ ref_states).
    def _states_equal(a_states, b_states) -> bool:  # type: ignore[no-untyped-def]
        """Compare two lists of MemoryState (or None) entries."""
        if a_states is None and b_states is None:
            return True
        if a_states is None or b_states is None:
            return False
        if len(a_states) != len(b_states):
            return False
        for a, b in zip(a_states, b_states):
            if a is None and b is None:
                continue
            if a is None or b is None:
                return False
            # MemoryState has .weights (list[Tensor]) and .momentum (list[Tensor]).
            if len(a.weights) != len(b.weights):
                return False
            for wa, wb in zip(a.weights, b.weights):
                if wa.shape != wb.shape or not torch.allclose(wa, wb, atol=1e-6):
                    return False
            if len(a.momentum) != len(b.momentum):
                return False
            for ma, mb in zip(a.momentum, b.momentum):
                if ma.shape != mb.shape or not torch.allclose(ma, mb, atol=1e-6):
                    return False
        return True

    calls: list[tuple[int, bool]] = []  # (seq_len, committed_states_matches_ref)
    orig_forward = model.forward

    def recording_forward(input_ids, states=None, **kwargs):  # type: ignore[no-untyped-def]
        seq_len = input_ids.shape[1]
        matches_ref = _states_equal(states, ref_states)
        calls.append((seq_len, matches_ref))
        return orig_forward(input_ids, states=states, **kwargs)

    model.forward = recording_forward  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            generate_rollouts(
                model,
                prompt,
                max_new_tokens=1,  # one step of decode is enough
                temperature=0.7,
                num_samples=1,
                eos_token_id=None,
                pad_token_id=0,
            )
    finally:
        model.forward = orig_forward  # type: ignore[method-assign]

    # The prefill phase must consist of exactly:
    #   - (prompt_len // chunk_size) full-chunk calls with seq_len == chunk_size
    #   - one partial-tail re-feed with seq_len == tail_len and
    #     states=committed_states that equals the full-chunks-only ref_states
    # The buggy code instead produced an extra prefill call: it iterated
    # input_ids.split(chunk_size) (yielding full chunks AND the partial tail),
    # committing the tail, then re-fed the tail a second time. We detect the
    # bug in two independent ways:
    #
    #   1. Total prefill-call count. After one decode step the call log is:
    #          fix:   [8]*n_full + [tail] + [tail+1]
    #          buggy: [8]*n_full + [tail] + [tail]   + [tail+1]
    #      So (total_calls - 1) after decoding one token equals
    #      n_full + 1 in the fix, and n_full + 2 in the buggy version.
    #   2. At the *prefill* tail re-feed (the last call with seq_len==tail_len
    #      during prefill), states must equal ref_states (full-chunks-only).
    #      In the buggy version states include the tail's commit from the
    #      previous call.
    tail_len = prompt_len - full_chunk_end
    assert tail_len > 0 and tail_len < chunk_size, "test assumes partial tail"

    n_full = prompt_len // chunk_size
    expected_prefill_calls = n_full + 1  # full chunks + one tail re-feed
    # We asked for max_new_tokens=1, which adds exactly one post-prefill call.
    expected_total = expected_prefill_calls + 1
    assert len(calls) == expected_total, (
        f"Expected {expected_total} model calls for n_full={n_full} + "
        f"1 tail re-feed + 1 decode, got {len(calls)}: {calls}. Extra calls "
        f"indicate the partial prompt tail was committed during prefill "
        f"(split iteration) AND re-fed (the double-commit bug)."
    )

    # Sanity: first n_full calls are full chunks.
    for seq_len, _ in calls[:n_full]:
        assert seq_len == chunk_size, (
            f"Prefill must only commit full chunks; got seq_len={seq_len}"
        )
    # The tail re-feed (call index n_full) must see ref_states.
    tail_call_seq_len, tail_call_matches = calls[n_full]
    assert tail_call_seq_len == tail_len, (
        f"Prefill tail re-feed must have seq_len={tail_len}; "
        f"got {tail_call_seq_len}. Calls: {calls}"
    )
    assert tail_call_matches, (
        "Partial-tail re-feed during prefill must receive committed_states "
        "equal to full-chunks-only prefill. Mismatch means the partial tail "
        "was committed into memory during prefill (the double-commit bug)."
    )


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
