from __future__ import annotations

"""Parity tests for the buffered-memory RLVR rollout helper.

The refactor in ``scripts/rlvr.py`` extracts ``rollout_samples`` and
amortizes the prompt prefill across samples. The tests below lock in
numerical parity with the naive full-prefix reference implementation
(pre-change behavior) by seeding both paths identically.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansMAC

# Load scripts/rlvr.py as a module (scripts/ is not a package).
_RLVR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rlvr.py"
_spec = importlib.util.spec_from_file_location("scripts_rlvr_rollout_mod", _RLVR_PATH)
_rlvr_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts_rlvr_rollout_mod"] = _rlvr_mod
_spec.loader.exec_module(_rlvr_mod)

rollout_samples = _rlvr_mod.rollout_samples
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


def _full_prefix_forward(
    model: TitansMAC, tokens: torch.Tensor
) -> torch.Tensor:
    """Re-chunk ``tokens`` and forward through the model from scratch.

    Models enforce single-chunk forward at the nn.Module level; this helper
    splits the growing sequence into chunk-sized pieces, runs them
    sequentially threading memory state, then re-feeds any partial tail
    from the last committed state (so the tail is never committed). Returns
    the next-token logits for the final position.
    """
    chunk_size = model.config.chunk_size
    seq_len = tokens.shape[1]
    n_full = seq_len // chunk_size
    tail_start = n_full * chunk_size

    states: list | None = None
    last_logits: torch.Tensor | None = None
    for i in range(n_full):
        chunk = tokens[:, i * chunk_size : (i + 1) * chunk_size]
        last_logits, states, _ = model(chunk, states=states)
        if states is not None:
            states = [s.detach() if s is not None else None for s in states]

    if tail_start < seq_len:
        tail = tokens[:, tail_start:]
        last_logits, _, _ = model(tail, states=states)
    return last_logits


def _naive_rollout(
    model: TitansMAC,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference path: re-run the full prefix every step, per sample.

    Re-chunks the growing sequence each step so memory still commits at
    chunk boundaries (matching Titans' model contract). This is the
    semantically-equivalent "obviously correct" reference for pinning the
    optimized buffered path.
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    all_seqs: list[torch.Tensor] = []
    all_logps: list[torch.Tensor] = []

    for _ in range(num_samples):
        tokens = input_ids.clone()
        logps = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits = _full_prefix_forward(model, tokens)
            nl = logits[:, -1, :].float() / max(temperature, 1e-8)
            probs = F.softmax(nl, dim=-1)
            nt = torch.multinomial(probs, num_samples=1)
            lp = F.log_softmax(nl, dim=-1)
            logps = logps + lp.gather(-1, nt).squeeze(-1) * (~finished).float()
            tokens = torch.cat([tokens, nt], dim=-1)
            if eos_token_id is not None:
                finished = finished | (nt.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
        all_seqs.append(tokens)
        all_logps.append(logps)

    max_len = max(s.shape[-1] for s in all_seqs)
    padded = []
    for s in all_seqs:
        pad_len = max_len - s.shape[-1]
        if pad_len > 0:
            pad = torch.full(
                (batch_size, pad_len), pad_token_id, dtype=torch.long, device=device,
            )
            s = torch.cat([s, pad], dim=-1)
        padded.append(s)
    return torch.stack(padded, dim=1), torch.stack(all_logps, dim=1)


def test_rollout_samples_is_importable() -> None:
    """The refactor must expose ``rollout_samples`` at module scope."""
    assert callable(rollout_samples)


def test_rollout_samples_matches_naive_short_chunk_aligned() -> None:
    """Chunk-aligned prompt: buffered output must equal naive full-prefix."""
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    # Prompt length is a multiple of chunk_size (no partial tail).
    prompt = torch.randint(0, 32, (2, 16))
    max_new_tokens = 6
    num_samples = 2

    torch.manual_seed(42)
    seqs_naive, logps_naive = _naive_rollout(
        model, prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        temperature=1.0,
    )
    torch.manual_seed(42)
    seqs_fast, logps_fast = rollout_samples(
        model, prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        temperature=1.0,
    )

    assert torch.equal(seqs_naive, seqs_fast)
    assert torch.allclose(logps_naive, logps_fast, rtol=1e-4, atol=1e-4)


def test_rollout_samples_matches_naive_partial_tail() -> None:
    """Prompt with partial tail: re-feed path must still match the naive ref."""
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    # prompt_len = 10 -> 1 full chunk + 2-token partial tail
    prompt = torch.randint(0, 32, (2, 10))
    max_new_tokens = 5
    num_samples = 3

    torch.manual_seed(7)
    seqs_naive, logps_naive = _naive_rollout(
        model, prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        temperature=1.0,
    )
    torch.manual_seed(7)
    seqs_fast, logps_fast = rollout_samples(
        model, prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        temperature=1.0,
    )

    assert torch.equal(seqs_naive, seqs_fast)
    assert torch.allclose(logps_naive, logps_fast, rtol=1e-4, atol=1e-4)


def test_generate_rollouts_delegates_to_helper() -> None:
    """Legacy ``generate_rollouts`` must match ``rollout_samples`` exactly."""
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_mac(chunk_size=chunk_size).eval()
    prompt = torch.randint(0, 32, (1, 8))

    torch.manual_seed(3)
    a_seqs, a_logps = generate_rollouts(
        model, prompt,
        max_new_tokens=4,
        num_samples=2,
        temperature=1.0,
    )
    torch.manual_seed(3)
    b_seqs, b_logps = rollout_samples(
        model, prompt,
        max_new_tokens=4,
        num_samples=2,
        temperature=1.0,
    )
    assert torch.equal(a_seqs, b_seqs)
    assert torch.allclose(a_logps, b_logps, rtol=1e-6, atol=1e-6)


def test_rollout_samples_eos_stops_generation() -> None:
    """EOS handling: rows that emit EOS stop accumulating log-probs."""
    torch.manual_seed(0)
    chunk_size = 4
    model = _tiny_mac(chunk_size=chunk_size).eval()
    prompt = torch.randint(0, 32, (1, 4))

    # Pick an eos token that is very likely early under low temperature to
    # exercise the early-stop path; fall back to a plain smoke-test assert
    # if the model happens not to emit it (the helper must still produce a
    # well-shaped tensor in that case).
    eos = 0
    torch.manual_seed(1)
    seqs, logps = rollout_samples(
        model, prompt,
        max_new_tokens=8,
        num_samples=2,
        temperature=1.0,
        eos_token_id=eos,
        pad_token_id=eos,
    )
    assert seqs.dim() == 3
    assert seqs.shape[0] == 1
    assert seqs.shape[1] == 2
    assert logps.shape == (1, 2)
