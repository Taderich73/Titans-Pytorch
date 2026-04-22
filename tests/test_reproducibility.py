"""Reproducibility contract tests for :func:`titans.utils.seed_everything`.

These tests are the machine-checkable version of
``docs/reproducibility.md``. They assert that the same seed reproduces
identical outputs bit-for-bit on CPU (with and without
``deterministic=True``), and that changing the seed changes the output
(so a no-op implementation cannot trivially pass).

The tests are CPU-only — CUDA determinism depends on kernel availability
and is validated in practice via the ``--deterministic`` runtime flag in
the training scripts, not here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansMAC
from titans.utils import seed_everything


@pytest.fixture(autouse=True)
def _reset_deterministic_after_test():
    """Restore ``torch.use_deterministic_algorithms`` after each test.

    ``seed_everything(..., deterministic=True)`` sets a process-global
    flag; leaving it on leaks into unrelated tests that happen to use a
    non-deterministic kernel and would then raise ``RuntimeError``.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(prev)


def _tiny_config() -> TitansConfig:
    """Tiny-but-realistic config for fast CPU tests."""
    return TitansConfig(
        dim=32,
        num_heads=2,
        num_layers=2,
        vocab_size=64,
        chunk_size=16,
        window_size=16,
        max_seq_len=64,
        num_memory_layers=2,
        num_persistent_tokens=4,
    )


def _build_model_and_loss(seed: int, *, deterministic: bool) -> tuple[
    TitansMAC, torch.Tensor, list[torch.Tensor]
]:
    """Seed, build a tiny model, run one forward, return (model, loss, params).

    Uses a fixed CPU-only ``torch.Generator`` for the input ids so that
    input generation is tied to ``seed`` via the global RNGs that
    ``seed_everything`` just set.
    """
    seed_everything(seed, deterministic=deterministic)
    config = _tiny_config()
    model = TitansMAC(config)
    model.eval()

    # seq_len must equal chunk_size — TitansMAC.forward() no longer
    # accepts multi-chunk inputs; callers chunk upstream.
    seq_len = config.chunk_size
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    labels = torch.randint(0, config.vocab_size, (1, seq_len))

    with torch.no_grad():
        logits, _states, _gates = model(input_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    )
    params = [p.detach().clone() for p in model.parameters()]
    return model, loss, params


def test_same_seed_deterministic_cpu_identical_loss() -> None:
    """Same seed + ``deterministic=True`` on CPU → identical loss to 1e-6."""
    _, loss_a, params_a = _build_model_and_loss(seed=42, deterministic=True)
    _, loss_b, params_b = _build_model_and_loss(seed=42, deterministic=True)

    assert torch.allclose(loss_a, loss_b, atol=1e-6, rtol=0), (
        f"Deterministic CPU runs with seed=42 diverged: "
        f"loss_a={loss_a.item()} loss_b={loss_b.item()}"
    )
    # Stronger check: every parameter tensor must be bit-identical after init.
    assert len(params_a) == len(params_b), "parameter counts differ across runs"
    for i, (pa, pb) in enumerate(zip(params_a, params_b, strict=True)):
        assert torch.equal(pa, pb), (
            f"parameter {i} differs between seeded runs "
            f"(shape={tuple(pa.shape)}, max_abs_diff="
            f"{(pa - pb).abs().max().item()})"
        )


def test_same_seed_without_deterministic_still_repeats_on_cpu() -> None:
    """Same seed + ``deterministic=False`` on CPU → still bit-identical.

    CPU is deterministic by default for all ops the tiny model touches; the
    ``--deterministic`` switch only meaningfully changes behaviour on CUDA.
    This test pins that property so a future CPU-op change that breaks it
    shows up in CI instead of at user-report time.
    """
    _, loss_a, params_a = _build_model_and_loss(seed=7, deterministic=False)
    _, loss_b, params_b = _build_model_and_loss(seed=7, deterministic=False)

    assert torch.allclose(loss_a, loss_b, atol=1e-6, rtol=0)
    for pa, pb in zip(params_a, params_b, strict=True):
        assert torch.equal(pa, pb)


def test_different_seeds_produce_different_losses() -> None:
    """Sanity: a no-op implementation cannot trivially pass the equality tests.

    Two different seeds must produce two different losses — otherwise
    ``seed_everything`` is not actually driving the RNGs the model uses.
    """
    _, loss_a, params_a = _build_model_and_loss(seed=0, deterministic=True)
    _, loss_b, params_b = _build_model_and_loss(seed=123, deterministic=True)

    assert not torch.allclose(loss_a, loss_b, atol=1e-6, rtol=0), (
        "Two different seeds produced identical losses; seeding is not "
        "actually taking effect."
    )
    # At least one parameter tensor must differ between seeds.
    any_param_differs = any(
        not torch.equal(pa, pb) for pa, pb in zip(params_a, params_b, strict=True)
    )
    assert any_param_differs, (
        "All parameters identical across different seeds — model init is "
        "not consuming the seeded RNG."
    )
