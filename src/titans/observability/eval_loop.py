"""Feature B: validation loss loop with deterministic holdout partitioning."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader


@dataclass
class EvalConfig:
    """Configuration for periodic validation.

    Attributes:
        every_n_steps: Run eval every N optimizer steps. 0 disables.
        num_batches: Number of batches consumed per eval.
        reset_memory_state: Reset Titans memory state before eval.
        holdout_fraction: Fraction of dataset reserved for eval (e.g. 0.001).
        holdout_seed: Fixed seed used by is_eval_example; independent of --seed.
        denom: Partition denominator (resolution of the fraction). Default 100000.
    """

    every_n_steps: int = 2500
    num_batches: int = 200
    reset_memory_state: bool = True
    holdout_fraction: float = 0.001
    holdout_seed: int = 12345
    denom: int = 100_000


def is_eval_example(key: str, cfg: EvalConfig) -> bool:
    """Deterministic partition: True iff this key belongs to the eval slice.

    Properties guaranteed:
        - Streaming-safe (decision computable per-example in isolation).
        - Stable across runs/resumption (pure function of key + cfg.holdout_seed).
        - Independent of any other seeding (no global RNG consulted).
        - Disjoint by construction: callers invert for the train loader.

    Args:
        key: A stable per-example identifier. Prefer the dataset's native
            'id' or 'url' field; fall back to the streaming row index.
        cfg: The active EvalConfig.

    Returns:
        True iff the example is in the eval partition.
    """
    numer = max(1, int(round(cfg.denom * cfg.holdout_fraction)))
    seed_bytes = cfg.holdout_seed.to_bytes(8, "little", signed=False)
    key_bytes = str(key).encode("utf-8")
    digest = hashlib.blake2b(seed_bytes + key_bytes, digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") % cfg.denom
    return bucket < numer


def stash_memory_states(memory_states: list[Any] | None) -> list[Any] | None:
    """Return a shallow copy of the training memory_states list for later restore.

    The caller is responsible for passing the result back via restore_memory_states
    once eval is complete. This is a list-level stash: the underlying state objects
    are referenced, not deep-copied. Since eval uses its own fresh states, the
    training objects are not mutated.
    """
    if memory_states is None:
        return None
    return list(memory_states)


def restore_memory_states(stashed: list[Any] | None) -> list[Any] | None:
    """Return the stashed training memory states unchanged."""
    return stashed


LossFn = Callable[[torch.nn.Module, dict[str, Any]], torch.Tensor]


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    accelerator: Any,
    cfg: EvalConfig,
    loss_fn: LossFn,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation over num_batches from eval_loader.

    The loss_fn callable decouples run_eval from any particular model calling
    convention. Simple HF-style models can pass
    ``lambda m, b: m(**b).loss``. Titans' chunked forward passes a function
    that iterates chunks and averages cross-entropy — see pretrain.py.

    Leaves model.training restored to its prior value on exit. Does not
    touch any external memory state: callers are expected to stash/restore
    around this call if state carry-over is in use.

    Args:
        model: The wrapped (or unwrapped) training model.
        eval_loader: Pre-prepared DataLoader yielding batch dicts.
        accelerator: Accelerator-like object providing gather_for_metrics
            and is_main_process.
        cfg: The active EvalConfig.
        loss_fn: Callable ``(model, batch) -> scalar Tensor`` that computes
            one batch's loss.
        max_batches: Optional cap. When set,
            ``min(cfg.num_batches, max_batches)`` batches are consumed.

    Returns:
        {'eval/loss', 'eval/ppl', 'eval/num_batches', 'eval/wall_time_s'}.
    """
    was_training = model.training
    model.eval()
    n_target = (
        cfg.num_batches if max_batches is None else min(cfg.num_batches, max_batches)
    )

    start = time.monotonic()
    total_loss = torch.zeros(1, dtype=torch.float64)
    total_batches = 0

    for i, batch in _take(iter(eval_loader), n_target):
        loss = loss_fn(model, batch)
        gathered = accelerator.gather_for_metrics(loss.detach().float().reshape(1))
        total_loss += gathered.double().mean().cpu()
        total_batches = i + 1

    if was_training:
        model.train()

    if total_batches == 0:
        return {
            "eval/loss": float("nan"),
            "eval/ppl": float("nan"),
            "eval/num_batches": 0,
            "eval/wall_time_s": 0.0,
        }

    mean_loss = float(total_loss.item() / total_batches)
    return {
        "eval/loss": mean_loss,
        "eval/ppl": math.exp(mean_loss),
        "eval/num_batches": total_batches,
        "eval/wall_time_s": round(time.monotonic() - start, 3),
    }


def _take(it: Iterator[Any], n: int) -> Iterator[tuple[int, Any]]:
    """Yield up to n (index, item) pairs from an iterator."""
    for i, item in enumerate(it):
        if i >= n:
            return
        yield i, item
