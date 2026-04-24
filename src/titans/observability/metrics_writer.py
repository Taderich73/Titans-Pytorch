"""JSONL metrics writer + tqdm postfix summarizer.

Shared infrastructure used by all four observability features.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Keys projected to the tqdm postfix in the order they should appear.
# All other keys go to JSONL only (per-block arrays and housekeeping).
_TQDM_KEY_MAP: dict[str, str] = {
    "loss": "loss",
    "lr": "lr",
    "grad/global_norm": "|g|",
    "eval/loss": "eval",
    "gate/alpha_mean": "α̂",
    "gate/alpha_std": "α̂_std",
    "layer/state_norm_mean": "|s|",
    "layer/state_norm_std": "|s|_std",
    "layer/weight_norm_mean": "|W|",
    "layer/weight_norm_std": "|W|_std",
    "decay_bias": "decay_bias",
}


class MetricsWriter:
    """Append-mode JSONL writer for training metrics.

    One row per call to log(). Rows contain 'step' and 'wall_time' plus any
    caller-supplied metrics. Feature modules namespace their keys with '/'.
    """

    def __init__(self, jsonl_path: Path, flush_every: int = 1) -> None:
        """Open jsonl_path in append mode. Creates parent dirs if needed.

        Args:
            jsonl_path: Destination file. Parent dirs are created.
            flush_every: Flush the file handle every N log() calls.
        """
        self._path = Path(jsonl_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._flush_every = max(1, int(flush_every))
        self._write_count = 0
        self._start_wall = time.monotonic()

    def log(self, step: int, **metrics: Any) -> None:
        """Append one JSON row with step + wall_time + supplied metrics."""
        row: dict[str, Any] = {
            "step": int(step),
            "wall_time": round(time.monotonic() - self._start_wall, 3),
        }
        row.update(metrics)
        self._fh.write(json.dumps(row, default=_json_default))
        self._fh.write("\n")
        self._write_count += 1
        if self._write_count % self._flush_every == 0:
            self._fh.flush()

    def tqdm_summary(self, pbar: Any, step: int, **metrics: Any) -> None:  # noqa: ARG002
        """Project the curated subset of metrics onto the tqdm postfix."""
        postfix: dict[str, str] = {}
        for key, short in _TQDM_KEY_MAP.items():
            if key in metrics:
                postfix[short] = _format_scalar(metrics[key])
        if postfix:
            pbar.set_postfix(**postfix)

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:  # noqa: BLE001 - best-effort close
            pass


class NullMetricsWriter:
    """JSONL-disabled writer: skips JSONL but still updates tqdm.

    Returned on non-main ranks (where log() must be a no-op to avoid
    inter-rank file contention) and when JSONL is explicitly disabled by
    passing an empty path (where the user wants to keep training quiet on
    disk but still see curated tqdm postfix updates).

    The ``log()`` and ``close()`` methods are true no-ops; ``tqdm_summary``
    projects the curated subset of metrics onto the pbar postfix using the
    same ``_TQDM_KEY_MAP`` as ``MetricsWriter`` so that disabling JSONL
    never silently disables the terminal display -- matching the intent
    documented in the package README and in the ``--metrics-jsonl`` help
    text ("Empty string disables JSONL and keeps tqdm-only logging").
    """

    def log(self, step: int, **metrics: Any) -> None:
        """No-op."""

    def tqdm_summary(self, pbar: Any, step: int, **metrics: Any) -> None:  # noqa: ARG002
        """Project the curated subset of metrics onto the tqdm postfix."""
        postfix: dict[str, str] = {}
        for key, short in _TQDM_KEY_MAP.items():
            if key in metrics:
                postfix[short] = _format_scalar(metrics[key])
        if postfix:
            pbar.set_postfix(**postfix)

    def close(self) -> None:
        """No-op."""


def build_metrics_writer(
    jsonl_path: str,
    accelerator: Any,
) -> MetricsWriter | NullMetricsWriter:
    """Factory. Returns a real writer on main process when path is non-empty.

    Returns NullMetricsWriter when:
    - jsonl_path is empty string
    - accelerator.is_main_process is False
    - the file cannot be opened (prints warning to stderr)
    """
    if not jsonl_path:
        return NullMetricsWriter()
    if not getattr(accelerator, "is_main_process", True):
        return NullMetricsWriter()
    try:
        return MetricsWriter(Path(jsonl_path))
    except OSError as exc:
        print(
            f"[observability] Could not open {jsonl_path}: {exc}. "
            "Falling back to NullMetricsWriter.",
            file=sys.stderr,
        )
        return NullMetricsWriter()


def _format_scalar(value: Any) -> str:
    """Format a scalar for tqdm postfix display."""
    if isinstance(value, float):
        abs_v = abs(value)
        if abs_v != 0 and (abs_v < 1e-3 or abs_v >= 1e4):
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def _json_default(obj: Any) -> Any:
    """JSON fallback for objects that are not natively serializable."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
