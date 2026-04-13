"""HuggingFace transformers integration for Titans models."""

from __future__ import annotations

try:
    import transformers  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "titans.hf requires the transformers library. "
        "Install with: pip install titans[hf]"
    ) from exc
