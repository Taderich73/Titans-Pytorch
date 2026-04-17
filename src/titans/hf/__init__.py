"""HuggingFace transformers integration for Titans models.

Primary usage (no trust_remote_code needed)::

    from titans.hf import TitansMACForCausalLM
    model = TitansMACForCausalLM.from_pretrained("repo/name")

Auto-registration (requires ``import titans.hf`` first)::

    import titans.hf
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("repo/name")
"""

from __future__ import annotations

try:
    import transformers  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "titans.hf requires the transformers library. "
        "Install with: pip install titans[hf]"
    ) from exc

from titans.hf.configuration import TitansMACConfig
from titans.hf.modeling import TitansMACForCausalLM
from titans.hf.trainer import TitansChunkMixin, TitansTrainer

# Auto-registration — one entry per variant.
# Adding a new variant means adding one tuple here.
_VARIANT_REGISTRY = [
    ("titans-mac", TitansMACConfig, TitansMACForCausalLM),
    # Future: ("titans-mag", TitansMAGConfig, TitansMAGForCausalLM),
    # Future: ("titans-mal", TitansMALConfig, TitansMALForCausalLM),
    # Future: ("titans-lmm", TitansLMMConfig, TitansLMMForCausalLM),
]

from transformers import AutoConfig, AutoModelForCausalLM


def _safe_register(model_type: str, config_cls, model_cls) -> None:
    """Register with AutoConfig / AutoModelForCausalLM; idempotent on re-import.

    Uses ``exist_ok=True`` (supported by transformers >= 5.0) to tolerate
    duplicate registration without catching exceptions. This avoids fragile
    substring matching against upstream error messages and is the supported
    way to make registration idempotent for notebooks / module reloads.
    """
    AutoConfig.register(model_type, config_cls, exist_ok=True)
    AutoModelForCausalLM.register(config_cls, model_cls, exist_ok=True)


for _model_type, _config_cls, _model_cls in _VARIANT_REGISTRY:
    _safe_register(_model_type, _config_cls, _model_cls)

__all__ = [
    "TitansMACConfig",
    "TitansMACForCausalLM",
    "TitansTrainer",
    "TitansChunkMixin",
]
