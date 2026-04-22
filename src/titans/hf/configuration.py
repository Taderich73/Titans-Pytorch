"""HuggingFace PretrainedConfig for Titans MAC models."""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from collections.abc import Iterator
from typing import Any

from transformers import PretrainedConfig

from titans import TITANS_SCHEMA_VERSION
from titans.config import TitansConfig


@contextlib.contextmanager
def _suppress_unversioned_warning() -> Iterator[None]:
    """Silence the schema-version ``DeprecationWarning`` emitted by
    ``TitansMACConfig.__init__`` when the HF base class makes internal
    default-instance calls (``self.__class__()``) for diffing or
    generation-parameter filtering.

    Scoped narrowly to our specific message so unrelated deprecations
    (e.g. ``torch_dtype``) still surface.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="HF config missing 'titans_schema_version' field.*",
            category=DeprecationWarning,
        )
        yield


class TitansMACConfig(PretrainedConfig):
    """HuggingFace config wrapping TitansConfig for the MAC architecture.

    All TitansConfig fields are stored as top-level attributes so they appear
    directly in config.json. Bidirectional conversion via ``to_titans_config()``
    and ``from_titans_config()``.
    """

    model_type = "titans-mac"

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        vocab_size: int = 50257,
        ffn_mult: float = 4.0,
        num_memory_layers: int = 2,
        memory_hidden_mult: float = 4.0,
        num_persistent_tokens: int = 16,
        chunk_size: int = 512,
        window_size: int = 512,
        max_seq_len: int = 8192,
        memory_lr: float = 0.1,
        memory_momentum: float = 0.9,
        memory_error_clip: float = 10.0,
        memory_grad_clip: float = 1.0,
        memory_objective: str = "l2",
        huber_delta_init: float = 0.0,
        use_conv: bool = True,
        conv_kernel_size: int = 4,
        use_rope: bool = True,
        rope_proportion: float = 1.0,
        use_tnt: bool = False,
        global_chunk_size: int = 2048,
        local_chunk_sizes: list[int] | None = None,
        local_shard_length: int = 2048,
        use_qk_projection: bool = True,
        tnt_qk_projection: str = "per_position",
        tnt_stage: int = 1,
        finetune_local_chunk_sizes: list[int] | None = None,
        use_attn_res: bool = False,
        num_attnres_blocks: int = 8,
        attnres_warmup_steps: int = 0,
        attnres_modulate_global_memory: bool = True,
        attnres_modulate_local_memory: bool = False,
        quantize_memory_state: bool = False,
        memory_state_weight_bits: int = 4,
        memory_state_momentum_bits: int = 8,
        adaptive_window: bool = False,
        adaptive_window_min: int = 64,
        adaptive_window_max: int | None = None,
        adaptive_window_temperature: float = 10.0,
        adaptive_window_lambda: float = 0.01,
        use_mca: bool = False,
        mca_insertion_layers: list[int] | None = None,
        mca_num_heads: int = 8,
        mca_gate_type: str = "scalar",
        mca_gate_bias_init: float = -3.0,
        gate_decay_bias_init: float = -2.0,
        per_chunk_decay: bool = True,
        delta_memory_param: bool = True,
        detach_memory_state_in_forward: bool = False,
        attnres_logit_clip: float = 30.0,
        mca_auto_dump: bool = False,
        mca_dump_trigger: str = "session_end",
        mca_dump_path: str = "./memory_dumps/",
        mca_dump_keep_last_n: int = 10,
        dropout: float = 0.0,
        activation: str = "silu",
        init_std: float = 0.02,
        auto_checkpoint: bool = False,
        checkpoint_config: Any | None = None,
        mac_per_position_memory_query: bool = True,
        num_memory_inner_steps: int = 1,
        titans_schema_version: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Set architectures before super().__init__ so save_pretrained includes it.
        kwargs.setdefault("architectures", ["TitansMACForCausalLM"])
        super().__init__(**kwargs)
        # Schema-version dispatch (P5). Sentinel default (``None``) is how
        # we distinguish "caller omitted the field entirely" (which includes
        # loading an unversioned, pre-0.7 ``config.json``) from "caller
        # asked for the current version explicitly." Silently defaulting to
        # ``TITANS_SCHEMA_VERSION`` would upgrade legacy configs in place
        # and defeat the whole point of versioning.
        if titans_schema_version is None:
            warnings.warn(
                (
                    "HF config missing 'titans_schema_version' field; "
                    "assuming pre-0.7 unversioned layout. Re-save the "
                    "config with a current version of titans to silence "
                    "this warning. See MIGRATIONS.md."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            titans_schema_version = TITANS_SCHEMA_VERSION  # best-effort
        elif titans_schema_version > TITANS_SCHEMA_VERSION:
            raise RuntimeError(
                f"HF config schema version {titans_schema_version} > "
                f"current {TITANS_SCHEMA_VERSION}; upgrade titans. See "
                "MIGRATIONS.md."
            )
        elif titans_schema_version < TITANS_SCHEMA_VERSION:
            # v1 has no migrations; future versions should dispatch here
            # via a registry analogous to titans.memory_dump._MIGRATIONS.
            raise RuntimeError(
                f"HF config schema version {titans_schema_version} is "
                f"older than current {TITANS_SCHEMA_VERSION} and no "
                "migration is registered. See MIGRATIONS.md."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.ffn_mult = ffn_mult
        self.num_memory_layers = num_memory_layers
        self.memory_hidden_mult = memory_hidden_mult
        self.num_persistent_tokens = num_persistent_tokens
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.memory_lr = memory_lr
        self.memory_momentum = memory_momentum
        self.memory_error_clip = memory_error_clip
        self.memory_grad_clip = memory_grad_clip
        self.memory_objective = memory_objective
        self.huber_delta_init = huber_delta_init
        self.use_conv = use_conv
        self.conv_kernel_size = conv_kernel_size
        self.use_rope = use_rope
        self.rope_proportion = rope_proportion
        self.use_tnt = use_tnt
        self.global_chunk_size = global_chunk_size
        self.local_chunk_sizes = (
            local_chunk_sizes if local_chunk_sizes is not None else [8, 16]
        )
        self.local_shard_length = local_shard_length
        self.use_qk_projection = use_qk_projection
        self.tnt_qk_projection = tnt_qk_projection
        self.tnt_stage = tnt_stage
        self.finetune_local_chunk_sizes = finetune_local_chunk_sizes
        self.use_attn_res = use_attn_res
        self.num_attnres_blocks = num_attnres_blocks
        self.attnres_warmup_steps = attnres_warmup_steps
        self.attnres_modulate_global_memory = attnres_modulate_global_memory
        self.attnres_modulate_local_memory = attnres_modulate_local_memory
        self.quantize_memory_state = quantize_memory_state
        self.memory_state_weight_bits = memory_state_weight_bits
        self.memory_state_momentum_bits = memory_state_momentum_bits
        self.adaptive_window = adaptive_window
        self.adaptive_window_min = adaptive_window_min
        self.adaptive_window_max = adaptive_window_max
        self.adaptive_window_temperature = adaptive_window_temperature
        self.adaptive_window_lambda = adaptive_window_lambda
        self.use_mca = use_mca
        self.mca_insertion_layers = mca_insertion_layers
        self.mca_num_heads = mca_num_heads
        self.mca_gate_type = mca_gate_type
        self.mca_gate_bias_init = mca_gate_bias_init
        self.gate_decay_bias_init = gate_decay_bias_init
        self.per_chunk_decay = per_chunk_decay
        self.delta_memory_param = delta_memory_param
        self.detach_memory_state_in_forward = detach_memory_state_in_forward
        self.attnres_logit_clip = attnres_logit_clip
        self.mca_auto_dump = mca_auto_dump
        self.mca_dump_trigger = mca_dump_trigger
        self.mca_dump_path = mca_dump_path
        self.mca_dump_keep_last_n = mca_dump_keep_last_n
        self.dropout = dropout
        self.activation = activation
        self.init_std = init_std
        self.mac_per_position_memory_query = mac_per_position_memory_query
        self.num_memory_inner_steps = num_memory_inner_steps
        # Schema version (P5) — always surfaced as a top-level config.json
        # attribute so downstream consumers can branch on it. Dispatch
        # (missing -> warn; newer/older -> RuntimeError) is handled
        # immediately after ``super().__init__`` above so any failure
        # happens before we populate the rest of the object.
        self.titans_schema_version = titans_schema_version
        self.auto_checkpoint = auto_checkpoint
        # Serialize MemoryCheckpointConfig as a dict on the HF side so it
        # survives JSON round-trips (config.json); rehydrate in
        # to_titans_config().
        if checkpoint_config is not None and not isinstance(checkpoint_config, dict):
            self.checkpoint_config = (
                checkpoint_config.to_dict()
                if hasattr(checkpoint_config, "to_dict")
                else dataclasses.asdict(checkpoint_config)
            )
        else:
            self.checkpoint_config = checkpoint_config

    def to_diff_dict(self) -> dict[str, Any]:
        """Serialize only fields that differ from class defaults.

        Overrides :meth:`PretrainedConfig.to_diff_dict` solely to suppress
        the unversioned-config ``DeprecationWarning`` that would otherwise
        fire when the base implementation constructs a throw-away
        ``self.__class__()`` instance to compute defaults.
        """
        with _suppress_unversioned_warning():
            return super().to_diff_dict()

    def _get_generation_parameters(self) -> dict[str, Any]:
        """Same rationale as :meth:`to_diff_dict` — the base impl
        constructs a throw-away default instance, which would otherwise
        emit a spurious unversioned-config warning on every
        :meth:`save_pretrained`."""
        with _suppress_unversioned_warning():
            return super()._get_generation_parameters()

    def to_titans_config(self) -> TitansConfig:
        """Convert to native TitansConfig for model construction."""
        field_names = {f.name for f in dataclasses.fields(TitansConfig)}
        kwargs = {k: getattr(self, k) for k in field_names if hasattr(self, k)}

        cp = kwargs.get("checkpoint_config")
        if isinstance(cp, dict):
            from titans.checkpointing import MemoryCheckpointConfig

            kwargs["checkpoint_config"] = MemoryCheckpointConfig.from_dict(cp)
        return TitansConfig(**kwargs)

    @classmethod
    def from_titans_config(
        cls, titans_config: TitansConfig, **kwargs: Any
    ) -> TitansMACConfig:
        """Create from an existing native TitansConfig.

        Stamps the current ``TITANS_SCHEMA_VERSION`` explicitly — this
        is a fresh in-memory construction, not a load from disk, so it
        should not take the unversioned legacy path. Callers can still
        override via ``**kwargs`` (useful for tests that exercise the
        version-dispatch branches).
        """
        d = titans_config.to_dict()
        d.setdefault("titans_schema_version", TITANS_SCHEMA_VERSION)
        d.update(kwargs)
        return cls(**d)
