# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Configuration for Titans PyTorch models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TitansConfig:
    """Configuration for Titans models.

    All fields are preserved from the MLX implementation. Features not yet
    ported (TNT, AttnRes, MCA, adaptive window) are accepted here but raise
    NotImplementedError at model construction time.
    """

    # Core dimensions
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    ffn_mult: float = 4.0

    # Memory configuration
    num_memory_layers: int = 2
    memory_hidden_mult: float = 4.0
    num_persistent_tokens: int = 16

    # Sequence configuration
    chunk_size: int = 512
    window_size: int = 512
    max_seq_len: int = 8192

    # Memory learning parameters
    memory_lr: float = 0.1
    memory_momentum: float = 0.9
    memory_error_clip: float = 10.0
    memory_grad_clip: float = 1.0

    # Memory objective (attentional bias)
    memory_objective: str = "l2"
    huber_delta_init: float = 0.0

    # Architecture options
    use_conv: bool = True
    conv_kernel_size: int = 4
    use_rope: bool = True

    # TNT Hierarchical Memory (deferred)
    use_tnt: bool = False
    global_chunk_size: int = 2048
    local_chunk_sizes: list[int] = field(default_factory=lambda: [8, 16])
    local_shard_length: int = 2048
    use_qk_projection: bool = True
    tnt_stage: int = 1
    finetune_local_chunk_sizes: list[int] | None = None

    # AttnRes (deferred)
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global_memory: bool = True
    attnres_modulate_local_memory: bool = False

    # Memory state quantization (deferred)
    quantize_memory_state: bool = False
    memory_state_weight_bits: int = 4
    memory_state_momentum_bits: int = 8

    # Adaptive window sizing (deferred)
    adaptive_window: bool = False
    adaptive_window_min: int = 64
    adaptive_window_max: int | None = None
    adaptive_window_temperature: float = 10.0
    adaptive_window_lambda: float = 0.01

    # Memory Cross-Attention (deferred)
    use_mca: bool = False
    mca_insertion_layers: list[int] | None = None
    mca_num_heads: int = 8
    mca_gate_type: str = "scalar"
    mca_gate_bias_init: float = -3.0

    # Gate initialization
    gate_decay_bias_init: float = -6.0

    # AttnRes numerical stability
    attnres_logit_clip: float = 30.0

    # Memory dump
    mca_auto_dump: bool = False
    mca_dump_trigger: str = "session_end"
    mca_dump_path: str = "./memory_dumps/"
    mca_dump_keep_last_n: int = 10

    # Training
    dropout: float = 0.0
    activation: str = "silu"
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return int(self.dim * self.ffn_mult)

    @property
    def memory_hidden_dim(self) -> int:
        return int(self.dim * self.memory_hidden_mult)

    @property
    def effective_adaptive_window_max(self) -> int:
        if self.adaptive_window_max is not None:
            return self.adaptive_window_max
        return self.window_size

    @property
    def num_local_memories(self) -> int:
        return len(self.local_chunk_sizes)

    @property
    def active_local_chunk_sizes(self) -> list[int]:
        if self.tnt_stage == 2 and self.finetune_local_chunk_sizes is not None:
            return self.finetune_local_chunk_sizes
        return self.local_chunk_sizes

    def __post_init__(self) -> None:
        valid_objectives = ("l2", "huber")
        if self.memory_objective not in valid_objectives:
            raise ValueError(
                f"memory_objective must be one of {valid_objectives}, "
                f"got '{self.memory_objective}'"
            )
        if self.use_mca:
            for idx in self.mca_active_insertion_layers:
                if idx >= self.num_layers:
                    raise ValueError(
                        f"MCA insertion layer {idx} >= num_layers {self.num_layers}"
                    )

    @property
    def mca_active_insertion_layers(self) -> list[int]:
        if not self.use_mca:
            return []
        if self.mca_insertion_layers is not None:
            return self.mca_insertion_layers
        return [self.num_layers // 2]

    @property
    def attnres_sub_layer_block_size(self) -> int:
        num_mca_layers = len(self.mca_active_insertion_layers) if self.use_mca else 0
        total_sub_layers = (self.num_layers * 2) + num_mca_layers
        return max(1, total_sub_layers // self.num_attnres_blocks)

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "ffn_mult": self.ffn_mult,
            "num_memory_layers": self.num_memory_layers,
            "memory_hidden_mult": self.memory_hidden_mult,
            "num_persistent_tokens": self.num_persistent_tokens,
            "chunk_size": self.chunk_size,
            "window_size": self.window_size,
            "max_seq_len": self.max_seq_len,
            "memory_lr": self.memory_lr,
            "memory_momentum": self.memory_momentum,
            "memory_error_clip": self.memory_error_clip,
            "memory_grad_clip": self.memory_grad_clip,
            "memory_objective": self.memory_objective,
            "huber_delta_init": self.huber_delta_init,
            "use_conv": self.use_conv,
            "conv_kernel_size": self.conv_kernel_size,
            "use_rope": self.use_rope,
            "use_tnt": self.use_tnt,
            "global_chunk_size": self.global_chunk_size,
            "local_chunk_sizes": self.local_chunk_sizes,
            "local_shard_length": self.local_shard_length,
            "use_qk_projection": self.use_qk_projection,
            "tnt_stage": self.tnt_stage,
            "finetune_local_chunk_sizes": self.finetune_local_chunk_sizes,
            "use_attn_res": self.use_attn_res,
            "num_attnres_blocks": self.num_attnres_blocks,
            "attnres_warmup_steps": self.attnres_warmup_steps,
            "attnres_modulate_global_memory": self.attnres_modulate_global_memory,
            "attnres_modulate_local_memory": self.attnres_modulate_local_memory,
            "quantize_memory_state": self.quantize_memory_state,
            "memory_state_weight_bits": self.memory_state_weight_bits,
            "memory_state_momentum_bits": self.memory_state_momentum_bits,
            "adaptive_window": self.adaptive_window,
            "adaptive_window_min": self.adaptive_window_min,
            "adaptive_window_max": self.adaptive_window_max,
            "adaptive_window_temperature": self.adaptive_window_temperature,
            "adaptive_window_lambda": self.adaptive_window_lambda,
            "use_mca": self.use_mca,
            "mca_insertion_layers": self.mca_insertion_layers,
            "mca_num_heads": self.mca_num_heads,
            "mca_gate_type": self.mca_gate_type,
            "mca_gate_bias_init": self.mca_gate_bias_init,
            "gate_decay_bias_init": self.gate_decay_bias_init,
            "attnres_logit_clip": self.attnres_logit_clip,
            "mca_auto_dump": self.mca_auto_dump,
            "mca_dump_trigger": self.mca_dump_trigger,
            "mca_dump_path": self.mca_dump_path,
            "mca_dump_keep_last_n": self.mca_dump_keep_last_n,
            "dropout": self.dropout,
            "activation": self.activation,
            "init_std": self.init_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TitansConfig:
        return cls(**d)

    @classmethod
    def tnt_stage1(cls, **kwargs) -> TitansConfig:
        defaults = dict(
            use_tnt=True,
            global_chunk_size=2048,
            local_chunk_sizes=[8, 16],
            local_shard_length=2048,
            tnt_stage=1,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def tnt_stage2(cls, stage1_config: TitansConfig) -> TitansConfig:
        d = stage1_config.to_dict()
        d["finetune_local_chunk_sizes"] = [
            max(1, cs // 2) for cs in d["local_chunk_sizes"]
        ]
        d["tnt_stage"] = 2
        return cls.from_dict(d)
