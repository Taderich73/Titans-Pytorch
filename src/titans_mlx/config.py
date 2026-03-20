# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Configuration for Titans MLX models.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TitansConfig:
    """Configuration for Titans models.

    Attributes:
        dim: Model dimension (d_model)
        num_heads: Number of attention heads
        num_layers: Number of Titans blocks
        vocab_size: Vocabulary size for embedding
        num_memory_layers: Depth of memory MLP (L_M >= 1)
        memory_hidden_mult: Hidden dimension multiplier for memory MLP
        num_persistent_tokens: Number of persistent memory tokens (N_p)
        chunk_size: Segment size for MAC variant
        window_size: Sliding window size for MAG/MAL variants
        memory_lr: Learning rate for memory updates (theta)
        memory_momentum: Momentum coefficient for memory (eta)
        memory_error_clip: Clip bound for error in gradient computation
        memory_grad_clip: Clip bound for computed gradients
        max_seq_len: Maximum sequence length
        use_conv: Whether to use 1D convolution
        conv_kernel_size: Kernel size for convolution
        use_rope: Whether to use Rotary Position Embeddings
        dropout: Dropout probability
        activation: Activation function name
        init_std: Standard deviation for weight initialization
        use_tnt: Whether to enable TNT hierarchical memory mode
        global_chunk_size: C_G — chunk size for global memory updates
        local_chunk_sizes: C_L — chunk sizes for each local memory (one per local memory)
        local_shard_length: S_L — shard length (local memory reset period)
        use_qk_projection: Whether to use Q-K projection for local retrieval
        tnt_stage: Training stage (1 = pre-training, 2 = fine-tuning)
        finetune_local_chunk_sizes: C_L' — smaller chunk sizes for stage 2 fine-tuning
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

    # Architecture options
    use_conv: bool = True
    conv_kernel_size: int = 4
    use_rope: bool = True

    # TNT Hierarchical Memory
    use_tnt: bool = False
    global_chunk_size: int = 2048
    local_chunk_sizes: list[int] = field(default_factory=lambda: [8, 16])
    local_shard_length: int = 2048
    use_qk_projection: bool = True
    tnt_stage: int = 1
    finetune_local_chunk_sizes: list[int] | None = None

    # Training
    dropout: float = 0.0
    activation: str = "silu"
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        """Feed-forward network hidden dimension."""
        return int(self.dim * self.ffn_mult)

    @property
    def memory_hidden_dim(self) -> int:
        """Hidden dimension for memory MLP."""
        return int(self.dim * self.memory_hidden_mult)

    @property
    def num_local_memories(self) -> int:
        """Number of local memories (N), derived from local_chunk_sizes."""
        return len(self.local_chunk_sizes)

    @property
    def active_local_chunk_sizes(self) -> list[int]:
        """Active local chunk sizes for the current training stage.

        Returns finetune sizes during stage 2 if provided, otherwise
        the base local_chunk_sizes.
        """
        if self.tnt_stage == 2 and self.finetune_local_chunk_sizes is not None:
            return self.finetune_local_chunk_sizes
        return self.local_chunk_sizes

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
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
            "dropout": self.dropout,
            "activation": self.activation,
            "init_std": self.init_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TitansConfig:
        """Create config from dictionary."""
        return cls(**d)

    @classmethod
    def tnt_stage1(cls, **kwargs) -> TitansConfig:
        """Default Stage 1 (pre-training) TNT configuration.

        Large global chunks (C_G=2048) for long-range context,
        moderate local chunks (C_L={8,16}) for detail. Focus on throughput.
        """
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
        """Create Stage 2 (fine-tuning) config from a Stage 1 config.

        Halves each local chunk size for finer-grained memorization.
        Global memory is intended to be frozen during stage 2 training.
        ~5% additional compute over stage 1.
        """
        d = stage1_config.to_dict()
        d["finetune_local_chunk_sizes"] = [
            max(1, cs // 2) for cs in d["local_chunk_sizes"]
        ]
        d["tnt_stage"] = 2
        return cls.from_dict(d)
