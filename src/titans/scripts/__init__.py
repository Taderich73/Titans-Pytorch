# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Script-level orchestration glue shipped with the ``titans`` package.

This subpackage is the single source of truth for helpers shared by the
training and inference scripts under ``scripts/`` at the repo root
(``pretrain.py``, ``sft.py``, ``lora.py``, ``dpo.py``, ``rlvr.py``,
``inference.py``).

It lives inside the installable package (rather than at repo-root
``scripts/_common.py``) so remote runners that receive a single script
file — notably HuggingFace Jobs via ``api.run_uv_job`` — can reach these
helpers via the pinned ``titans`` wheel instead of needing the whole
repo on disk.

Usage::

    from titans.scripts import (
        build_titans_config,
        create_model,
        init_accelerator_and_logging,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
    )

The helpers here are script-level glue only; anything that belongs to the
library proper stays under ``titans`` directly.
"""

from __future__ import annotations

from titans.scripts._common import (
    CHATML_IM_END,
    CHATML_IM_START,
    MODEL_CLASSES,
    AcceleratorBundle,
    CheckpointSetup,
    base_argparse_parser,
    build_loss_mask,
    build_titans_config,
    chunked_forward,
    create_model,
    format_chatml,
    init_accelerator_and_logging,
    initialize_missing_optimizer_state,
    is_optimizer_state_compatible,
    loss_mask_to_zero_one,
    make_dataloader,
    make_optimizer,
    maybe_compile,
    move_optimizer_state_to_params,
    setup_checkpoint_dir,
    tokenize_chat,
)

__all__ = [
    "CHATML_IM_END",
    "CHATML_IM_START",
    "MODEL_CLASSES",
    "AcceleratorBundle",
    "CheckpointSetup",
    "base_argparse_parser",
    "build_loss_mask",
    "build_titans_config",
    "chunked_forward",
    "create_model",
    "format_chatml",
    "init_accelerator_and_logging",
    "initialize_missing_optimizer_state",
    "is_optimizer_state_compatible",
    "loss_mask_to_zero_one",
    "make_dataloader",
    "make_optimizer",
    "maybe_compile",
    "move_optimizer_state_to_params",
    "setup_checkpoint_dir",
    "tokenize_chat",
]
