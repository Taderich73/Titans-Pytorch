# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "titans @ git+https://github.com/Taderich73/OpenTitans.git",
#     "torch>=2.2.0,<2.8.0",
#     "accelerate>=0.27.0",
#     "transformers>=4.36.0",
#     "datasets>=2.16.0",
#     "huggingface-hub>=0.20.0",
#     "wandb>=0.16.0",
#     "tqdm>=4.67.1",
#     "numpy>=2.0.0",
# ]
# ///

"""
Titans pretraining on HuggingFace Jobs.

Trains TitansMAC on FineWeb-Edu (streaming) and pushes checkpoints to Hub.
Designed to run via: hf jobs uv run scripts/pretrain.py --flavor a10g-large
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

# Force unbuffered stdout/stderr so HF Jobs logs stream in real-time.
# PYTHONUNBUFFERED must be set before interpreter start; this is the
# in-process equivalent.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from titans import TitansConfig
from titans._logging import setup_logging
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory_dump import save_memory_states
from titans.observability import (
    EvalConfig,
    GateHookRegistry,
    build_metrics_writer,
    collect_layer_stats,
    global_grad_norm,
    is_eval_example,
    restore_memory_states,
    run_eval,
    stash_memory_states,
)

# Shared script-level helpers ship inside the ``titans`` wheel under the
# ``titans.scripts`` subpackage so HuggingFace Jobs (which uploads only
# this file) can reach them via the pinned git dependency above. The
# local ``scripts/`` directory on disk is irrelevant at runtime on the
# remote. ``build_titans_config`` is re-exported for API parity with the
# other training scripts — pretrain.py itself still constructs its
# ``TitansConfig`` directly because it hard-codes the feature groups
# rather than flowing them through a duck-typed cfg namespace.
from titans.scripts import (
    build_titans_config,  # noqa: F401 — re-exported for API parity
    create_model,
    init_accelerator_and_logging,
    initialize_missing_optimizer_state,
    is_optimizer_state_compatible,
    make_dataloader,
    make_optimizer,
    maybe_compile,
    move_optimizer_state_to_params,
    remap_optimizer_state_by_name,
    setup_checkpoint_dir,
)
from titans.utils import seed_everything

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name()}")


# ---------------------------------------------------------------------------
# Memory profiling helpers
# ---------------------------------------------------------------------------
#
# Toggled on by PROFILE_MEMORY (set via --profile-memory in the launcher).
# Used to localize OOM root causes by capturing torch.cuda.memory_allocated()
# / max_memory_allocated() / memory_reserved() at key points in the training
# loop, plus per-block snapshots via forward hooks. All functions no-op when
# PROFILE_MEMORY is False or CUDA is unavailable.


def _mem_snapshot() -> tuple[float, float, float]:
    """Return (allocated_gb, max_allocated_gb, reserved_gb).

    Calls torch.cuda.synchronize() so the numbers reflect actual on-device
    state rather than queued kernels. Returns (0.0, 0.0, 0.0) when CUDA is
    not available.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return alloc, max_alloc, reserved


def _log_mem(label: str) -> None:
    """Log a memory snapshot with a descriptive label.

    No-op when PROFILE_MEMORY is False.
    """
    if not PROFILE_MEMORY:
        return
    alloc, max_alloc, reserved = _mem_snapshot()
    logger.info(
        f"[mem] {label:32s}  "
        f"alloc={alloc:6.2f}GB  "
        f"max_alloc={max_alloc:6.2f}GB  "
        f"reserved={reserved:6.2f}GB"
    )


def _reset_max_mem() -> None:
    """Reset the CUDA max-allocated counter so subsequent snapshots reflect
    the peak since the last reset. No-op when CUDA is unavailable.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _instrument_blocks_for_memory(model: torch.nn.Module) -> None:
    """Instrument each block with memory snapshots around its core_forward
    and around its memory module's forward call.

    The PyTorch forward_hook system fires on `__call__`, but
    `process_chunk` in titans.models calls `block.core_forward(...)`
    directly (a regular method call, not `__call__`), so a standard
    forward hook on the block never fires. This helper instead
    monkey-patches each block's `core_forward` attribute with a wrapper
    that snapshots memory before and after the original call. The
    `block.memory` submodule IS called via `__call__` from inside
    core_forward, so a standard register_forward_hook on `block.memory`
    works for capturing the memory module's contribution specifically.

    Combined output per block per chunk:
        [mem]   block[NN/19] core_forward start: ...
        [mem]   block[NN/19] memory done:        ...
        [mem]   block[NN/19] core_forward done:  ...

    For seq_len > chunk_size each block is called once per chunk per
    forward, so num_chunks * n_blocks lines fire per forward pass.

    Output is gated by PROFILE_MEMORY_STEPS so only the first few
    forward passes emit lines (after that the wrappers and hooks are
    silent but still installed). The pass counter is incremented by a
    forward hook on the top-level model so it counts ACTUAL forward
    passes, not block calls or chunks. No-op when PROFILE_MEMORY is
    False, when PROFILE_MEMORY_PER_BLOCK is False (the common case —
    per-block trace is opt-in even under --profile-memory because it's
    noisy), or when CUDA is unavailable.
    """
    if (
        not PROFILE_MEMORY
        or not PROFILE_MEMORY_PER_BLOCK
        or not torch.cuda.is_available()
    ):
        return

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return
    n_blocks = len(blocks)
    counter = {"pass_idx": 0}

    def _snapshot() -> tuple[float, float]:
        torch.cuda.synchronize()
        return (
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.max_memory_allocated() / 1e9,
        )

    def _under_limit() -> bool:
        return counter["pass_idx"] < PROFILE_MEMORY_STEPS

    # Top-level model post-forward hook: increment pass counter once per
    # actual call to model(...) regardless of how many chunks/blocks fired
    # inside. This is the source of truth for how many passes have run.
    def _bump_pass_counter(_module, _inputs, _output):
        counter["pass_idx"] += 1

    model.register_forward_hook(_bump_pass_counter)

    for i, block in enumerate(blocks):
        if not hasattr(block, "core_forward"):
            continue

        original_core_forward = block.core_forward
        block_idx = i

        def make_core_forward_wrapper(idx: int, original):
            def wrapped(*args, **kwargs):
                if not _under_limit():
                    return original(*args, **kwargs)
                before_alloc, _ = _snapshot()
                logger.debug(
                    f"[mem]   block[{idx:02d}/{n_blocks - 1}] core_forward start: "
                    f"alloc={before_alloc:6.2f}GB"
                )
                result = original(*args, **kwargs)
                after_alloc, max_alloc = _snapshot()
                delta = after_alloc - before_alloc
                logger.debug(
                    f"[mem]   block[{idx:02d}/{n_blocks - 1}] core_forward done:  "
                    f"alloc={after_alloc:6.2f}GB  delta={delta:+6.3f}GB  "
                    f"max={max_alloc:6.2f}GB"
                )
                return result

            return wrapped

        block.core_forward = make_core_forward_wrapper(block_idx, original_core_forward)

        # Forward hook on the memory submodule (called via __call__ inside
        # core_forward, so a regular forward_hook fires here).
        memory_module = getattr(block, "memory", None)
        if memory_module is None:
            continue

        def make_memory_hook(idx: int):
            def hook(_module, _inputs, _output):
                if not _under_limit():
                    return
                alloc, max_alloc = _snapshot()
                logger.debug(
                    f"[mem]   block[{idx:02d}/{n_blocks - 1}] memory done:        "
                    f"alloc={alloc:6.2f}GB  max={max_alloc:6.2f}GB"
                )

            return hook

        memory_module.register_forward_hook(make_memory_hook(block_idx))


# ---------------------------------------------------------------------------
# Configuration — edit these for your run
# ---------------------------------------------------------------------------

# Model — ~1.5B parameter MAC configuration (with TNT/AttnRes/MCA/Huber/AdaptiveWindow)
DIM = 1024
NUM_HEADS = 16
NUM_LAYERS = 20
VOCAB_SIZE = 50257
CHUNK_SIZE = 512
WINDOW_SIZE = 512
NUM_MEMORY_LAYERS = 2
NUM_PERSISTENT_TOKENS = 16
ROPE_PROPORTION = 1.0  # Fraction of head_dim pairs to apply RoPE to (0.0-1.0)

# Data
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-100BT"
TOKENIZER_NAME = "gpt2"
SEQ_LEN = 2048

# Training
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 24
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.03
MAX_STEPS = 50000
LOG_EVERY = 10
SAVE_EVERY = 2500
SAVE_FORMAT = "pt"
MIXED_PRECISION = "bf16"

# Hub persistence
HUB_REPO = "FlatFootInternational/titans-mac-1.5B"  # Where to push checkpoints
PUSH_CHECKPOINTS = True
# Flat "checkpoints/" matches the Hub upload paths at scripts/pretrain.py:803
# and :842 (path_in_repo=f"checkpoints/{fpath.name}"). The argparse default at
# :884 says "checkpoints/pretrain" for sibling-script parity, but that parser
# result is currently discarded — the mismatch is a pre-existing inconsistency.
CHECKPOINT_DIR = "checkpoints"

# Resume — set to a Hub checkpoint path to continue training, e.g.:
#   RESUME_FROM = "checkpoints/step_10000.pt"   (resumes from step 10000)
#   RESUME_FROM = "checkpoints/final.pt"         (resumes from final checkpoint)
#   RESUME_FROM = None                           (train from scratch)
RESUME_FROM = None

RESET_GLOBAL_STATE_PER_BATCH = (
    True  # set False to let global state carry across batches
)
STATE_CARRY_WARMUP_STEPS = (
    500  # reset for this many steps, then carry (ignored if reset=True)
)

# Seed
SEED = 42
# Flip to True to also enable torch.use_deterministic_algorithms(True) and
# set CUBLAS_WORKSPACE_CONFIG=:4096:8. See docs/reproducibility.md.
DETERMINISTIC = False

# Diagnostics — toggled on by --profile-memory in the launcher. When True the
# training loop logs torch.cuda.max_memory_allocated() at key points (model
# build, first batch load, before/after forward, after backward, after
# optimizer step). Adds minor overhead from synchronize() calls; leave False
# for production training.
PROFILE_MEMORY = False

# Per-block memory trace — OFF by default even when PROFILE_MEMORY is on,
# because it emits num_blocks * num_chunks lines per forward pass (e.g.
# 80 lines per step for the 1.5B config) which drowns the high-level
# checkpoint output. Keep this False for steady-state monitoring; flip to
# True via --profile-memory-per-block in the launcher only when diagnosing
# per-block memory growth (e.g. an O(num_chunks) blowup like the one the
# chunk-activation-checkpointing fix addressed). Requires PROFILE_MEMORY to
# also be True; has no effect on its own.
PROFILE_MEMORY_PER_BLOCK = False

# When PROFILE_MEMORY is True, only emit memory logs for the first N optimizer
# steps (after that the picture is steady-state and the logs become noise).
# Per-block hook output (when PROFILE_MEMORY_PER_BLOCK is also True) is also
# limited to the first N forward passes.
PROFILE_MEMORY_STEPS = 5

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
# JSONL metrics path. Parent dirs are auto-created. Empty string disables
# JSONL; tqdm postfix stays on in that case. Default on so long runs
# always produce a queryable post-hoc record of loss / eval / grad norm /
# gate alpha / per-layer stats; override via --metrics-jsonl "" to disable.
METRICS_JSONL = "logs/metrics.jsonl"
# Per-feature toggles for the four observability modules.
LOG_GRAD_NORM = True
LOG_LAYER_STATS = True
LOG_GATE_ALPHA = True
# Periodic validation: every_n_steps, batch count per eval, holdout partition.
EVAL_EVERY = 2500
EVAL_BATCHES = 200
EVAL_HOLDOUT_FRACTION = 0.001
EVAL_HOLDOUT_SEED = 12345
EVAL_RESET_MEMORY_STATE = True

# ---------------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------------


class StreamingTextDataset(IterableDataset):
    """Stream tokenized text from HuggingFace datasets."""

    def __init__(
        self,
        dataset_name,
        subset,
        tokenizer,
        seq_len,
        seed=42,
        partition: str = "all",
        eval_cfg: EvalConfig | None = None,
    ):
        from datasets import load_dataset

        self.ds = load_dataset(
            dataset_name,
            subset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        ).shuffle(seed=seed, buffer_size=100000)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._buffer = []
        self.partition = partition
        self.eval_cfg = eval_cfg
        if partition in {"train", "eval"} and eval_cfg is None:
            raise ValueError(
                "StreamingTextDataset: partition 'train' or 'eval' requires eval_cfg."
            )

    def __iter__(self):
        row_index = 0
        for example in self.ds:
            row_index += 1
            if self.partition != "all":
                if example.get("id") is not None:
                    key = str(example.get("id"))
                elif example.get("url") is not None:
                    key = str(example.get("url"))
                else:
                    key = f"row-{row_index}"
                is_eval = is_eval_example(key, self.eval_cfg)
                if self.partition == "train" and is_eval:
                    continue
                if self.partition == "eval" and not is_eval:
                    continue

            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self._buffer.extend(tokens)

            while len(self._buffer) >= self.seq_len + 1:
                chunk = self._buffer[: self.seq_len + 1]
                self._buffer = self._buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


class SyntheticDataset(Dataset):
    """Fallback for quick testing.

    Note: this dataset does not seed ``numpy.random`` itself — the global
    numpy RNG is seeded once, centrally, by :func:`titans.utils.seed_everything`
    at the start of ``train()``. Constructing this dataset twice in the same
    process will therefore draw different samples, which is the correct
    behaviour for a single-seed training run.
    """

    def __init__(self, vocab_size, seq_len, num_samples=10000, seed=42):
        # ``seed`` is accepted for API compatibility with callers that used
        # to pass one, but the dataset draws from the global numpy RNG which
        # is seeded by seed_everything() at train() start.
        del seed  # unused; seeding is centralized
        self.data = np.random.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.from_numpy(self.data[idx, :-1].copy()).long(),
            "labels": torch.from_numpy(self.data[idx, 1:].copy()).long(),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_accel_cfg():
    """Return a lightweight object with the attributes
    ``init_accelerator_and_logging`` needs.

    pretrain.py is driven by module-level constants (no argparse), so we
    synthesize the shim here rather than at parse time.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
        wandb=False,
    )


def train():
    token = os.environ.get("HF_TOKEN")
    if not token and PUSH_CHECKPOINTS:
        logger.warning("HF_TOKEN not found — checkpoints will NOT be pushed to Hub")

    # Seed RNGs before anything that might touch CUDA or allocate tensors
    # (Accelerator construction below can initialize CUDA). Seeding early
    # also ensures CUBLAS_WORKSPACE_CONFIG is set before any cuBLAS call
    # when DETERMINISTIC=True.
    seed_everything(SEED, deterministic=DETERMINISTIC)

    bundle = init_accelerator_and_logging(_build_accel_cfg())
    accelerator = bundle.accelerator

    if accelerator.is_main_process:
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {MIXED_PRECISION}")
        logger.info(
            f"Model: dim={DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS}, "
            f"memory_layers={NUM_MEMORY_LAYERS}"
        )

    # Model
    config = TitansConfig(
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        vocab_size=VOCAB_SIZE,
        chunk_size=CHUNK_SIZE,
        window_size=WINDOW_SIZE,
        num_memory_layers=NUM_MEMORY_LAYERS,
        num_persistent_tokens=NUM_PERSISTENT_TOKENS,
        rope_proportion=ROPE_PROPORTION,
        use_tnt=True,
        use_attn_res=True,
        use_mca=True,
        memory_objective="huber",
        adaptive_window=True,
    )
    model = create_model("mac", config)
    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"Parameters: {num_params:,}")
    _log_mem("after model construction (CPU)")

    eval_cfg = EvalConfig(
        every_n_steps=int(EVAL_EVERY),
        num_batches=int(EVAL_BATCHES),
        reset_memory_state=bool(EVAL_RESET_MEMORY_STATE),
        holdout_fraction=float(EVAL_HOLDOUT_FRACTION),
        holdout_seed=int(EVAL_HOLDOUT_SEED),
    )

    # Dataset
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=token)
        # Defeat the transformers "Token indices sequence length is longer
        # than the specified maximum sequence length for this model (N > M)"
        # warning. StreamingTextDataset does its own chunking against SEQ_LEN;
        # the tokenizer's built-in max_length is irrelevant here and the
        # warning is pure log spam. Fires on the first long document of any
        # fresh process (Python warnings default to once-per-location).
        tokenizer.model_max_length = int(1e9)
        train_partition = "train" if eval_cfg.every_n_steps > 0 else "all"
        dataset = StreamingTextDataset(
            DATASET_NAME,
            DATASET_SUBSET,
            tokenizer,
            SEQ_LEN,
            seed=SEED,
            partition=train_partition,
            eval_cfg=eval_cfg if eval_cfg.every_n_steps > 0 else None,
        )
        logger.info(f"Streaming from {DATASET_NAME}/{DATASET_SUBSET}")

        eval_dataset = None
        if eval_cfg.every_n_steps > 0:
            eval_dataset = StreamingTextDataset(
                DATASET_NAME,
                DATASET_SUBSET,
                tokenizer,
                SEQ_LEN,
                seed=SEED,
                partition="eval",
                eval_cfg=eval_cfg,
            )
    except Exception as e:
        logger.warning(f"Could not load dataset ({e}), falling back to synthetic")
        dataset = SyntheticDataset(VOCAB_SIZE, SEQ_LEN)
        eval_dataset = None

    dataloader = make_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=int(os.environ.get("NUM_WORKERS", "4")),
        device_type=accelerator.device.type,
        streaming=isinstance(dataset, IterableDataset),
        drop_last=True,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = make_dataloader(
            eval_dataset,
            batch_size=BATCH_SIZE,
            num_workers=int(os.environ.get("NUM_WORKERS", "4")),
            device_type=accelerator.device.type,
            streaming=True,
            drop_last=True,
        )

    # Optimizer + scheduler
    optimizer = make_optimizer(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device_type=accelerator.device.type,
    )

    # accelerator.prepare() wraps scheduler.step() so it only fires on real
    # optimizer updates (sync_gradients=True). All schedule units must be in
    # optimizer-step space, NOT mini-batch space, or the schedule will only
    # consume ~1/GRADIENT_ACCUMULATION_STEPS of itself over the run.
    total_opt_steps = max(1, MAX_STEPS // GRADIENT_ACCUMULATION_STEPS)
    warmup_opt_steps = max(1, int(total_opt_steps * WARMUP_RATIO))
    cosine_opt_steps = max(1, total_opt_steps - warmup_opt_steps)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_opt_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_opt_steps,
        eta_min=LR * 0.1,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_opt_steps],
    )

    if accelerator.is_main_process:
        logger.info(
            f"Scheduler: warmup={warmup_opt_steps} + cosine={cosine_opt_steps} "
            f"optimizer steps (peak LR={LR:.2e}, min LR={LR * 0.1:.2e})"
        )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    _log_mem("after accelerator.prepare")

    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)

    # Observability setup -- MetricsWriter + GateHookRegistry.
    metrics = build_metrics_writer(METRICS_JSONL, accelerator)
    gate_hooks: GateHookRegistry | None = None
    if LOG_GATE_ALPHA:
        gate_hooks = GateHookRegistry(accelerator.unwrap_model(model))

    # INVARIANT: _chunked_loss_fn MUST start from eval_states=None on every
    # batch. stash_memory_states() is a shallow list copy -- any state leaked
    # back into training through this function would corrupt the training
    # memory state via shared tensor references. If you change this function
    # to carry state across eval batches, deepen stash_memory_states first.
    def _chunked_loss_fn(
        m: torch.nn.Module, b: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Eval loss for the chunked Titans forward.

        Uses a fresh memory state per batch (no carry-over). Averages
        cross-entropy across all chunks.
        """
        chunks_e = b["input_ids"].split(CHUNK_SIZE, dim=1)
        label_chunks_e = b["labels"].split(CHUNK_SIZE, dim=1)
        eval_states = None
        total = torch.zeros((), dtype=torch.float32, device=b["input_ids"].device)
        for chunk_ids, chunk_labels in zip(chunks_e, label_chunks_e, strict=True):
            logits_e, eval_states, _ = m(chunk_ids, states=eval_states)
            total = total + F.cross_entropy(
                logits_e.reshape(-1, VOCAB_SIZE), chunk_labels.reshape(-1)
            )
        return total / max(1, len(chunks_e))

    # Opt-in torch.compile (COMPILE=1). No-op on CPU or when use_attn_res.
    model = maybe_compile(
        model,
        enabled=bool(int(os.environ.get("COMPILE", "0"))),
        device_type=accelerator.device.type,
        use_attn_res=getattr(config, "use_attn_res", False),
    )

    # Instrument each block with memory snapshots around core_forward and
    # around the memory module's forward call. No-op when PROFILE_MEMORY is
    # False or when CUDA is unavailable. Monkey-patches `core_forward` (which
    # process_chunk calls directly, bypassing PyTorch's standard forward
    # hook system) and registers a regular forward hook on `block.memory`
    # (which IS called via __call__ from inside core_forward).
    _instrument_blocks_for_memory(accelerator.unwrap_model(model))

    # Hub setup
    if PUSH_CHECKPOINTS and token and accelerator.is_main_process:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(HUB_REPO, exist_ok=True, private=True)
        # Save config
        config_path = Path(tempfile.mkdtemp()) / "config.json"
        config_path.write_text(json.dumps(config.to_dict(), indent=2))
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
            repo_id=HUB_REPO,
        )

    # ---------------------------------------------------------------------------
    # Resume from Hub checkpoint
    # ---------------------------------------------------------------------------
    # pretrain.py never touches a local checkpoint dir at train time — the
    # save flow uses tempfile.TemporaryDirectory + hf_hub_api.upload_file.
    # We still call setup_checkpoint_dir up front so the local checkpoints/
    # directory exists for any ancillary artifacts (e.g., config.json, token
    # sidecars) and so pretrain shares the plan-3 helper with sft/lora/dpo/rlvr.
    # RESUME_FROM is a Hub filename (not a local path), so we intentionally do
    # not pass it here — the Hub resume flow at lines 538-614 handles that.
    setup_checkpoint_dir(CHECKPOINT_DIR)

    global_step = 0
    memory_states = None
    running_loss = 0.0

    if RESUME_FROM is not None:
        logger.info(f"Resuming from {HUB_REPO}/{RESUME_FROM} ...")
        from huggingface_hub import hf_hub_download

        ckpt_local = hf_hub_download(
            repo_id=HUB_REPO,
            filename=RESUME_FROM,
            token=token,
        )

        # For safetensors, also download the .meta.pt sidecar (holds step,
        # optimizer, scheduler).  The sidecar lives at the same stem with a
        # .meta.pt extension.
        if RESUME_FROM.endswith(".safetensors"):
            sidecar_filename = RESUME_FROM.replace(".safetensors", ".meta.pt")
            try:
                sidecar_local = hf_hub_download(
                    repo_id=HUB_REPO,
                    filename=sidecar_filename,
                    token=token,
                )
                # Place sidecar next to checkpoint so load_checkpoint finds it
                from pathlib import Path as _P

                expected = _P(ckpt_local).with_suffix(".meta.pt")
                if not expected.exists():
                    import shutil

                    shutil.copy2(sidecar_local, expected)
                logger.info(f"Downloaded metadata sidecar: {sidecar_filename}")
            except Exception as e:
                logger.warning(
                    f"No metadata sidecar found ({e}), step/optimizer/scheduler will reset"
                )

        checkpoint = load_checkpoint(ckpt_local, weights_only=False)

        # Resolve the resume step early so the scheduler block below can
        # fast-forward to it if we end up skipping the saved scheduler
        # state. Falls back to parsing the filename when the checkpoint
        # metadata sidecar is missing (e.g. safetensors-only uploads).
        _step_from_ckpt = checkpoint.get("step", 0)
        if _step_from_ckpt == 0 and "step_" in RESUME_FROM:
            import re as _re

            _m = _re.search(r"step_(\d+)", RESUME_FROM)
            if _m:
                _step_from_ckpt = int(_m.group(1))

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint["model"])

        optimizer_state_loaded = False
        if "optimizer" in checkpoint:
            # If the checkpoint carries optimizer_param_names (written by
            # post-PR#16 save paths), rebuild the state_dict keyed by
            # live param positions via name matching. This survives param-
            # order drift — module refactors or feature-flag toggles can
            # reorder named_parameters() between save and load, and
            # Optimizer.load_state_dict maps positionally. Legacy
            # checkpoints without the names list fall through to the
            # shape-compatibility check from PR #15.
            opt_state_dict = checkpoint["optimizer"]
            saved_names = checkpoint.get("optimizer_param_names")
            if saved_names is not None:
                live_names = [
                    name
                    for name, _ in accelerator.unwrap_model(model).named_parameters()
                ]
                opt_state_dict, preserved, dropped = remap_optimizer_state_by_name(
                    opt_state_dict, saved_names, live_names
                )
                logger.info(
                    "[observability] remapped %d/%d optimizer state "
                    "entries by name (%d saved entries dropped as not "
                    "present in live model)",
                    preserved,
                    len(live_names),
                    dropped,
                )

            # Shape-check the (possibly remapped) state against the live
            # optimizer BEFORE loading. This catches both positional drift
            # on legacy checkpoints AND param-shape changes that name-
            # based remap can't rescue.
            compatible, mismatches, checked = is_optimizer_state_compatible(
                optimizer, opt_state_dict
            )
            if compatible:
                optimizer.load_state_dict(opt_state_dict)
                migrated, seen = move_optimizer_state_to_params(optimizer)
                logger.info(
                    f"[observability] migrated {migrated}/{seen} optimizer "
                    "state tensors after load_state_dict (device + dtype "
                    "coerced to match live params)"
                )
                optimizer_state_loaded = True
            else:
                logger.warning(
                    "[observability] optimizer state from %s has "
                    "param-order drift vs current model: %d/%d state "
                    "slots have shape mismatch. Skipping optimizer state "
                    "restore to avoid fused-Adam crash; momentum "
                    "estimates will re-calibrate over the first few "
                    "hundred training steps.",
                    RESUME_FROM,
                    mismatches,
                    checked,
                )
            init_count, total = initialize_missing_optimizer_state(optimizer)
            logger.info(
                f"[observability] initialized {init_count}/{total} "
                "optimizer state slots for params missing state after "
                "load_state_dict"
            )
        if "scheduler" in checkpoint:
            # Scheduler and optimizer form a pair: a saved scheduler's
            # last_epoch and _last_lr were observed against a specific
            # optimizer-step trajectory. If we skipped the optimizer
            # load due to drift, replaying that scheduler state is
            # nonsensical — we'd be jumping the LR schedule into a
            # position the momentum estimates know nothing about. Load
            # only when optimizer state loaded cleanly; otherwise fast-
            # forward a fresh scheduler to the resume point so the first
            # few training steps don't replay warmup from scratch.
            if optimizer_state_loaded:
                scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                import warnings as _warnings

                target_opt_step = max(
                    0,
                    _step_from_ckpt // GRADIENT_ACCUMULATION_STEPS,
                )
                with _warnings.catch_warnings():
                    # scheduler.step() before optimizer.step() logs a
                    # UserWarning; suppress since we're intentionally
                    # advancing the schedule without training iterations.
                    _warnings.simplefilter("ignore")
                    for _ in range(target_opt_step):
                        scheduler.step()
                logger.warning(
                    "[observability] skipped scheduler state load "
                    "(optimizer state was incompatible); fast-forwarded "
                    "fresh scheduler by %d opt steps to match resume "
                    "position",
                    target_opt_step,
                )

        # global_step was computed earlier (as _step_from_ckpt) so the
        # scheduler fast-forward path could reach it. Reuse that value;
        # emit the "inferred from filename" warning path only when the
        # sidecar was missing.
        global_step = _step_from_ckpt
        if checkpoint.get("step", 0) == 0 and global_step != 0:
            logger.warning(
                f"No step in checkpoint metadata, inferred step={global_step} "
                f"from filename (optimizer/scheduler state NOT restored)"
            )

        logger.info(f"Resumed at step {global_step}, training to {MAX_STEPS}")

        # Also try to load memory states — strip any extension to get the stem
        resume_stem = RESUME_FROM.rsplit(".", 1)[
            0
        ]  # works for both .pt and .safetensors
        mem_filename = resume_stem.replace("step_", "memory_step_") + ".npz"
        if "final" in RESUME_FROM:
            mem_filename = "checkpoints/memory_final.npz"
        try:
            from titans.memory_dump import load_memory_states

            mem_local = hf_hub_download(
                repo_id=HUB_REPO, filename=mem_filename, token=token
            )
            memory_states = load_memory_states(
                mem_local,
                device=accelerator.device,
                reset_for_inference=False,
            )
            logger.info(f"Loaded memory states from {mem_filename}")
        except Exception as e:
            logger.info(f"No memory states found ({e}), starting fresh")

        del checkpoint

    model.train()
    pbar = tqdm(
        total=MAX_STEPS,
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    try:
        for batch in dataloader:
            if global_step >= MAX_STEPS:
                break

            # Memory profiling for the first few steps. After PROFILE_MEMORY_STEPS
            # the helpers no-op (gated by global_step) so production runs aren't
            # spammed.
            _profile_this_step = PROFILE_MEMORY and global_step < PROFILE_MEMORY_STEPS
            if _profile_this_step:
                _reset_max_mem()
                _log_mem(f"step {global_step:03d}: batch loaded")

            # Initialize per-step observability accumulators.
            _pre_clip_grad_norm: float | None = None

            try:
                with accelerator.accumulate(model):
                    chunks = batch["input_ids"].split(CHUNK_SIZE, dim=1)
                    label_chunks = batch["labels"].split(CHUNK_SIZE, dim=1)
                    num_chunks = len(chunks)
                    batch_loss = 0.0

                    for chunk_ids, chunk_labels in zip(chunks, label_chunks):
                        logits, memory_states, _ = model(
                            chunk_ids, states=memory_states
                        )
                        if _profile_this_step:
                            _log_mem(f"step {global_step:03d}: after forward")
                        chunk_loss = F.cross_entropy(
                            logits.reshape(-1, VOCAB_SIZE), chunk_labels.reshape(-1)
                        )
                        accelerator.backward(chunk_loss / num_chunks)
                        if _profile_this_step:
                            _log_mem(f"step {global_step:03d}: after backward")
                        batch_loss += chunk_loss.item() / num_chunks

                        # Truncated BPTT: detach state at chunk boundary
                        if memory_states is not None:
                            memory_states = [
                                s.detach() if s is not None else None
                                for s in memory_states
                            ]

                    # Pre-clip grad norm; guarded by sync_gradients so accum
                    # off-steps are skipped.
                    if accelerator.sync_gradients:
                        if LOG_GRAD_NORM:
                            _pre_clip_grad_norm = global_grad_norm(
                                accelerator.unwrap_model(model)
                            )
                        accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                        # Fused Adam requires per-param stride parity across
                        # (param, grad, exp_avg, exp_avg_sq). Some backward
                        # ops — most notably SDPA in the first block, where
                        # BlockAttnRes takes the single-source shortcut and
                        # returns the input tensor unchanged — can emit grads
                        # whose memory format diverges from the contiguous
                        # param. When that happens the fused CUDA kernel's
                        # check_fast_path_restrictions trips; the surfaced
                        # error message ("params, grads, exp_avgs, and
                        # exp_avg_sqs must have same dtype, device, and
                        # layout") is misleadingly narrow — the underlying
                        # check also covers stride parity. Forcing grads
                        # contiguous here restores the invariant the kernel
                        # actually enforces.
                        _fixed_grads = 0
                        _total_grads = 0
                        for _p in model.parameters():
                            if _p.grad is None:
                                continue
                            _total_grads += 1
                            if not _p.grad.is_contiguous():
                                _p.grad = _p.grad.contiguous()
                                _fixed_grads += 1
                        if not getattr(train, "_titans_logged_grad_contig", False):
                            train._titans_logged_grad_contig = True  # type: ignore[attr-defined]
                            logger.info(
                                "[observability] forced %d/%d grads "
                                "contiguous on first sync step "
                                "(global_step=%d)",
                                _fixed_grads,
                                _total_grads,
                                global_step,
                            )

                        # One-shot emulation of the C++ check that actually
                        # fires ("params, grads, exp_avgs, and exp_avg_sqs
                        # must have same dtype, device, and layout"). The
                        # real check — check_fast_path_restrictions in
                        # aten/src/ATen/native/ForeachUtils.h — verifies
                        #   1) dtype / device / layout==strided AND
                        #      is_non_overlapping_and_dense() on every
                        #      tensor, and
                        #   2) sizes + strides match at matching positions
                        #      across (params, grads, exp_avgs, exp_avg_sqs),
                        #      ignoring size-1 dims in the stride check.
                        # Histograms + contig checks have ruled out (1). This
                        # block isolates (2) by walking each grad-bearing
                        # param's quartet and logging the exact attributes
                        # of any that violates the per-position check.
                        if not getattr(train, "_titans_logged_quartet", False):
                            train._titans_logged_quartet = True  # type: ignore[attr-defined]
                            _names = {
                                id(p): name
                                for name, p in accelerator.unwrap_model(
                                    model
                                ).named_parameters()
                            }
                            _violations: list[tuple[str, str]] = []
                            _nond_violations: list[tuple[str, str, str]] = []
                            for group in optimizer.param_groups:
                                for p in group["params"]:
                                    if p.grad is None:
                                        continue
                                    state = optimizer.state.get(p, {})
                                    ea = state.get("exp_avg")
                                    es = state.get("exp_avg_sq")
                                    if ea is None or es is None:
                                        continue
                                    name = _names.get(id(p), "<unmapped>")
                                    # is_non_overlapping_and_dense() is
                                    # C++-only in PyTorch 2.7 (not exposed
                                    # via Python). is_contiguous() is a
                                    # close proxy: a C-contiguous tensor
                                    # is always non-overlapping-and-dense.
                                    # The converse isn't strictly true, but
                                    # our model has no channels_last or
                                    # exotic-stride tensors — prior diags
                                    # and checkpoint inspection confirm
                                    # every param/state tensor is
                                    # contig=True.
                                    for tag, t in (
                                        ("p", p),
                                        ("grad", p.grad),
                                        ("exp_avg", ea),
                                        ("exp_avg_sq", es),
                                    ):
                                        if not t.is_contiguous():
                                            _nond_violations.append(
                                                (name, tag, str(t.stride()))
                                            )
                                    # Per-position size + non-size-1-dim
                                    # stride check against p.
                                    p_sizes = tuple(p.shape)
                                    p_strides = tuple(p.stride())
                                    for tag, t in (
                                        ("grad", p.grad),
                                        ("exp_avg", ea),
                                        ("exp_avg_sq", es),
                                    ):
                                        t_sizes = tuple(t.shape)
                                        t_strides = tuple(t.stride())
                                        if t_sizes != p_sizes:
                                            _violations.append(
                                                (
                                                    f"{name}:{tag}",
                                                    f"size {t_sizes} != p {p_sizes}",
                                                )
                                            )
                                            continue
                                        for dim, (s, ps, ts) in enumerate(
                                            zip(p_sizes, p_strides, t_strides)
                                        ):
                                            if s == 1:
                                                continue
                                            if ps != ts:
                                                _violations.append(
                                                    (
                                                        f"{name}:{tag}",
                                                        f"dim{dim} size={s} "
                                                        f"p_stride={ps} "
                                                        f"{tag}_stride={ts}",
                                                    )
                                                )
                            logger.info(
                                "[diag] fast_path check: %d non_overlap "
                                "violations, %d size/stride violations",
                                len(_nond_violations),
                                len(_violations),
                            )
                            for n, tag, st in _nond_violations[:10]:
                                logger.info(
                                    "[diag]   non_overlap: %s tag=%s stride=%s",
                                    n,
                                    tag,
                                    st,
                                )
                            for spec, detail in _violations[:20]:
                                logger.info("[diag]   size/stride: %s %s", spec, detail)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if _profile_this_step:
                        _log_mem(f"step {global_step:03d}: after optimizer step")
            except torch.cuda.OutOfMemoryError as oom:
                logger.error(f"[mem] CUDA OOM at step {global_step}")
                logger.error(f"[mem] {oom}")
                if torch.cuda.is_available():
                    summary = torch.cuda.memory_summary(abbreviated=True)
                    logger.error(f"[mem] memory_summary:\n{summary}")
                raise

            # Per-layer stats (memory state + weight norms) across all blocks.
            _layer_stats = None
            if LOG_LAYER_STATS and memory_states is not None:
                try:
                    _layer_stats = collect_layer_stats(
                        accelerator.unwrap_model(model), memory_states
                    )
                except Exception:  # noqa: BLE001 -- logging must not crash training
                    _layer_stats = None

            # Optional per-batch global memory state reset.
            # When RESET_GLOBAL_STATE_PER_BATCH is False, still reset during warmup
            # so gates learn reasonable values before state begins carrying.
            reset_this_batch = (
                RESET_GLOBAL_STATE_PER_BATCH or global_step < STATE_CARRY_WARMUP_STEPS
            )
            if reset_this_batch and memory_states is not None:
                unwrapped_for_reset = accelerator.unwrap_model(model)
                reset_batch_size = batch["input_ids"].shape[0]
                new_memory_states = []
                for block, state in zip(unwrapped_for_reset.blocks, memory_states):
                    if state is None:
                        new_memory_states.append(None)
                        continue
                    global_mem = getattr(
                        getattr(block, "memory", None), "global_memory", None
                    )
                    if global_mem is not None and hasattr(state, "global_state"):
                        fresh_global = global_mem.init_state(reset_batch_size)
                        new_memory_states.append(
                            replace(state, global_state=fresh_global)
                        )
                    else:
                        new_memory_states.append(state)
                memory_states = new_memory_states

            running_loss += batch_loss
            global_step += 1
            pbar.update(1)

            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                lr = optimizer.param_groups[0]["lr"]

                row: dict[str, Any] = {"loss": avg_loss, "lr": lr}
                if _pre_clip_grad_norm is not None:
                    row["grad/global_norm"] = _pre_clip_grad_norm
                if _layer_stats is not None:
                    row.update(_layer_stats.to_dict())
                if gate_hooks is not None:
                    row.update(gate_hooks.snapshot())
                    gate_hooks.clear()

                # Preserve the existing decay_bias probe -- block-0-only scalar
                # that is distinct from gate/alpha_mean.
                try:
                    unwrapped_for_log = accelerator.unwrap_model(model)
                    block0 = unwrapped_for_log.blocks[0]
                    gate_proj = getattr(
                        getattr(getattr(block0, "memory", None), "global_memory", None),
                        "memory",
                        None,
                    )
                    gate_proj = getattr(gate_proj, "gate_decay_proj", None)
                    if gate_proj is None:
                        gate_proj = getattr(
                            getattr(block0, "memory", None), "gate_decay_proj", None
                        )
                    if gate_proj is not None:
                        row["decay_bias"] = float(gate_proj.bias.item())
                except Exception:  # noqa: BLE001
                    pass

                metrics.log(global_step, **row)
                metrics.tqdm_summary(pbar, global_step, **row)
                running_loss = 0.0

            if (
                eval_loader is not None
                and eval_cfg.every_n_steps > 0
                and global_step % eval_cfg.every_n_steps == 0
            ):
                stashed = (
                    stash_memory_states(memory_states)
                    if eval_cfg.reset_memory_state
                    else None
                )
                eval_metrics_row = run_eval(
                    model=model,
                    eval_loader=eval_loader,
                    accelerator=accelerator,
                    cfg=eval_cfg,
                    loss_fn=_chunked_loss_fn,
                )
                if eval_cfg.reset_memory_state:
                    memory_states = restore_memory_states(stashed)
                metrics.log(global_step, **eval_metrics_row)
                metrics.tqdm_summary(pbar, global_step, **eval_metrics_row)

            if global_step % SAVE_EVERY == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                with tempfile.TemporaryDirectory() as tmpdir:
                    ckpt_stem = Path(tmpdir) / f"step_{global_step}"
                    # optimizer_param_names captures the saved param order
                    # so a future resume can name-remap optimizer state
                    # across named_parameters() drift (module refactors,
                    # feature toggles) without dropping momentum. See
                    # titans.scripts.remap_optimizer_state_by_name.
                    ckpt_files = save_checkpoint(
                        unwrapped.state_dict(),
                        ckpt_stem,
                        format=SAVE_FORMAT,
                        metadata={
                            "optimizer": optimizer.state_dict(),
                            "optimizer_param_names": [
                                name for name, _ in unwrapped.named_parameters()
                            ],
                            "scheduler": scheduler.state_dict(),
                            "config": config.to_dict(),
                            "step": global_step,
                        },
                    )

                    if PUSH_CHECKPOINTS and token:
                        for fpath in ckpt_files:
                            api.upload_file(
                                path_or_fileobj=str(fpath),
                                path_in_repo=f"checkpoints/{fpath.name}",
                                repo_id=HUB_REPO,
                            )
                        logger.info(
                            f"Pushed checkpoint step {global_step} to {HUB_REPO}"
                        )

                    # Also save memory states
                    if memory_states is not None:
                        mem_path = Path(tmpdir) / f"memory_step_{global_step}.npz"
                        save_memory_states(memory_states, mem_path)
                        if PUSH_CHECKPOINTS and token:
                            api.upload_file(
                                path_or_fileobj=str(mem_path),
                                path_in_repo=f"checkpoints/memory_step_{global_step}.npz",
                                repo_id=HUB_REPO,
                            )

        pbar.close()

        # Final checkpoint
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            with tempfile.TemporaryDirectory() as tmpdir:
                final_stem = Path(tmpdir) / "final"
                final_files = save_checkpoint(
                    unwrapped.state_dict(),
                    final_stem,
                    format=SAVE_FORMAT,
                    metadata={
                        "optimizer": optimizer.state_dict(),
                        "optimizer_param_names": [
                            name for name, _ in unwrapped.named_parameters()
                        ],
                        "scheduler": scheduler.state_dict(),
                        "config": config.to_dict(),
                        "step": global_step,
                    },
                )

                if PUSH_CHECKPOINTS and token:
                    for fpath in final_files:
                        api.upload_file(
                            path_or_fileobj=str(fpath),
                            path_in_repo=f"checkpoints/{fpath.name}",
                            repo_id=HUB_REPO,
                        )
                    if memory_states is not None:
                        mem_path = Path(tmpdir) / "memory_final.npz"
                        save_memory_states(memory_states, mem_path)
                        api.upload_file(
                            path_or_fileobj=str(mem_path),
                            path_in_repo="checkpoints/memory_final.npz",
                            repo_id=HUB_REPO,
                        )
                    logger.info(f"Final checkpoint pushed to {HUB_REPO}")

            logger.info(f"Training complete — {global_step} steps")

    finally:
        # Observability teardown -- runs on every exit path (normal, OOM,
        # KeyboardInterrupt, checkpoint upload failure). Without this, hooks
        # leak model references and buffered JSONL lines are lost.
        if gate_hooks is not None:
            gate_hooks.remove()
        metrics.close()


def _parse_args() -> None:
    """Minimal argparse shim for pretrain.

    pretrain.py is configured via module-level constants (DIM, LR, ...).
    This shim only exists so ``scripts/pretrain.py --help`` prints the
    shared Titans flag surface and exits 0, matching the migration
    smoke-test contract used by the other training scripts.

    Unknown args are ignored (nothing here drives runtime behaviour;
    edit the module-level constants above to change a run).
    """
    from titans.scripts import base_argparse_parser

    parser = base_argparse_parser(
        description=(
            "Titans pretraining on HuggingFace Jobs. "
            "Configuration is driven by module-level constants (DIM, LR, "
            "BATCH_SIZE, MAX_STEPS, ...) — command-line flags are accepted "
            "for --help compatibility with the shared training-script CLI "
            "surface but are ignored at runtime."
        )
    )
    parser.set_defaults(
        checkpoint_dir="checkpoints/pretrain",
        wandb_project="titans-pretrain",
    )
    parser.parse_known_args()


if __name__ == "__main__":
    _parse_args()
    try:
        train()
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
