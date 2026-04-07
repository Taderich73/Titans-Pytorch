# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "titans @ git+https://github.com/Taderich73/Titans-Pytorch.git",
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
Designed to run via: hf jobs uv run scripts/hf_pretrain.py --flavor a10g-large
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from titans import TitansConfig, TitansMAC
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory_dump import save_memory_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

# ---------------------------------------------------------------------------
# Configuration — edit these for your run
# ---------------------------------------------------------------------------

# Model — ~1.5B parameter MAC configuration (with TNT/AttnRes/MCA/Huber/AdaptiveWindow)
DIM = 1024
NUM_HEADS = 16
NUM_LAYERS = 20
VOCAB_SIZE = 50257
CHUNK_SIZE = 512
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

# Resume — set to a Hub checkpoint path to continue training, e.g.:
#   RESUME_FROM = "checkpoints/step_10000.pt"   (resumes from step 10000)
#   RESUME_FROM = "checkpoints/final.pt"         (resumes from final checkpoint)
#   RESUME_FROM = None                           (train from scratch)
RESUME_FROM = None

# Seed
SEED = 42

# ---------------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------------


class StreamingTextDataset(IterableDataset):
    """Stream tokenized text from HuggingFace datasets."""

    def __init__(self, dataset_name, subset, tokenizer, seq_len, seed=42):
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

    def __iter__(self):
        for example in self.ds:
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
    """Fallback for quick testing."""

    def __init__(self, vocab_size, seq_len, num_samples=10000, seed=42):
        np.random.seed(seed)
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


def train():
    token = os.environ.get("HF_TOKEN")
    if not token and PUSH_CHECKPOINTS:
        logger.warning("HF_TOKEN not found — checkpoints will NOT be pushed to Hub")

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
    )

    if accelerator.is_main_process:
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {MIXED_PRECISION}")
        logger.info(
            f"Model: dim={DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS}, "
            f"memory_layers={NUM_MEMORY_LAYERS}"
        )

    torch.manual_seed(SEED)

    # Model
    config = TitansConfig(
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        vocab_size=VOCAB_SIZE,
        chunk_size=CHUNK_SIZE,
        num_memory_layers=NUM_MEMORY_LAYERS,
        num_persistent_tokens=NUM_PERSISTENT_TOKENS,
        rope_proportion=ROPE_PROPORTION,
        use_tnt=True,
        use_attn_res=True,
        use_mca=True,
        memory_objective="huber",
        adaptive_window=True,
    )
    model = TitansMAC(config)
    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"Parameters: {num_params:,}")

    # Dataset
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=token)
        dataset = StreamingTextDataset(
            DATASET_NAME, DATASET_SUBSET, tokenizer, SEQ_LEN, seed=SEED
        )
        logger.info(f"Streaming from {DATASET_NAME}/{DATASET_SUBSET}")
    except Exception as e:
        logger.warning(f"Could not load dataset ({e}), falling back to synthetic")
        dataset = SyntheticDataset(VOCAB_SIZE, SEQ_LEN)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        drop_last=True,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
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
        checkpoint = load_checkpoint(ckpt_local, weights_only=False)

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        global_step = checkpoint.get("step", 0)
        logger.info(f"Resumed at step {global_step}, training to {MAX_STEPS}")

        # Also try to load memory states
        mem_filename = RESUME_FROM.replace(".pt", "").replace("step_", "memory_step_") + ".npz"
        if "final" in RESUME_FROM:
            mem_filename = "checkpoints/memory_final.npz"
        try:
            from titans.memory_dump import load_memory_states

            mem_local = hf_hub_download(
                repo_id=HUB_REPO, filename=mem_filename, token=token
            )
            memory_states = load_memory_states(mem_local, device=accelerator.device)
            logger.info(f"Loaded memory states from {mem_filename}")
        except Exception as e:
            logger.info(f"No memory states found ({e}), starting fresh")

        del checkpoint

    model.train()
    pbar = tqdm(total=MAX_STEPS, initial=global_step, desc="Training", disable=not accelerator.is_main_process)

    for batch in dataloader:
        if global_step >= MAX_STEPS:
            break

        with accelerator.accumulate(model):
            logits, memory_states = model(batch["input_ids"], states=memory_states)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE), batch["labels"].view(-1)
            )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if memory_states is not None:
            memory_states = [s.detach() for s in memory_states]

        running_loss += loss.item()
        global_step += 1
        pbar.update(1)

        if global_step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            running_loss = 0.0

        if global_step % SAVE_EVERY == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_stem = Path(tmpdir) / f"step_{global_step}"
                ckpt_files = save_checkpoint(
                    unwrapped.state_dict(),
                    ckpt_stem,
                    format=SAVE_FORMAT,
                    metadata={
                        "optimizer": optimizer.state_dict(),
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
                    logger.info(f"Pushed checkpoint step {global_step} to {HUB_REPO}")

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


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
