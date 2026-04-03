# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "titans @ git+https://huggingface.co/FlatFootInternational/titans",
#     "torch>=2.2.0",
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

import sys
import traceback

# Force stdout/stderr to flush immediately so HF Jobs logs capture everything
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=== Titans HF Jobs Training Script Starting ===", flush=True)

try:
    import json
    import logging
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, Dataset, IterableDataset
    from tqdm import tqdm

    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    from titans import TitansConfig, TitansMAC
    from titans.memory_dump import save_memory_states

    print("All imports successful", flush=True)
except Exception as e:
    print(f"IMPORT ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit these for your run
# ---------------------------------------------------------------------------

# Model
DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 12
VOCAB_SIZE = 32000
CHUNK_SIZE = 512
NUM_MEMORY_LAYERS = 2
NUM_PERSISTENT_TOKENS = 16

# Data
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-10BT"
TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"
SEQ_LEN = 2048

# Training
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LR = 4e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.03
MAX_STEPS = 10000
LOG_EVERY = 10
SAVE_EVERY = 2500
MIXED_PRECISION = "bf16"

# Hub persistence
HUB_REPO = "FlatFootInternational/titans-mac-512"  # Where to push checkpoints
PUSH_CHECKPOINTS = True

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
        ).shuffle(seed=seed, buffer_size=10000)
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
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, MAX_STEPS - warmup_steps)
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

    # Training loop
    global_step = 0
    memory_states = None
    running_loss = 0.0

    model.train()
    pbar = tqdm(total=MAX_STEPS, desc="Training", disable=not accelerator.is_main_process)

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
                ckpt_path = Path(tmpdir) / f"step_{global_step}.pt"
                torch.save(
                    {"model": unwrapped.state_dict(), "config": config.to_dict(), "step": global_step},
                    ckpt_path,
                )

                if PUSH_CHECKPOINTS and token:
                    api.upload_file(
                        path_or_fileobj=str(ckpt_path),
                        path_in_repo=f"checkpoints/step_{global_step}.pt",
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
            final_path = Path(tmpdir) / "final.pt"
            torch.save(
                {"model": unwrapped.state_dict(), "config": config.to_dict(), "step": global_step},
                final_path,
            )

            if PUSH_CHECKPOINTS and token:
                api.upload_file(
                    path_or_fileobj=str(final_path),
                    path_in_repo="checkpoints/final.pt",
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
