# DPO & RLVR Training — Design Spec

## Overview

Add two new training scripts to the Titans MLX project: `scripts/dpo.py` for Direct Preference Optimization and `scripts/rlvr.py` for Reinforcement Learning with Verifiable Rewards. Both follow the existing one-script-per-mode convention established by `pretrain.py`, `sft.py`, and `lora.py`.

### Training Pipeline Position

```
pretrain → sft → dpo or rlvr
```

Both scripts consume checkpoints from any prior stage and produce checkpoints in the same format.

---

## DPO Script (`scripts/dpo.py`)

### Supported Methods

- **Standard DPO** — Rafailov et al. Requires a reference model.
- **SimPO** — Reference-free. Uses length-normalized average log-probabilities.

Selected via `--method {dpo,simpo}`.

### Data Pipeline

**Input format:** HuggingFace preference datasets with `chosen` and `rejected` fields containing message lists with `role` and `content` keys (e.g., `allenai/Dolci-Instruct-DPO`).

**`DPOStreamingDataset`:**
- Streams from HuggingFace datasets
- Extracts `role` and `content` from each message, discards metadata fields (`country`, `hashed_ip`, `toxic`, `header`, etc.)
- Formats both chosen and rejected sequences as ChatML (`<|im_start|>role\n...<|im_end|>\n`)
- Tokenizes and pads/truncates to `--max-len`

**Batch shape:**
```python
{
    "chosen_ids": (batch, seq_len),
    "chosen_mask": (batch, seq_len),
    "rejected_ids": (batch, seq_len),
    "rejected_mask": (batch, seq_len),
}
```

### Loss Functions

**Standard DPO:**
```
log_ratio_chosen = sum(log_pi(chosen) - log_ref(chosen))
log_ratio_rejected = sum(log_pi(rejected) - log_ref(rejected))
loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
```
- `beta` (default: 0.1) controls KL divergence penalty strength

**SimPO:**
```
avg_logp_chosen = mean(log_pi(chosen))
avg_logp_rejected = mean(log_pi(rejected))
loss = -log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected - gamma)))
```
- `gamma` (default: 1.0) is the reward margin
- No reference model needed — length normalization acts as implicit reward

### Reference Model Handling

**LoRA mode** (`--lora`): The frozen base model weights serve as the reference model. A forward pass without LoRA contributions produces reference log-probs; a forward pass with LoRA produces policy log-probs. Single model in memory.

Implementation: `compute_reference_logprobs` helper that temporarily bypasses LoRA adapter forward passes (zeros out `lora_B` or skips the adapter) to get reference log-probs without loading a second model.

**Full-parameter mode**: Loads a second frozen copy of the model as the reference. Double memory cost. Script warns at startup about memory requirements.

**SimPO mode**: No reference model needed in either LoRA or full-parameter mode.

### CLI Interface

```bash
python scripts/dpo.py \
    --checkpoint checkpoints/my-sft-model \
    --dataset allenai/Dolci-Instruct-DPO \
    --method {dpo,simpo} \
    --beta 0.1 \
    --gamma 1.0 \
    --lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets attn,ffn \
    --max-len 2048 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --lr 5e-7 \
    --weight-decay 0.1 \
    --warmup-ratio 0.1 \
    --epochs 3 \
    --eval-every 500 \
    --save-every 1000 \
    --checkpoint-dir checkpoints/dpo-run \
    --wandb --wandb-project titans-dpo
```

### Hyperparameter Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--method` | `dpo` | `dpo` or `simpo` |
| `--beta` | `0.1` | KL penalty strength |
| `--gamma` | `1.0` | SimPO reward margin |
| `--lr` | `5e-7` | Lower than SFT |
| `--max-len` | `2048` | Max sequence length |
| `--batch-size` | `2` | Per-step batch size |
| `--gradient-accumulation-steps` | `8` | Effective batch = 16 |

---

## RLVR Script (`scripts/rlvr.py`)

### Supported Methods

- **GRPO** — Group Relative Policy Optimization. Estimates baselines from groups of rollouts. No value model needed.
- **REINFORCE with baseline** — Single-sample with exponential moving average baseline.

Selected via `--method {grpo,reinforce}`.

### Operating Modes

Selected via `--mode {offline,live}`:

**Offline mode** — uses pre-computed rollouts from a dataset (e.g., `allenai/Dolci-Think-RL-7B`). The dataset provides `prompt`, `ground_truth`, `outputs` (rollouts), and `passrate`.

**Live mode** — generates fresh rollouts at training time, scores them with a verifier function against `ground_truth`.

### Data Pipeline

**`OfflineRLDataset`** (offline mode):
- Streams from HuggingFace
- Extracts `prompt`, `ground_truth`, `outputs` (pre-generated rollouts)
- Tokenizes prompt + each rollout as ChatML
- Computes binary rewards from `ground_truth` matching

**Batch shape (both modes):**
```python
{
    "prompt_ids": (batch, prompt_len),
    "rollout_ids": (batch, num_rollouts, seq_len),
    "rollout_masks": (batch, num_rollouts, seq_len),
    "rewards": (batch, num_rollouts),
}
```

**`LiveRLDataset`** (live mode):
- Streams prompts + ground truth only
- At each training step:
  1. Generate `--num-rollouts` completions per prompt via temperature sampling
  2. Run verifier function to score each rollout
  3. Produce same batch shape as offline

### Verifier Framework

Pluggable verifier functions with signature:
```python
def verify(response: str, ground_truth: list[str]) -> float:
    """Returns reward in [0, 1]."""
```

**Built-in verifiers:**
- `exact_match` — strip, normalize whitespace, case-insensitive compare
- `numeric_match` — extract final number from response, compare to ground truth with configurable tolerance
- `code_exec` — execute generated code against test cases in a sandboxed subprocess, return pass/fail

**Custom verifiers:** `--verifier path/to/verifier.py:function_name` loads any Python function matching the signature above.

### Loss Functions

**GRPO:**
```
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
loss = -mean(advantages * log_pi(rollout)) + kl_beta * KL(pi || ref)
```
- Group-relative baseline eliminates need for a value model
- KL penalty is optional (`--kl-beta`, default: 0.0)
- Rollouts where all rewards are identical (all-pass or all-fail) produce zero advantage — skipped with a logged warning, no gradient signal

**REINFORCE with baseline:**
```
baseline = EMA(rewards)  # exponential moving average
advantage = reward - baseline
loss = -advantage * log_pi(rollout)
```
- `--ema-decay 0.99` controls baseline smoothness
- Works with `--num-rollouts 1` for minimal compute
- Higher variance than GRPO but lower cost

### Generation (Live Mode)

Reuses temperature sampling logic from `inference.py`:
- `--temperature 0.7` for rollout diversity
- `--max-new-tokens 2048` generation cap
- `--num-rollouts 8` per prompt (GRPO needs groups; REINFORCE can use 1)
- Stops at `<|im_end|>` token

### CLI Interface

```bash
python scripts/rlvr.py \
    --checkpoint checkpoints/my-sft-model \
    --dataset allenai/Dolci-Think-RL-7B \
    --mode {offline,live} \
    --method {grpo,reinforce} \
    --num-rollouts 8 \
    --verifier exact_match \
    --kl-beta 0.0 \
    --ema-decay 0.99 \
    --temperature 0.7 \
    --max-new-tokens 2048 \
    --lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets attn,ffn \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --lr 1e-6 \
    --weight-decay 0.1 \
    --warmup-ratio 0.03 \
    --max-steps 5000 \
    --eval-every 500 \
    --save-every 1000 \
    --checkpoint-dir checkpoints/rlvr-run \
    --wandb --wandb-project titans-rlvr
```

### Hyperparameter Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--method` | `grpo` | `grpo` or `reinforce` |
| `--mode` | `offline` | `offline` or `live` |
| `--num-rollouts` | `8` | Rollouts per prompt |
| `--kl-beta` | `0.0` | KL penalty (GRPO) |
| `--ema-decay` | `0.99` | Baseline decay (REINFORCE) |
| `--temperature` | `0.7` | Generation temperature (live) |
| `--max-new-tokens` | `2048` | Generation cap (live) |
| `--lr` | `1e-6` | Higher than DPO, lower than SFT |
| `--batch-size` | `2` | Per-step batch size |

---

## Shared Infrastructure

### Log-Probability Computation

Both scripts need per-token log-probs. Each script contains its own copy of:

```python
def compute_logprobs(model, input_ids, mask=None):
    logits, _ = model(input_ids)
    log_probs = log_softmax(logits[:, :-1], axis=-1)
    token_log_probs = gather(log_probs, input_ids[:, 1:])
    if mask is not None:
        token_log_probs = token_log_probs * mask[:, 1:]
    return token_log_probs  # (batch, seq_len - 1)
```

Duplicated across scripts rather than extracted into a shared module — consistent with the project's self-contained script convention.

### LoRA Integration

Both scripts import from `lora.py`:
- `wrap_lora_layers(model, targets, rank, alpha, dropout)`
- `save_adapters(model, path, metadata)`
- `load_adapters(model, path)`

Same `--lora-rank`, `--lora-targets`, `--lora-alpha` flags as existing LoRA training.

### Training Loop

Both scripts follow the established pattern from `pretrain.py` and `sft.py`:
- Adam optimizer with cosine LR schedule + linear warmup
- `sanitize_and_clip_grads` for NaN replacement + global norm clipping
- `apply_gradients` with weight tying re-application
- Gradient accumulation via `--gradient-accumulation-steps`
- Periodic evaluation, checkpointing, optional wandb logging

### Checkpoint Compatibility

Both scripts use the existing `save_checkpoint`/`load_checkpoint` format. Metadata in the checkpoint records which training stage produced it (`training_stage: "dpo"` or `training_stage: "rlvr"`). Checkpoints are interchangeable — you can chain DPO after RLVR or vice versa.

### What's Not Included

- No reward model training — RLVR uses deterministic verifiers, DPO is reward-model-free
- No PPO / value heads — GRPO and REINFORCE don't need them
- No multi-GPU — consistent with the rest of the MLX codebase (single-device)
- No IPO or other DPO variants — standard DPO + SimPO cover the practical range

---

## Target Datasets

| Script | Primary Dataset | Format |
|--------|----------------|--------|
| `dpo.py` | `allenai/Dolci-Instruct-DPO` | 260k preference pairs, message lists with `role`/`content` |
| `rlvr.py` | `allenai/Dolci-Think-RL-7B` | 102k prompts with rollouts across math/code/IF/chat |

Both scripts support any HuggingFace dataset matching the expected column schema.
