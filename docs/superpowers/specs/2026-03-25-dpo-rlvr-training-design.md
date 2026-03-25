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

Implementation: Add an `enabled: bool` attribute to `LoRALinear` (default `True`). When `enabled=False`, `__call__` returns only `self.base(x)`, skipping the LoRA delta. A `set_lora_enabled(model, enabled: bool)` utility walks the model tree and toggles all `LoRALinear.enabled` flags. Reference log-probs are computed with LoRA disabled; policy log-probs with LoRA enabled. This avoids array copies and MLX recompilation overhead.

**Full-parameter mode**: Loads a second frozen copy of the model as the reference. Double memory cost. Script warns at startup about memory requirements.

**SimPO mode**: No reference model needed in either LoRA or full-parameter mode.

### Memory State Handling

The model returns `(logits, states)`. For DPO, chosen and rejected sequences from the same example are independent — memory states are **reset (set to `None`) before each forward pass** (chosen, rejected, and reference). States are not carried between the two sequences of a pair, as they represent different continuations of the same prompt and cross-contamination would corrupt the loss signal.

### CLI Interface

```bash
python scripts/dpo.py \
    --model mac \
    --resume checkpoints/my-sft-model \
    --dataset allenai/Dolci-Instruct-DPO \
    --method {dpo,simpo} \
    --beta 0.1 \
    --gamma 1.0 \
    --lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets attn,ffn \
    --lora-dropout 0.0 \
    --max-len 2048 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --lr 5e-7 \
    --weight-decay 0.1 \
    --warmup-ratio 0.1 \
    --grad-clip 1.0 \
    --epochs 3 \
    --max-steps -1 \
    --eval-every 500 \
    --save-every 1000 \
    --checkpoint-dir checkpoints/dpo-run \
    --use-tnt \
    --use-attn-res \
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
| `--grad-clip` | `1.0` | Global gradient norm clip |
| `--max-steps` | `-1` | Step limit (-1 = use epochs) |

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
    "rollout_ids": (batch, num_rollouts, seq_len),   # padded to max length in group
    "rollout_masks": (batch, num_rollouts, seq_len),  # 1 for real tokens, 0 for padding
    "rewards": (batch, num_rollouts),
}
```
Rollouts have variable length — they are right-padded to the longest rollout in the batch. `rollout_masks` distinguishes real tokens from padding so log-prob sums exclude pad positions.

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
- `code_exec` — execute generated code against test cases via `subprocess.run` with timeout, no network access. Deferred to v2 if sandboxing proves complex; `exact_match` and `numeric_match` ship first

**Custom verifiers:** `--verifier path/to/verifier.py:function_name` loads any Python function matching the signature above.

### Loss Functions

**GRPO (with clipped importance ratios, per DeepSeekMath):**
```
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
ratio = exp(log_pi(rollout) - log_pi_old(rollout))
clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
loss = -mean(min(ratio * advantages, clipped_ratio * advantages)) + kl_beta * KL(pi || ref)
```
- `epsilon` (default: 0.2) controls the clipping range, same role as PPO's clip parameter
- `log_pi_old` is computed once before each optimization step and held fixed (no gradient)
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

### Memory State Handling

Same as DPO: memory states are **reset to `None` before each rollout forward pass**. Each rollout is an independent sequence — states must not leak between rollouts of the same prompt or across prompts.

### Generation (Live Mode)

Reuses temperature sampling logic from `inference.py`:
- `--temperature 0.7` for rollout diversity
- `--max-new-tokens 2048` generation cap
- `--num-rollouts 8` per prompt (GRPO needs groups; REINFORCE can use 1)
- Stops at `<|im_end|>` token

### CLI Interface

```bash
python scripts/rlvr.py \
    --model mac \
    --resume checkpoints/my-sft-model \
    --dataset allenai/Dolci-Think-RL-7B \
    --mode {offline,live} \
    --method {grpo,reinforce} \
    --num-rollouts 8 \
    --epsilon 0.2 \
    --verifier exact_match \
    --kl-beta 0.0 \
    --ema-decay 0.99 \
    --temperature 0.7 \
    --max-new-tokens 2048 \
    --lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets attn,ffn \
    --lora-dropout 0.0 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --lr 1e-6 \
    --weight-decay 0.1 \
    --warmup-ratio 0.03 \
    --grad-clip 1.0 \
    --max-steps 5000 \
    --eval-every 500 \
    --save-every 1000 \
    --checkpoint-dir checkpoints/rlvr-run \
    --use-tnt \
    --use-attn-res \
    --wandb --wandb-project titans-rlvr
```

### Hyperparameter Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--method` | `grpo` | `grpo` or `reinforce` |
| `--mode` | `offline` | `offline` or `live` |
| `--num-rollouts` | `8` | Rollouts per prompt |
| `--epsilon` | `0.2` | GRPO clipping range |
| `--kl-beta` | `0.0` | KL penalty (GRPO) |
| `--ema-decay` | `0.99` | Baseline decay (REINFORCE) |
| `--temperature` | `0.7` | Generation temperature (live) |
| `--max-new-tokens` | `2048` | Generation cap (live) |
| `--lr` | `1e-6` | Higher than DPO, lower than SFT |
| `--batch-size` | `2` | Per-step batch size |
| `--grad-clip` | `1.0` | Global gradient norm clip |

---

## Shared Infrastructure

### Log-Probability Computation

Both scripts need per-token log-probs. Each script contains its own copy (MLX-native, no PyTorch equivalents):

```python
def compute_logprobs(model, input_ids, mask=None):
    logits, _ = model(input_ids)
    # MLX log-softmax: logits - logsumexp(logits)
    log_probs = logits[:, :-1] - mx.logsumexp(logits[:, :-1], axis=-1, keepdims=True)
    # MLX gather: take_along_axis instead of PyTorch's gather
    token_log_probs = mx.take_along_axis(log_probs, input_ids[:, 1:, None], axis=-1).squeeze(-1)
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

Same `--lora-rank`, `--lora-targets`, `--lora-alpha`, `--lora-dropout` flags as existing LoRA training.

### Training Loop

Both scripts follow the established pattern from `pretrain.py` and `sft.py`:
- Adam optimizer with cosine LR schedule + linear warmup
- `sanitize_and_clip_grads` for NaN replacement + global norm clipping
- `apply_gradients` with weight tying re-application
- Gradient accumulation via `--gradient-accumulation-steps`
- Periodic evaluation, checkpointing, optional wandb logging

### Checkpoint Compatibility

Both scripts use the existing `save_checkpoint`/`load_checkpoint` format. A new `training_stage` metadata field is introduced — `"dpo"` or `"rlvr"` for these scripts. This is a new convention; existing scripts (`pretrain.py`, `sft.py`, `lora.py`) do not currently set this field but could be updated for consistency in a follow-up. Checkpoints are interchangeable — you can chain DPO after RLVR or vice versa.

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
