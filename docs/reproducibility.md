# Reproducibility

This doc is the contract for what is bit-identical when you run OpenTitans
with a fixed `--seed`. Two identical invocations should produce either
identical outputs or provably-not-identical outputs — the table below tells
you which.

If something below is wrong in practice, the determinism CI test (see
[Running a bit-identical smoke test](#running-a-bit-identical-smoke-test))
should catch it.

## TL;DR — what to expect

| Setup | Bit-identical across runs? |
|-------|----------------------------|
| CPU-only + `--seed` + `--deterministic` | Yes |
| CPU-only + `--seed` (no `--deterministic`) | Yes — CPU is deterministic for the ops OpenTitans uses |
| Single GPU + `--seed` + `--deterministic` | Usually yes; PyTorch raises `RuntimeError` if a non-deterministic op has no deterministic variant |
| Single GPU + `--seed` (no `--deterministic`) | Close, but not provably zero run-to-run delta |
| Multi-GPU (DDP / FSDP / Accelerate multi-process) | No — collective ordering (all-reduce, all-gather) is not deterministic |
| Across PyTorch versions | No |
| Across CUDA / cuDNN versions | No |
| Across CPU micro-architectures (AVX2 vs AVX-512, Apple Silicon vs x86) | No for floating-point; yes for integer ops |

## What `--deterministic` actually does

Passing `--deterministic` to any training script (`pretrain.py`, `sft.py`,
`lora.py`, `dpo.py`, `rlvr.py`) calls:

1. `torch.use_deterministic_algorithms(True, warn_only=False)` — instructs
   PyTorch to pick deterministic kernels where available and raise
   `RuntimeError` on any op that has no deterministic equivalent.
2. `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` — required by cuBLAS
   for deterministic matmul selection. Must be set before CUDA is
   initialized, which is why `seed_everything` is called before the
   Accelerator is constructed.

Without `--deterministic`, the code still seeds `random`, `numpy.random`,
and PyTorch RNGs, so run-to-run divergence on the same hardware is usually
small, but not provably zero.

Expect a moderate speed hit with `--deterministic` on GPU. CPU runs are
essentially unaffected.

## What `seed_everything` seeds

`titans.utils.seed_everything(seed, deterministic=False)` is the single
entry point for RNG seeding in OpenTitans. It seeds:

- Python `random`
- NumPy `np.random` (the legacy global RNG — `np.random.default_rng(...)`
  instances are unaffected and must be seeded at construction)
- PyTorch CPU RNG (`torch.manual_seed`)
- PyTorch CUDA RNG on all devices (`torch.cuda.manual_seed_all`; no-op if
  CUDA is unavailable)

When `deterministic=True`, it additionally:

- Exports `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Calls `torch.use_deterministic_algorithms(True, warn_only=False)`

## What it does NOT seed

- **DataLoader worker RNGs.** If `num_workers > 0`, each worker gets an
  independent NumPy / Python RNG seed derived from PyTorch's base seed.
  Pass `worker_init_fn=lambda w: seed_everything(seed + w)` or
  `generator=torch.Generator().manual_seed(seed)` on your DataLoader if
  you need worker-side determinism.
- **HuggingFace tokenizer shuffling / dataset shuffle buffers.** Use
  `transformers.set_seed` or pass `seed=seed` to the HF dataset
  `.shuffle(...)` call. Our streaming datasets already take `seed` and
  pass it through.
- **CUDA kernel ordering under DDP / FSDP.** Not reachable without
  rewriting collective ops.
- **`np.random.Generator` instances** constructed before `seed_everything`
  runs. Only the legacy `np.random` module-level RNG is seeded.

## Where `seed_everything` is called

Each training script calls it exactly once, immediately after argparse /
config construction and **before** building the Accelerator, the model, or
the DataLoader:

- `scripts/pretrain.py` → `train()` entry (uses module-level `SEED` /
  `DETERMINISTIC` constants)
- `scripts/sft.py` → `train(config)` entry
- `scripts/lora.py` → `train(config)` entry
- `scripts/dpo.py` → `train(config)` entry
- `scripts/rlvr.py` → `train(config)` entry

Grepping `scripts/` for `torch.manual_seed`, `np.random.seed`, or
`random.seed` should return zero hits. If you see one, something drifted.

## Running a bit-identical smoke test

```bash
uv run pytest tests/test_reproducibility.py -v
```

The test builds a tiny `TitansMAC`, runs two forward passes on CPU with
the same seed, and asserts losses are identical to `1e-6` (they should be
exactly equal). It also verifies that different seeds produce different
losses, so a no-op implementation can't trivially pass.

If this test fails in CI, something seed-affecting got added without
routing through `seed_everything`. Bisect, find it, and route it.

---

[Back to docs index](README.md)
