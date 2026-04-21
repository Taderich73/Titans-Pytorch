# Plan Audit Gap Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six remaining gaps identified by the 2026-04-21 audit of the nine 2026-04-16 plans against main: finish plan-3 migrations on `pretrain.py` and `inference.py`, add the SFT integration-fence test, run the ruff F401 pass, fix test-collection for DPO/RLVR plumbing tests, and correct the stale `TRIGGERED` docstring in `memory_checkpointer.py`.

**Architecture:** Six small independent surgical tasks. Each lands in its own commit so it stays bisectable. We do not refactor beyond what the plans originally required. Pretrain and inference migrations are narrow — only the `setup_checkpoint_dir` wiring is missing; everything else already landed. The test-collection fix is a `conftest.py` edit that adds `scripts/` to `sys.path`, so the `importlib.util.spec_from_file_location` loader in the DPO/RLVR test modules can resolve `scripts._common`. The SFT fence is a new test class; we fix re-export gaps only if the test flags them.

**Tech Stack:** Python 3.12, pytest, ruff, git.

---

### Task 1: Wire `setup_checkpoint_dir` into `scripts/pretrain.py`

**Files:**
- Modify: `scripts/pretrain.py` (import block at `scripts/pretrain.py:56-73`; `train()` output-dir handling around the checkpoint constants)
- Test: `tests/test_common.py` (extend `TestPretrainMigrationSmoke` — new assertion)

- [ ] **Step 1: Locate the in-file checkpoint-dir creation in pretrain.py**

Run:
```bash
grep -n "CHECKPOINT_DIR\|Path.*checkpoint\|mkdir" scripts/pretrain.py
```
Expected: finds the `CHECKPOINT_DIR = "checkpoints"` module constant and any `Path(...).mkdir(...)` call inside `train()`. Note the exact line numbers before editing.

- [ ] **Step 2: Write the failing assertion**

Open `tests/test_common.py`. Find `class TestPretrainMigrationSmoke`. Append a new method:

```python
    def test_pretrain_imports_setup_checkpoint_dir(self) -> None:
        """pretrain.py must import setup_checkpoint_dir from scripts._common."""
        import pathlib
        src = pathlib.Path("scripts/pretrain.py").read_text()
        assert "setup_checkpoint_dir" in src, (
            "pretrain.py did not import setup_checkpoint_dir; "
            "plan 3 Task 13 Step 5 required it."
        )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_common.py::TestPretrainMigrationSmoke::test_pretrain_imports_setup_checkpoint_dir -v`
Expected: FAIL — `setup_checkpoint_dir not imported in pretrain.py`.

- [ ] **Step 4: Add `setup_checkpoint_dir` to the pretrain import block**

Edit `scripts/pretrain.py`. In both import branches of the try/except at `scripts/pretrain.py:56-73`, add `setup_checkpoint_dir` to the import list. After the edit the block should read:

```python
try:
    from scripts._common import (  # type: ignore[import-not-found]
        build_titans_config,  # noqa: F401 — re-exported for API parity
        create_model,
        init_accelerator_and_logging,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _common import (  # type: ignore[no-redef]
        build_titans_config,  # noqa: F401 — re-exported for API parity
        create_model,
        init_accelerator_and_logging,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
    )
```

- [ ] **Step 5: Replace the mkdir block inside `train()`**

Find the block where `CHECKPOINT_DIR` / output-dir bootstrapping happens (identified in Step 1). Replace the hand-rolled directory creation with a single call:

```python
    ckpt_setup = setup_checkpoint_dir(
        CHECKPOINT_DIR,
        resume_path=RESUME_FROM if RESUME_FROM else None,
    )
    output_dir = ckpt_setup.output_dir
    if ckpt_setup.resume_path is not None and global_step == 0:
        # Use the parsed step as a conservative fallback when sidecar lookups fail.
        fallback_resume_step = ckpt_setup.resume_step
    else:
        fallback_resume_step = 0
```

If pretrain has no `mkdir` call at all (because HF Jobs creates the dir for it), a minimum valid integration is to keep only the first three lines above (no fallback_resume_step). In that case confirm with `grep -n "mkdir" scripts/pretrain.py` returning zero hits after the edit — no `Path(...).mkdir(...)` on the output dir remains.

- [ ] **Step 6: Run tests**

Run:
- `uv run pytest tests/test_common.py::TestPretrainMigrationSmoke -v`
- `uv run python scripts/pretrain.py --help`

Expected: both PASS. `--help` must exit zero; grep for `mkdir` on the output dir in pretrain.py must find no occurrences.

- [ ] **Step 7: Commit**

```bash
git add scripts/pretrain.py tests/test_common.py
git commit -m "$(cat <<'EOF'
refactor(pretrain): wire setup_checkpoint_dir from _common

Completes plan 3 Task 13 Step 5 — pretrain.py now uses the shared
checkpoint-dir helper instead of bespoke Path.mkdir(). Adds a migration
guard in TestPretrainMigrationSmoke so a future regression fails loudly.
EOF
)"
```

---

### Task 2: Wire `setup_checkpoint_dir` into `scripts/inference.py`

> **Amendment (during implementation):** Using `setup_checkpoint_dir` for `--checkpoint` validation produces a mis-labeled error ("--resume file not found"), silently creates typo'd parent directories, and regresses extensionless-path lookup. Inference is a read path; this task pivoted to a direct existence check that mirrors `load_checkpoint`'s extension resolution. The guard test was updated to enforce the new behavior.

**Files:**
- Modify: `scripts/inference.py:27-36` (import block); `scripts/inference.py:269` (`load_model` call site)
- Test: `tests/test_common.py` (extend `TestInferenceMigrationSmoke`)

- [ ] **Step 1: Write the failing assertion**

Open `tests/test_common.py`. Find `class TestInferenceMigrationSmoke`. Append:

```python
    def test_inference_imports_setup_checkpoint_dir(self) -> None:
        """inference.py must import setup_checkpoint_dir for --checkpoint validation."""
        import pathlib
        src = pathlib.Path("scripts/inference.py").read_text()
        assert "setup_checkpoint_dir" in src, (
            "inference.py did not import setup_checkpoint_dir; "
            "plan 3 Task 14 Step 3 required it for --checkpoint validation."
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_common.py::TestInferenceMigrationSmoke::test_inference_imports_setup_checkpoint_dir -v`
Expected: FAIL — `setup_checkpoint_dir not imported in inference.py`.

- [ ] **Step 3: Add the import**

Edit `scripts/inference.py:27-36`. Both branches of the try/except should read:

```python
try:
    from scripts._common import (  # type: ignore[import-not-found]
        MODEL_CLASSES,
        create_model,
        setup_checkpoint_dir,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _common import (  # type: ignore[no-redef]
        MODEL_CLASSES,
        create_model,
        setup_checkpoint_dir,
    )
```

- [ ] **Step 4: Validate `--checkpoint` through `setup_checkpoint_dir`**

In `scripts/inference.py`, find `main()` and locate the line where `args.checkpoint` is first used (just before `model, config = load_model(args.checkpoint, device, variant=args.model)` at `scripts/inference.py:269`). Insert a validation call immediately before that line:

```python
    # Validate --checkpoint exists via the shared helper (raises
    # FileNotFoundError with a helpful message when the file is missing).
    # Using the parent directory as output_dir is a no-op in inference
    # (we never write to it); the helper is here purely for its
    # resume_path existence check and filename step parsing.
    ckpt_file = Path(args.checkpoint)
    setup_checkpoint_dir(str(ckpt_file.parent), resume_path=str(ckpt_file))
```

Confirm `from pathlib import Path` is already at the top of `scripts/inference.py:17` (it is — do not duplicate).

- [ ] **Step 5: Run tests**

Run:
- `uv run pytest tests/test_common.py::TestInferenceMigrationSmoke -v`
- `uv run python scripts/inference.py --help`

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/inference.py tests/test_common.py
git commit -m "$(cat <<'EOF'
refactor(inference): use setup_checkpoint_dir for --checkpoint validation

Completes plan 3 Task 14 Step 3.2 — inference.py now validates the
--checkpoint file through the shared helper (FileNotFoundError with a
clearer message) instead of deferring the error to load_checkpoint.
EOF
)"
```

---

### Task 3: Add `TestSFTIntegrationFence` and fix SFT re-exports

**Files:**
- Modify: `tests/test_common.py` (new class)
- Modify (only if fence fails): `scripts/sft.py:59-83` (add `format_chatml` and `build_loss_mask` to the `_common` imports)

- [ ] **Step 1: Add the failing fence test**

Open `tests/test_common.py`. Add a new class (append at the end of the file):

```python
class TestSFTIntegrationFence:
    """sft.py must still expose create_model / build_titans_config /
    tokenize_chat / format_chatml / build_loss_mask as re-exports
    (or at least importable names) so external callers don't break.

    This is the fence test from plan 3 Task 9 Step 1 that was never
    added when the migration landed.
    """

    def test_sft_re_exports(self) -> None:
        from scripts import sft

        assert callable(sft.create_model)
        assert callable(sft.build_titans_config)
        assert callable(sft.tokenize_chat)
        assert callable(sft.format_chatml)
        assert callable(sft.build_loss_mask)
```

- [ ] **Step 2: Run test to see the current state**

Run: `uv run pytest tests/test_common.py::TestSFTIntegrationFence -v`
Expected: FAIL — `sft.format_chatml` and/or `sft.build_loss_mask` not found (audit confirmed these are not in the sft.py `_common` import list at `scripts/sft.py:59-83`).

- [ ] **Step 3: Extend the sft.py `_common` import list**

Edit both branches of the `try/except ModuleNotFoundError` import block at `scripts/sft.py:59-83`. Add `format_chatml` and `build_loss_mask` alphabetically into each import list. After the edit the relevant block reads:

```python
try:
    from scripts._common import (  # type: ignore[import-not-found]
        base_argparse_parser,
        build_loss_mask,
        build_titans_config,
        chunked_forward,
        create_model,
        format_chatml,
        init_accelerator_and_logging,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
        tokenize_chat,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in test-only sys.path layouts
    from _common import (  # type: ignore[no-redef]
        base_argparse_parser,
        build_loss_mask,
        build_titans_config,
        chunked_forward,
        create_model,
        format_chatml,
        init_accelerator_and_logging,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
        tokenize_chat,
    )
```

- [ ] **Step 4: Run the fence test again**

Run: `uv run pytest tests/test_common.py::TestSFTIntegrationFence -v`
Expected: PASS.

- [ ] **Step 5: Confirm no duplicate definitions in sft.py**

Run: `uv run pytest tests/test_scripts_no_duplication.py -v`
Expected: PASS — the existing duplication guard must stay green (it already whitelists the helpers `sft.py` re-exports).

- [ ] **Step 6: Commit**

```bash
git add scripts/sft.py tests/test_common.py
git commit -m "$(cat <<'EOF'
test(scripts): add plan 3 SFT integration fence

Adds TestSFTIntegrationFence from plan 3 Task 9 Step 1 and wires
format_chatml / build_loss_mask through sft.py's _common import
list so external callers relying on sft.format_chatml continue to
work after the DRY consolidation.
EOF
)"
```

---

### Task 4: Make DPO / RLVR plumbing tests collectible

**Files:**
- Modify: `tests/conftest.py`
- Verify (no edit needed): `tests/test_dpo_chunked_logprobs.py`, `tests/test_rlvr_memory_plumbing.py`

- [ ] **Step 1: Reproduce the failure**

Run: `uv run pytest tests/test_dpo_chunked_logprobs.py tests/test_rlvr_memory_plumbing.py --collect-only -q`
Expected: collection error — `ModuleNotFoundError: No module named 'scripts._common'` (or `No module named '_common'`). This is the audit's reported failure.

- [ ] **Step 2: Extend `tests/conftest.py` with a `scripts/` sys.path prelude**

Edit `tests/conftest.py`. Insert a block at the **top of the file**, before any other imports:

```python
"""Shared test fixtures for Titans PyTorch tests."""

import sys
from pathlib import Path

# scripts/ is not a package. Some tests (DPO, RLVR memory plumbing)
# load scripts/dpo.py and scripts/rlvr.py via importlib.util; those
# modules in turn try both `from scripts._common import ...` and the
# sibling fallback `from _common import ...`. Putting scripts/ on
# sys.path here makes the sibling fallback succeed under the
# spec_from_file_location loader without otherwise changing test
# collection behaviour.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pytest
import torch

from titans.config import TitansConfig
```

Keep the remaining body of the file (fixtures `device`, `default_config`, `linear_memory_config`, `batch_size`, `seq_len`) unchanged.

- [ ] **Step 3: Re-run collection to confirm the fix**

Run: `uv run pytest tests/test_dpo_chunked_logprobs.py tests/test_rlvr_memory_plumbing.py --collect-only -q`
Expected: collection succeeds — at least one test id listed per file, no `ModuleNotFoundError`.

- [ ] **Step 4: Run the actual test suites**

Run: `uv run pytest tests/test_dpo_chunked_logprobs.py tests/test_rlvr_memory_plumbing.py -v`
Expected: PASS (both suites exercise plan-2 functionality that the audit confirmed is on main).

- [ ] **Step 5: Run the broader test suite to ensure no collection-time regressions**

Run: `uv run pytest -q --co 2>&1 | tail -20`
Expected: no new collection errors; overall test count unchanged or higher than before Step 2.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py
git commit -m "$(cat <<'EOF'
test(conftest): add scripts/ to sys.path for plumbing tests

tests/test_dpo_chunked_logprobs.py and test_rlvr_memory_plumbing.py
load scripts/dpo.py and scripts/rlvr.py via importlib.util. The
scripts rely on the `from _common import ...` fallback when they
are not loaded as `scripts._common`, which requires scripts/ on
sys.path. Without this, the two test files fail at collection with
ModuleNotFoundError.
EOF
)"
```

---

### Task 5: Run the plan-3 ruff F401 pass

**Files:**
- Modify (as needed): any script under `scripts/` flagged by ruff

- [ ] **Step 1: Inspect current F401 violations**

Run: `uv run ruff check --select F401 scripts/`
Record the output. Each line is `path:line:col: F401 'symbol' imported but unused`.

- [ ] **Step 2: Evaluate each violation**

For each reported line:
- If the symbol is genuinely unused in the module, it is a candidate for removal.
- If the symbol is re-exported for API parity (grep the module for `# noqa: F401` near an existing import — several migrations already annotate these), keep the import and add `# noqa: F401 — re-exported for API parity` on the line. Specifically, `build_titans_config` in `scripts/pretrain.py:58` is already annotated this way.

- [ ] **Step 3: Apply fixes**

Run: `uv run ruff check --select F401 --fix scripts/`
This auto-removes provably unused imports (ones without `noqa`). Review the diff:
```bash
git diff scripts/
```
Revert any removal that is actually a re-export. Re-add the import with an explicit `# noqa: F401 — re-exported for API parity` annotation.

- [ ] **Step 4: Verify clean state**

Run: `uv run ruff check --select F401 scripts/`
Expected: zero violations, or only lines suppressed with `# noqa: F401`.

- [ ] **Step 5: Ensure duplication guard and migration smoke tests still pass**

Run: `uv run pytest tests/test_scripts_no_duplication.py tests/test_common.py -q`
Expected: all tests PASS. A removed import that was silently used via re-export would trip `TestSFTIntegrationFence` (from Task 3) or the per-script migration smoke tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/
git commit -m "$(cat <<'EOF'
style(scripts): finish plan 3 Task 15 ruff F401 pass

Removes genuinely unused imports after the _common migration and
annotates intentional re-exports with `# noqa: F401 — re-exported
for API parity` so ruff stays clean.
EOF
)"
```

If Step 3 produced zero changes (the audit's "UNCLEAR" state was already clean), skip Step 6 and instead note the verification in the task checklist as "no changes needed — already clean".

---

### Task 6: Fix stale `TRIGGERED` reference in `memory_checkpointer.py` docstring

**Files:**
- Modify: `src/titans/memory_checkpointer.py:4-15` (module docstring)

- [ ] **Step 1: Confirm the stale reference**

Run: `grep -n "TRIGGERED" src/titans/memory_checkpointer.py`
Expected: exactly one hit on line 7 inside the module docstring (`MONITORING → TRIGGERED → CAPTURING_AFTER → COOLDOWN`). The `CheckpointerState` enum at `src/titans/memory_checkpointer.py:50-62` has only three states (MONITORING, CAPTURING_AFTER, COOLDOWN); the `TRIGGERED` arrow is a leftover from the plan-1 cleanup.

- [ ] **Step 2: Write the regression test**

Open `tests/test_checkpoint_signals.py`. Append:

```python
def test_memory_checkpointer_docstring_lists_three_states() -> None:
    """The module docstring must not reference the removed TRIGGERED state."""
    import pathlib
    src = pathlib.Path("src/titans/memory_checkpointer.py").read_text()
    # Isolate the module docstring (first triple-quoted block).
    head = src.split('"""', 2)[1]
    assert "TRIGGERED" not in head, (
        "Plan 1 Task 12 removed CheckpointerState.TRIGGERED but the "
        "module docstring still references it. The current enum has "
        "three states: MONITORING, CAPTURING_AFTER, COOLDOWN."
    )
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_checkpoint_signals.py::test_memory_checkpointer_docstring_lists_three_states -v`
Expected: FAIL — `TRIGGERED` still in docstring.

- [ ] **Step 4: Edit the docstring**

In `src/titans/memory_checkpointer.py`, change the second paragraph of the module docstring from:

```
Paper alignment: N/A — novel engineering. The four-state machine
(MONITORING → TRIGGERED → CAPTURING_AFTER → COOLDOWN) and ring buffer are
project-specific plumbing, not derived from any Titans / TNT / AttnRes paper.
```

to:

```
Paper alignment: N/A — novel engineering. The three-state machine
(MONITORING → CAPTURING_AFTER → COOLDOWN) and ring buffer are
project-specific plumbing, not derived from any Titans / TNT / AttnRes paper.
```

Leave the rest of the docstring (including the `MONITORING → CAPTURING_AFTER → COOLDOWN` arrow on line 13 that is already correct) untouched.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_checkpoint_signals.py::test_memory_checkpointer_docstring_lists_three_states -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/titans/memory_checkpointer.py tests/test_checkpoint_signals.py
git commit -m "$(cat <<'EOF'
docs(memory-checkpointer): drop stale TRIGGERED from module docstring

Plan 1 Task 12 removed CheckpointerState.TRIGGERED but left one
reference in the module-level state-flow arrow. Updates the
docstring to match the enum and adds a grep-based guard test so
the two cannot drift again.
EOF
)"
```

---

## Final Verification

- [ ] **Run the full test suite**

```bash
uv run pytest -q
```
Expected: all tests PASS. Cross-check the test count against the pre-plan baseline to confirm no tests were hidden or lost.

- [ ] **Confirm ruff is clean across scripts/**

```bash
uv run ruff check --select F401 scripts/
```
Expected: zero violations.

- [ ] **Confirm no stale `TRIGGERED` references remain**

```bash
grep -rn "TRIGGERED" src/ tests/
```
Expected: zero hits (the enum member is gone and docstring is now corrected).

- [ ] **Confirm `mkdir` usage on output dirs is gone from pretrain.py**

```bash
grep -n "mkdir" scripts/pretrain.py
```
Expected: zero hits (or only unrelated hits outside the output-dir/checkpoint path).

- [ ] **Confirm both inference and pretrain import `setup_checkpoint_dir`**

```bash
grep -n "setup_checkpoint_dir" scripts/pretrain.py scripts/inference.py
```
Expected: at least two hits per file — one in each try/except import branch plus a call site.
