# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for the ``titans.scripts`` helpers that don't have dedicated files."""

from __future__ import annotations

import logging
import pathlib

import pytest
import torch

from tests._subprocess_helpers import subprocess_python
from titans import TitansConfig
from titans.scripts import (
    CHATML_IM_END,
    CHATML_IM_START,
    MODEL_CLASSES,
    base_argparse_parser,
    build_loss_mask,
    build_titans_config,
    create_model,
    format_chatml,
    loss_mask_to_zero_one,
    make_dataloader,
    make_optimizer,
    maybe_compile,
    tokenize_chat,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestFormatChatML:
    """Tests for ``format_chatml``."""

    def test_single_user_turn(self) -> None:
        out = format_chatml([{"role": "user", "content": "hi"}])
        assert out == f"{CHATML_IM_START}user\nhi{CHATML_IM_END}\n"

    def test_three_turn_dialog(self) -> None:
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = format_chatml(msgs)
        expected = (
            f"{CHATML_IM_START}system\nbe nice{CHATML_IM_END}\n"
            f"{CHATML_IM_START}user\nhi{CHATML_IM_END}\n"
            f"{CHATML_IM_START}assistant\nhello{CHATML_IM_END}\n"
        )
        assert out == expected

    def test_missing_role_defaults_to_user(self) -> None:
        out = format_chatml([{"content": "hello"}])
        assert f"{CHATML_IM_START}user\nhello{CHATML_IM_END}" in out

    def test_empty_messages(self) -> None:
        assert format_chatml([]) == ""

    def test_constants_match_chatml_spec(self) -> None:
        assert CHATML_IM_START == "<|im_start|>"
        assert CHATML_IM_END == "<|im_end|>"


def test_maybe_compile_noop_on_cpu() -> None:
    """CPU device type always skips torch.compile."""
    model = torch.nn.Linear(4, 4)
    out = maybe_compile(model, enabled=True, device_type="cpu", use_attn_res=False)
    assert out is model


def test_maybe_compile_disabled_flag() -> None:
    """``enabled=False`` is a no-op regardless of device/config."""
    model = torch.nn.Linear(4, 4)
    out = maybe_compile(model, enabled=False, device_type="cuda", use_attn_res=False)
    assert out is model


def test_maybe_compile_attn_res_no_longer_auto_disables(caplog) -> None:
    """``use_attn_res=True`` must NOT emit the legacy auto-disable warning.

    The AttnRes sub-layer schedule is now compile-compatible
    (see ``src/titans/models.py::_build_attnres_schedule``), so the old
    warning + skip behavior has been removed. We still return ``model``
    unchanged on CPU because ``torch.compile`` only activates on CUDA.
    """
    model = torch.nn.Linear(4, 4)
    with caplog.at_level(logging.WARNING, logger="titans.scripts._common"):
        out = maybe_compile(model, enabled=True, device_type="cpu", use_attn_res=True)
    assert out is model  # CPU path is a no-op
    assert not any("attn_res" in r.message.lower() for r in caplog.records)


def test_make_optimizer_cpu_defaults_to_foreach() -> None:
    """On CPU, fused=True must NOT be requested."""
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    opt = make_optimizer(params, lr=1e-3, weight_decay=0.1, device_type="cpu")
    assert isinstance(opt, torch.optim.AdamW)
    # foreach default is True on CPU; fused must NOT be set.
    assert opt.defaults.get("fused") in (False, None)


def test_make_optimizer_cuda_uses_fused() -> None:
    """When device_type=='cuda', fused=True is requested."""
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    # The _force_fused_flag test hook bypasses torch.cuda.is_available()
    # so we can verify the fused kwarg is passed through on CPU runners.
    opt = make_optimizer(
        params,
        lr=1e-3,
        weight_decay=0.1,
        device_type="cuda",
        _force_fused_flag=True,
    )
    assert opt.defaults.get("fused") is True


class TestMoveOptimizerStateToParams:
    """Regression tests for ``move_optimizer_state_to_params``.

    The helper fixes the resume-path bug where ``optimizer.load_state_dict``
    leaves state tensors on CPU while live params have been moved to CUDA
    by ``accelerator.prepare``. Fused Adam then refuses to mix devices and
    raises ``RuntimeError``. Local CI runs on CPU-only; the cross-device
    migration test is gated on CUDA availability.
    """

    from titans.scripts import move_optimizer_state_to_params

    def _optimizer_with_populated_state(self) -> torch.optim.AdamW:
        """Return an AdamW whose state slots carry exp_avg / exp_avg_sq."""
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Run one forward + backward + step so state slots populate.
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        return opt

    def test_helper_is_noop_when_state_already_matches_param(self) -> None:
        """Same-device, same-dtype call must report zero migrations."""
        from titans.scripts import move_optimizer_state_to_params

        opt = self._optimizer_with_populated_state()

        before = [
            (k, v.device, v.dtype)
            for group in opt.param_groups
            for p in group["params"]
            for k, v in opt.state[p].items()
            if torch.is_tensor(v)
        ]
        assert before, "precondition: optimizer state must carry tensors"

        migrated, seen = move_optimizer_state_to_params(opt)

        after = [
            (k, v.device, v.dtype)
            for group in opt.param_groups
            for p in group["params"]
            for k, v in opt.state[p].items()
            if torch.is_tensor(v)
        ]
        assert before == after
        assert migrated == 0
        assert seen == len(before)

    def test_helper_migrates_state_onto_param_device(self) -> None:
        """Simulate the resume path: state tensors loaded on one device,
        params on another. The helper must place every state tensor onto
        the corresponding param's device.

        On runners where CUDA is available we exercise the true CPU ->
        CUDA path that triggers the production bug. Otherwise this is a
        synthetic same-device baseline that still verifies the helper
        iterates the structure correctly.
        """
        from titans.scripts import move_optimizer_state_to_params

        opt = self._optimizer_with_populated_state()

        target_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if target_device.type == "cuda":
            for group in opt.param_groups:
                for p in group["params"]:
                    p.data = p.data.to(target_device)
                    if p.grad is not None:
                        p.grad = p.grad.to(target_device)
            # Force state tensors to stay on CPU to simulate the post-load
            # state from ``torch.load(..., map_location="cpu")``.
            for group in opt.param_groups:
                for p in group["params"]:
                    for k, v in list(opt.state[p].items()):
                        if torch.is_tensor(v):
                            opt.state[p][k] = v.cpu()

        migrated, seen = move_optimizer_state_to_params(opt)

        for group in opt.param_groups:
            for p in group["params"]:
                for k, v in opt.state[p].items():
                    if torch.is_tensor(v):
                        assert v.device == p.device, (
                            f"state[{k}] still on {v.device}, param on {p.device}"
                        )
        # On CUDA: all state tensors had to move; on CPU: none did.
        if target_device.type == "cuda":
            assert migrated == seen
        else:
            assert migrated == 0
        assert seen >= 1

    def test_helper_coerces_float_state_dtype_to_param_dtype(self) -> None:
        """Float state tensors saved in a different dtype are coerced.

        Covers the checkpoint scenario where a training run saved
        optimizer state in one dtype (e.g. bf16) and the resume process
        loads it into an optimizer whose params are another dtype
        (e.g. fp32). Fused Adam requires an exact dtype match across the
        ``(params, grads, exp_avgs, exp_avg_sqs)`` quartet, so the
        helper must coerce float state to match the param.
        """
        from titans.scripts import move_optimizer_state_to_params

        opt = self._optimizer_with_populated_state()

        # Force every float state tensor into bf16 while params stay fp32.
        for group in opt.param_groups:
            for p in group["params"]:
                assert p.dtype == torch.float32
                for k, v in list(opt.state[p].items()):
                    if torch.is_tensor(v) and v.is_floating_point():
                        opt.state[p][k] = v.to(torch.bfloat16)

        migrated, seen = move_optimizer_state_to_params(opt)

        for group in opt.param_groups:
            for p in group["params"]:
                for k, v in opt.state[p].items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        assert v.dtype == p.dtype, (
                            f"state[{k}] dtype={v.dtype}, param dtype={p.dtype}"
                        )
        assert migrated >= 1
        assert seen >= migrated

    def test_helper_preserves_int_state_dtype(self) -> None:
        """Integer state (e.g. ``step`` counter on some builds) keeps int dtype.

        Newer PyTorch tracks ``step`` as a 0-d int64 tensor on CPU. The
        helper must not coerce int state to the param's float dtype --
        that would break counter semantics and potentially cause LR
        scheduler drift.
        """
        from titans.scripts import move_optimizer_state_to_params

        opt = self._optimizer_with_populated_state()

        # Inject an int state tensor alongside the real float state.
        for group in opt.param_groups:
            for p in group["params"]:
                opt.state[p]["_int_counter"] = torch.tensor(42, dtype=torch.int64)

        move_optimizer_state_to_params(opt)

        for group in opt.param_groups:
            for p in group["params"]:
                v = opt.state[p]["_int_counter"]
                assert v.dtype == torch.int64, (
                    f"int state tensor coerced to {v.dtype}, must stay int64"
                )
                assert int(v.item()) == 42, "int state value must be preserved"

    def test_helper_skips_params_without_state(self) -> None:
        """Params with no registered state must not raise or be touched."""
        from titans.scripts import move_optimizer_state_to_params

        # Fresh optimizer: no step() call => empty state for every param.
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        assert not any(opt.state[p] for g in opt.param_groups for p in g["params"])

        migrated, seen = move_optimizer_state_to_params(opt)
        assert migrated == 0
        assert seen == 0


class TestInitializeMissingOptimizerState:
    """Regression tests for ``initialize_missing_optimizer_state``.

    The helper fixes a fused-Adam crash on resume when a checkpoint's
    optimizer state_dict is missing entries for some live params. It
    eagerly seeds zero-initialized Adam state so the first sync-gradient
    step sees uniform state across the whole param list.
    """

    def test_helper_initializes_missing_state(self) -> None:
        """Empty state slots get zero-filled exp_avg / exp_avg_sq / step."""
        from titans.scripts import initialize_missing_optimizer_state

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        assert not any(opt.state[p] for g in opt.param_groups for p in g["params"])

        initialized, seen = initialize_missing_optimizer_state(opt)

        num_params = sum(1 for g in opt.param_groups for _ in g["params"])
        assert initialized == num_params
        assert seen == num_params

        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                assert "exp_avg" in state
                assert "exp_avg_sq" in state
                assert "step" in state
                assert state["exp_avg"].shape == p.shape
                assert state["exp_avg_sq"].shape == p.shape
                assert state["exp_avg"].dtype == p.dtype
                assert state["exp_avg_sq"].dtype == p.dtype
                assert state["exp_avg"].device == p.device
                assert state["exp_avg_sq"].device == p.device
                assert torch.all(state["exp_avg"] == 0)
                assert torch.all(state["exp_avg_sq"] == 0)

    def test_helper_noop_when_all_params_have_state(self) -> None:
        """Params with full state are left untouched."""
        from titans.scripts import initialize_missing_optimizer_state

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Populate via a backward + step.
        out = model(torch.randn(1, 4)).sum()
        out.backward()
        opt.step()

        pre_state = {
            id(p): {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in opt.state[p].items()
            }
            for g in opt.param_groups
            for p in g["params"]
        }

        initialized, seen = initialize_missing_optimizer_state(opt)
        assert initialized == 0
        num_params = sum(1 for g in opt.param_groups for _ in g["params"])
        assert seen == num_params

        # Existing state unchanged.
        for group in opt.param_groups:
            for p in group["params"]:
                for k, v in pre_state[id(p)].items():
                    live = opt.state[p][k]
                    if torch.is_tensor(v):
                        assert torch.equal(live, v), f"{k} was mutated"
                    else:
                        assert live == v, f"{k} was mutated"

    def test_helper_only_fills_missing_params(self) -> None:
        """Partial-state optimizer: only the empty slots get filled."""
        from titans.scripts import initialize_missing_optimizer_state

        # Two separate param tensors so we can leave one state-less.
        p_with_state = torch.nn.Parameter(torch.randn(3, 3))
        p_no_state = torch.nn.Parameter(torch.randn(3, 3))
        opt = torch.optim.AdamW([p_with_state, p_no_state], lr=1e-3)

        # Populate state only for p_with_state by stepping with grad on it.
        loss = (p_with_state * p_with_state).sum()
        loss.backward()
        opt.step()

        assert "exp_avg" in opt.state[p_with_state]
        assert not opt.state[p_no_state]

        pre_exp_avg = opt.state[p_with_state]["exp_avg"].clone()

        initialized, seen = initialize_missing_optimizer_state(opt)
        assert initialized == 1
        assert seen == 2

        # New state for the missing param, untouched for the other.
        assert torch.all(opt.state[p_no_state]["exp_avg"] == 0)
        assert torch.equal(opt.state[p_with_state]["exp_avg"], pre_exp_avg)

    def test_helper_respects_fused_flag_for_step_placement(self) -> None:
        """With fused=True, ``step`` is a device-0d tensor (matches _init_group)."""
        from titans.scripts import initialize_missing_optimizer_state, make_optimizer

        model = torch.nn.Linear(4, 4)
        # _force_fused_flag skips the CUDA availability guard so we can
        # exercise the fused branch on CI hardware without a GPU.
        opt = make_optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.0,
            device_type="cpu",
            _force_fused_flag=True,
        )

        initialize_missing_optimizer_state(opt)

        for group in opt.param_groups:
            for p in group["params"]:
                step = opt.state[p]["step"]
                assert step.shape == ()
                assert step.device == p.device


class TestRemapOptimizerStateByName:
    """Regression tests for ``remap_optimizer_state_by_name``.

    Name-based remap preserves momentum across named_parameters() order
    drift (module refactors, feature-flag toggles) where positional
    load_state_dict silently lands state on the wrong param.
    """

    @staticmethod
    def _populated_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        out = model(torch.randn(1, 4)).sum()
        out.backward()
        opt.step()
        return opt

    def test_identical_order_preserves_all_state(self) -> None:
        from titans.scripts import remap_optimizer_state_by_name

        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        opt = self._populated_optimizer(model)
        saved_names = [n for n, _ in model.named_parameters()]
        saved_state = opt.state_dict()

        remapped, preserved, dropped = remap_optimizer_state_by_name(
            saved_state, saved_names, saved_names
        )
        num_params = sum(1 for _ in model.parameters())
        assert preserved == num_params
        assert dropped == 0
        assert len(remapped["state"]) == num_params

    def test_permutation_rescues_state(self) -> None:
        """Same names in different order: state lands at the *right* live
        positions after remap."""
        from titans.scripts import remap_optimizer_state_by_name

        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        opt = self._populated_optimizer(model)
        saved_names = [n for n, _ in model.named_parameters()]
        saved_state = opt.state_dict()

        # Same names, reversed order
        live_names = list(reversed(saved_names))
        remapped, preserved, dropped = remap_optimizer_state_by_name(
            saved_state, saved_names, live_names
        )
        assert preserved == len(live_names)
        assert dropped == 0

        # First saved exp_avg must now live at last live position and vice versa
        first_saved_exp_avg = saved_state["state"][0]["exp_avg"]
        last_live_pos = len(live_names) - 1
        assert torch.equal(
            remapped["state"][last_live_pos]["exp_avg"], first_saved_exp_avg
        )

    def test_added_param_has_no_state_in_remap(self) -> None:
        """Live model has a new param not in saved: remap skips it so
        caller's initialize_missing_optimizer_state fills it later."""
        from titans.scripts import remap_optimizer_state_by_name

        model_old = torch.nn.Sequential(torch.nn.Linear(4, 8))
        opt_old = self._populated_optimizer(model_old)
        saved_names = [n for n, _ in model_old.named_parameters()]
        saved_state = opt_old.state_dict()

        live_names = saved_names + ["blocks.new.weight", "blocks.new.bias"]
        remapped, preserved, dropped = remap_optimizer_state_by_name(
            saved_state, saved_names, live_names
        )
        assert preserved == len(saved_names)
        assert dropped == 0
        # New live positions aren't in the remapped state
        for i, name in enumerate(live_names):
            if name in saved_names:
                assert i in remapped["state"]
            else:
                assert i not in remapped["state"]

    def test_removed_param_state_is_dropped(self) -> None:
        """Saved has a param no longer in live model: state is dropped."""
        from titans.scripts import remap_optimizer_state_by_name

        model_big = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        opt_big = self._populated_optimizer(model_big)
        saved_names = [n for n, _ in model_big.named_parameters()]
        saved_state = opt_big.state_dict()

        live_names = saved_names[:2]  # drop the second linear's params
        remapped, preserved, dropped = remap_optimizer_state_by_name(
            saved_state, saved_names, live_names
        )
        assert preserved == 2
        assert dropped == len(saved_names) - 2

    def test_empty_state_dict_is_passthrough(self) -> None:
        from titans.scripts import remap_optimizer_state_by_name

        out, p, d = remap_optimizer_state_by_name({}, [], [])
        assert out == {}
        assert p == 0 and d == 0


class TestIsOptimizerStateCompatible:
    """Regression tests for ``is_optimizer_state_compatible``.

    Verifies that param-order drift between a saved optimizer state_dict
    and a live optimizer is detected by shape-checking each saved
    state entry against its would-be target live param. This is what
    PRs #11-14 were missing — every prior diag axis was uniform under
    drift (device, dtype, layout, contig, non-overlapping-dense), so
    only a shape check catches it.
    """

    def test_compatible_when_shapes_match(self) -> None:
        """Identical model layout: state loads cleanly."""
        from titans.scripts import is_optimizer_state_compatible

        model_a = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
        out = model_a(torch.randn(1, 4)).sum()
        out.backward()
        opt_a.step()
        saved = opt_a.state_dict()

        model_b = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

        compatible, mismatches, checked = is_optimizer_state_compatible(opt_b, saved)
        assert compatible
        assert mismatches == 0
        assert checked > 0

    def test_drift_detected_when_layer_dims_change(self) -> None:
        """Hidden-dim mismatch between save and load: drift detected."""
        from titans.scripts import is_optimizer_state_compatible

        model_a = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
        out = model_a(torch.randn(1, 4)).sum()
        out.backward()
        opt_a.step()
        saved = opt_a.state_dict()

        # Live model with different hidden dim — param shapes will
        # mismatch at positional slots.
        model_b = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Linear(16, 4))
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

        compatible, mismatches, checked = is_optimizer_state_compatible(opt_b, saved)
        assert not compatible
        assert mismatches > 0
        assert checked > 0

    def test_drift_detected_when_param_count_changes(self) -> None:
        """Extra layer on load side: compatibility check refuses."""
        from titans.scripts import is_optimizer_state_compatible

        model_a = torch.nn.Sequential(torch.nn.Linear(4, 8))
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
        out = model_a(torch.randn(1, 4)).sum()
        out.backward()
        opt_a.step()
        saved = opt_a.state_dict()

        model_b = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

        compatible, _, _ = is_optimizer_state_compatible(opt_b, saved)
        assert not compatible

    def test_empty_saved_state_is_compatible(self) -> None:
        """Fresh optimizer (no state yet) is trivially compatible."""
        from titans.scripts import is_optimizer_state_compatible

        model = torch.nn.Linear(4, 4)
        opt_saved = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt_live = torch.optim.AdamW(torch.nn.Linear(4, 4).parameters(), lr=1e-3)

        compatible, mismatches, checked = is_optimizer_state_compatible(
            opt_live, opt_saved.state_dict()
        )
        assert compatible
        assert mismatches == 0
        assert checked == 0  # nothing to check since saved state is empty


def test_make_dataloader_multiworker_flags_cuda() -> None:
    """Multi-worker DataLoader enables pin_memory and persistent_workers on CUDA."""
    from torch.utils.data import TensorDataset

    ds = TensorDataset(torch.arange(128))
    dl = make_dataloader(
        ds,
        batch_size=8,
        num_workers=4,
        device_type="cuda",
        shuffle=True,
    )
    assert dl.num_workers == 4
    assert dl.pin_memory is True
    assert dl.persistent_workers is True


def test_make_dataloader_streaming_forces_zero_workers() -> None:
    """Streaming datasets force num_workers=0 (unsafe with iterable)."""

    class FakeStream:  # no __len__, no __getitem__
        def __iter__(self):
            yield torch.zeros(4)

    ds = FakeStream()
    dl = make_dataloader(
        ds,
        batch_size=1,
        num_workers=4,
        device_type="cuda",
        shuffle=False,
        streaming=True,
    )
    assert dl.num_workers == 0
    assert dl.pin_memory is True  # pin_memory still OK without workers


def test_make_dataloader_zero_workers_no_persistent() -> None:
    """num_workers=0 must not set persistent_workers/prefetch_factor."""
    from torch.utils.data import TensorDataset

    ds = TensorDataset(torch.arange(32))
    dl = make_dataloader(
        ds,
        batch_size=4,
        num_workers=0,
        device_type="cpu",
        shuffle=False,
    )
    assert dl.num_workers == 0
    assert dl.pin_memory is False
    # persistent_workers default is False when num_workers=0.
    assert dl.persistent_workers is False


class TestBuildLossMask:
    """Parity tests vs sft.py/dpo.py build_loss_mask."""

    def test_train_on_all_short_circuits(self) -> None:
        assert build_loss_mask(5, [(0, 5)], train_on_all=True) == [1, 1, 1, 1, 1]

    def test_empty_spans_yields_zeros(self) -> None:
        assert build_loss_mask(4, [], train_on_all=False) == [0, 0, 0, 0]

    def test_single_span(self) -> None:
        assert build_loss_mask(6, [(2, 5)], train_on_all=False) == [0, 0, 1, 1, 1, 0]

    def test_span_clamped_to_seq_len(self) -> None:
        assert build_loss_mask(4, [(2, 10)], train_on_all=False) == [0, 0, 1, 1]

    def test_eos_inclusion(self) -> None:
        mask = build_loss_mask(
            6,
            [(1, 3)],
            include_eos=True,
            eos_positions=[3],
            train_on_all=False,
        )
        assert mask == [0, 1, 1, 1, 0, 0]

    def test_eos_outside_range_ignored(self) -> None:
        mask = build_loss_mask(
            4,
            [(0, 2)],
            include_eos=True,
            eos_positions=[99, -1],
            train_on_all=False,
        )
        assert mask == [1, 1, 0, 0]


class TestLossMaskToZeroOne:
    """Adapter that turns label arrays with -100 into 0/1 masks (lora variant)."""

    def test_minus_100_becomes_zero(self) -> None:
        assert loss_mask_to_zero_one([-100, 5, -100, 7]) == [0, 1, 0, 1]

    def test_all_minus_100(self) -> None:
        assert loss_mask_to_zero_one([-100, -100]) == [0, 0]

    def test_all_valid(self) -> None:
        assert loss_mask_to_zero_one([1, 2, 3]) == [1, 1, 1]

    def test_zero_is_still_a_real_token(self) -> None:
        assert loss_mask_to_zero_one([0, 0, -100]) == [1, 1, 0]


class _FakeTokenizer:
    """Minimal whitespace-ish tokenizer (no chat template)."""

    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 256 for c in text]

    def apply_chat_template(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


class TestTokenizeChat:
    """tokenize_chat falls back to ChatML markup when chat_template is None."""

    def test_shift_and_mask_length_consistency(self) -> None:
        tok = _FakeTokenizer()
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        out = tokenize_chat(msgs, tok, max_len=64)
        assert set(out.keys()) == {"input_ids", "labels", "loss_mask"}
        assert len(out["input_ids"]) == len(out["labels"]) == len(out["loss_mask"])

    def test_train_on_all_all_ones(self) -> None:
        tok = _FakeTokenizer()
        msgs = [{"role": "user", "content": "a"}]
        out = tokenize_chat(msgs, tok, max_len=32, train_on_all=True)
        assert all(m == 1 for m in out["loss_mask"])

    def test_assistant_tokens_supervised_user_tokens_not(self) -> None:
        tok = _FakeTokenizer()
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        out = tokenize_chat(msgs, tok, max_len=128)
        # At least one supervised position must exist (the assistant content).
        assert sum(out["loss_mask"]) > 0
        # And at least one non-supervised position (the user content).
        assert any(m == 0 for m in out["loss_mask"])

    def test_max_len_truncates(self) -> None:
        tok = _FakeTokenizer()
        msgs = [{"role": "assistant", "content": "x" * 100}]
        out = tokenize_chat(msgs, tok, max_len=10)
        # After max_len=10 truncation we have 9 shifted tokens.
        assert len(out["input_ids"]) == 9


class TestTokenizeChatParityWithSFT:
    """Parity check: consolidated helper must match sft.py's old output."""

    def test_output_matches_reference_emit(self) -> None:
        tok = _FakeTokenizer()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        out = tokenize_chat(msgs, tok, max_len=128)
        # Reference emit constructed the same way sft.py did pre-migration.
        expected_text = (
            "<|im_start|>system\nsys<|im_end|>\n"
            "<|im_start|>user\nq<|im_end|>\n"
            "<|im_start|>assistant\na<|im_end|>\n"
        )
        expected_ids = tok.encode(expected_text, add_special_tokens=False)
        assert out["input_ids"] + [out["labels"][-1]] == expected_ids


class TestCreateModel:
    """Tests for the shared Titans variant registry."""

    @pytest.mark.parametrize("variant", ["mac", "mag", "mal", "lmm"])
    def test_every_variant_instantiates(self, variant: str) -> None:
        cfg = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=2,
            vocab_size=128,
            chunk_size=8,
            window_size=8,
        )
        model = create_model(variant, cfg)
        # One forward pass must not explode.
        ids = torch.zeros(1, 8, dtype=torch.long)
        _ = model(ids)

    def test_unknown_variant_raises_with_options(self) -> None:
        cfg = TitansConfig(dim=16, num_heads=1, num_layers=1, vocab_size=8)
        with pytest.raises(ValueError, match="Unknown variant"):
            create_model("xyz", cfg)

    def test_registry_keys_are_lowercase_and_exhaustive(self) -> None:
        assert set(MODEL_CLASSES.keys()) == {"mac", "mag", "mal", "lmm"}


# ---------------------------------------------------------------------------
# build_titans_config tests
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402


@dataclass
class _FakeCfg:
    """Minimal duck-typed cfg exposing every field build_titans_config reads."""

    # Core
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    vocab_size: int = 256
    chunk_size: int = 16
    window_size: int = 16
    rope_proportion: float = 1.0
    num_persistent_tokens: int = 4
    num_memory_layers: int = 2
    memory_objective: str = "l2"
    huber_delta_init: float = 0.0
    dropout: float = 0.0
    use_conv: bool = False
    # TNT
    use_tnt: bool = False
    global_chunk_size: int = 64
    local_chunk_sizes: list[int] = field(default_factory=lambda: [4, 8])
    local_shard_length: int = 64
    use_qk_projection: bool = True
    tnt_stage: int = 1
    finetune_local_chunk_sizes: list[int] | None = None
    # Attn res
    use_attn_res: bool = False
    num_attnres_blocks: int = 2
    attnres_warmup_steps: int = 0
    attnres_modulate_global_memory: bool = True
    attnres_modulate_local_memory: bool = False
    # Adaptive window
    adaptive_window: bool = False
    adaptive_window_min: int = 8
    adaptive_window_max: int | None = None
    adaptive_window_temperature: float = 10.0
    adaptive_window_lambda: float = 0.01
    # MCA
    use_mca: bool = False
    mca_insertion_layers: list[int] | None = None
    mca_num_heads: int = 4
    mca_gate_type: str = "scalar"
    mca_gate_bias_init: float = -3.0
    # Plan 5 additions
    num_memory_inner_steps: int = 1
    mac_per_position_memory_query: bool = False


class TestBuildTitansConfig:
    """build_titans_config must produce a config exactly equivalent to the old
    sft.py/dpo.py/rlvr.py inline builders, including all optional feature
    field-sets (TNT / AttnRes / MCA / adaptive window / Plan 5 fields)."""

    def test_core_fields_copied(self) -> None:
        c = build_titans_config(_FakeCfg(dim=128, num_heads=8))
        assert c.dim == 128
        assert c.num_heads == 8

    def test_tnt_fields_only_populated_when_enabled(self) -> None:
        base = build_titans_config(_FakeCfg(use_tnt=False))
        assert base.use_tnt is False
        with_tnt = build_titans_config(
            _FakeCfg(use_tnt=True, local_chunk_sizes=[8, 16]),
        )
        assert with_tnt.use_tnt is True
        assert with_tnt.local_chunk_sizes == [8, 16]

    def test_attn_res_fields(self) -> None:
        c = build_titans_config(_FakeCfg(use_attn_res=True, num_attnres_blocks=4))
        assert c.use_attn_res is True
        assert c.num_attnres_blocks == 4

    def test_mca_fields(self) -> None:
        c = build_titans_config(
            _FakeCfg(use_mca=True, num_layers=4, mca_insertion_layers=[1, 3]),
        )
        assert c.use_mca is True
        assert c.mca_insertion_layers == [1, 3]

    def test_adaptive_window_fields(self) -> None:
        c = build_titans_config(
            _FakeCfg(adaptive_window=True, adaptive_window_min=32),
        )
        assert c.adaptive_window is True
        assert c.adaptive_window_min == 32

    def test_plan5_fields_forwarded_when_present(self) -> None:
        c = build_titans_config(
            _FakeCfg(
                num_memory_inner_steps=3,
                mac_per_position_memory_query=True,
            ),
        )
        # If TitansConfig exposes these fields (Plan 5 merged), they must match.
        if hasattr(c, "num_memory_inner_steps"):
            assert c.num_memory_inner_steps == 3
        if hasattr(c, "mac_per_position_memory_query"):
            assert c.mac_per_position_memory_query is True


class TestBaseArgparseParser:
    """Shared parser must expose every flag all four training scripts share."""

    def test_returns_argparse_parser(self) -> None:
        import argparse

        p = base_argparse_parser(description="x")
        assert isinstance(p, argparse.ArgumentParser)

    def test_defaults_match_current_scripts(self) -> None:
        p = base_argparse_parser(description="x")
        ns = p.parse_args([])
        # Seed/training defaults
        assert ns.seed == 42
        assert ns.batch_size == 4
        assert ns.gradient_accumulation_steps == 8
        assert ns.weight_decay == 0.1
        assert ns.grad_clip == 1.0
        assert ns.warmup_ratio == 0.03
        assert ns.mixed_precision == "no"
        # Model defaults
        assert ns.dim == 512
        assert ns.num_heads == 8
        assert ns.num_layers == 12
        assert ns.vocab_size == 32000
        assert ns.chunk_size == 512
        assert ns.window_size == 512
        assert ns.rope_proportion == 1.0
        assert ns.num_persistent_tokens == 16
        assert ns.num_memory_layers == 2
        assert ns.memory_objective == "l2"
        # Feature toggles default off
        assert ns.use_tnt is False
        assert ns.use_attn_res is False
        assert ns.use_mca is False
        assert ns.adaptive_window is False
        # Logging / checkpointing
        assert ns.log_every == 10
        assert ns.save_format == "pt"

    def test_mixed_precision_choices(self) -> None:
        p = base_argparse_parser(description="x")
        ns = p.parse_args(["--mixed-precision", "bf16"])
        assert ns.mixed_precision == "bf16"
        with pytest.raises(SystemExit):
            p.parse_args(["--mixed-precision", "invalid"])

    def test_variant_choices(self) -> None:
        p = base_argparse_parser(description="x")
        for v in ("mac", "mag", "mal", "lmm"):
            ns = p.parse_args(["--model", v])
            assert ns.model == v
        with pytest.raises(SystemExit):
            p.parse_args(["--model", "bogus"])

    def test_script_can_add_extra_flags(self) -> None:
        p = base_argparse_parser(description="x")
        p.add_argument("--beta", type=float, default=0.1)
        ns = p.parse_args(["--beta", "0.5"])
        assert ns.beta == 0.5
        # And the base flags still work.
        assert ns.dim == 512

    def test_no_dead_flags_reintroduced(self) -> None:
        """Plan 4 removed these; the base parser must not add them back."""
        p = base_argparse_parser(description="x")
        ns = p.parse_args([])
        # Neither historic aliases nor dead flags should appear.
        assert not hasattr(ns, "datasetonly_alias")


from titans.scripts import init_accelerator_and_logging


class _CfgForAcc:
    gradient_accumulation_steps = 4
    mixed_precision = "no"
    wandb = False


class TestInitAcceleratorAndLogging:
    def test_returns_namespace_with_accelerator(self) -> None:
        try:
            from accelerate import Accelerator  # noqa: F401
        except ImportError:
            pytest.skip("accelerate not installed")

        bundle = init_accelerator_and_logging(_CfgForAcc())
        assert hasattr(bundle, "accelerator")
        assert hasattr(bundle, "is_main_process")
        assert bundle.is_main_process is True  # single-proc test env

    def test_without_accelerate_graceful(self, monkeypatch) -> None:
        """Callers that run without accelerate still get a usable object."""
        from titans.scripts import _common

        monkeypatch.setattr(_common, "_HAS_ACCELERATE", False, raising=False)
        bundle = _common.init_accelerator_and_logging(_CfgForAcc())
        # Stub accelerator should still expose is_main_process + device attrs.
        assert bundle.is_main_process is True


from titans.scripts import setup_checkpoint_dir


class TestSetupCheckpointDir:
    def test_creates_missing_directory(self, tmp_path) -> None:
        target = tmp_path / "new_run"
        result = setup_checkpoint_dir(str(target), resume_path=None)
        assert target.exists()
        assert result.output_dir == target
        assert result.resume_path is None
        assert result.resume_step == 0

    def test_existing_directory_accepted(self, tmp_path) -> None:
        target = tmp_path / "existing"
        target.mkdir()
        result = setup_checkpoint_dir(str(target), resume_path=None)
        assert result.output_dir == target

    def test_explicit_resume_path(self, tmp_path) -> None:
        target = tmp_path / "run"
        target.mkdir()
        ckpt = target / "step_123.pt"
        ckpt.write_bytes(b"\x00")
        result = setup_checkpoint_dir(str(target), resume_path=str(ckpt))
        assert result.resume_path == ckpt
        assert result.resume_step == 123

    def test_resume_step_unparseable_falls_back_to_zero(self, tmp_path) -> None:
        target = tmp_path / "run"
        target.mkdir()
        ckpt = target / "arbitrary_name.pt"
        ckpt.write_bytes(b"\x00")
        result = setup_checkpoint_dir(str(target), resume_path=str(ckpt))
        assert result.resume_step == 0

    def test_resume_path_nonexistent_raises(self, tmp_path) -> None:
        target = tmp_path / "run"
        target.mkdir()
        with pytest.raises(FileNotFoundError):
            setup_checkpoint_dir(str(target), resume_path=str(tmp_path / "missing.pt"))


class TestLoraFeatureFlagsHonoured:
    """Regression: lora.py used to silently drop feature flags. After
    migration to build_titans_config, they must reach TitansConfig."""

    def test_use_tnt_reaches_config(self) -> None:
        from scripts.lora import LoRATrainingConfig, build_titans_config

        cfg = LoRATrainingConfig()
        # The dataclass must expose use_tnt (added as part of migration).
        cfg.use_tnt = True
        cfg.global_chunk_size = 1024
        cfg.local_chunk_sizes = [4, 8]
        cfg.local_shard_length = 1024
        cfg.use_qk_projection = True
        cfg.tnt_stage = 1
        cfg.finetune_local_chunk_sizes = None
        tc = build_titans_config(cfg)
        assert tc.use_tnt is True

    def test_use_attn_res_reaches_config(self) -> None:
        from scripts.lora import LoRATrainingConfig, build_titans_config

        cfg = LoRATrainingConfig()
        cfg.use_attn_res = True
        cfg.num_attnres_blocks = 4
        cfg.attnres_warmup_steps = 0
        cfg.attnres_modulate_global_memory = True
        cfg.attnres_modulate_local_memory = False
        tc = build_titans_config(cfg)
        assert tc.use_attn_res is True


class TestDPOMigrationSmoke:
    """Smoke: dpo.py still imports and its parser constructs."""

    def test_imports(self) -> None:
        from scripts import dpo

        assert callable(dpo.create_model)
        assert callable(dpo.build_titans_config)
        assert callable(dpo.tokenize_chat)

    def test_parse_args_help_exits_zero(self) -> None:
        import subprocess

        r = subprocess.run(
            [*subprocess_python(), "scripts/dpo.py", "--help"],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0


class TestRLVRMigrationSmoke:
    """Smoke: rlvr.py still imports and its parser constructs."""

    def test_imports(self) -> None:
        from scripts import rlvr

        assert callable(rlvr.create_model)
        assert callable(rlvr.build_titans_config)

    def test_parse_args_help_exits_zero(self) -> None:
        import subprocess

        r = subprocess.run(
            [*subprocess_python(), "scripts/rlvr.py", "--help"],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0


class TestPretrainMigrationSmoke:
    """Smoke: pretrain.py still imports and exposes shared helpers."""

    def test_imports(self) -> None:
        from scripts import pretrain

        assert callable(pretrain.create_model) or callable(
            getattr(pretrain, "build_model", None)
        )

    def test_parse_args_help_exits_zero(self) -> None:
        import subprocess

        r = subprocess.run(
            [*subprocess_python(), "scripts/pretrain.py", "--help"],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0

    def test_pretrain_imports_setup_checkpoint_dir(self) -> None:
        """pretrain.py must import setup_checkpoint_dir from titans.scripts."""
        src = (_REPO_ROOT / "scripts" / "pretrain.py").read_text()
        assert "setup_checkpoint_dir," in src or "    setup_checkpoint_dir," in src, (
            "pretrain.py did not import setup_checkpoint_dir; "
            "plan 3 Task 13 Step 5 required it."
        )


class TestInferenceMigrationSmoke:
    """Smoke: inference.py still imports and exposes shared helpers."""

    def test_imports(self) -> None:
        from scripts import inference

        assert callable(inference.main) or callable(
            getattr(inference, "load_model", None)
        )

    def test_help_exits_zero(self) -> None:
        import subprocess

        r = subprocess.run(
            [*subprocess_python(), "scripts/inference.py", "--help"],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0

    def test_inference_validates_checkpoint_existence(self) -> None:
        """inference.py must surface a clear FileNotFoundError for a missing --checkpoint.

        Plan 3 Task 14 originally asked for setup_checkpoint_dir here, but
        inference is a read path (no output_dir, no resume_path concept, and
        load_checkpoint supports extensionless lookup). A direct existence
        check with a --checkpoint-specific error message is the right fit.
        """
        src = (_REPO_ROOT / "scripts" / "inference.py").read_text()
        # Must not import the training-script helper.
        assert "setup_checkpoint_dir" not in src, (
            "inference.py re-imported setup_checkpoint_dir; it is a training-script "
            "helper and its FileNotFoundError message mis-labels --checkpoint as --resume."
        )
        # Must surface a --checkpoint-specific error.
        assert '"--checkpoint file not found:' in src, (
            "inference.py must raise a --checkpoint-specific FileNotFoundError "
            "for missing checkpoints (see plan-audit-gap-fixes Task 2 pivot)."
        )


class TestCommonModuleDocumentation:
    def test_module_docstring_lists_public_helpers(self) -> None:
        from titans.scripts import _common

        doc = _common.__doc__ or ""
        for name in (
            "format_chatml",
            "build_loss_mask",
            "tokenize_chat",
            "create_model",
            "build_titans_config",
            "base_argparse_parser",
            "init_accelerator_and_logging",
            "setup_checkpoint_dir",
        ):
            assert name in doc, f"{name} missing from titans.scripts._common docstring"


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
