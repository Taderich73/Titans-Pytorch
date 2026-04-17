# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for the ``scripts._common`` helpers that don't have dedicated files."""

from __future__ import annotations

import logging
import pathlib
import sys

import torch

# Allow `from scripts._common import ...` under pytest; the repo root is
# not on sys.path by default.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from titans import TitansConfig  # noqa: E402

from scripts._common import (  # noqa: E402
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
    with caplog.at_level(logging.WARNING, logger="scripts._common"):
        out = maybe_compile(
            model, enabled=True, device_type="cpu", use_attn_res=True
        )
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
            dim=32, num_heads=2, num_layers=2, vocab_size=128,
            chunk_size=8, window_size=8,
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


from scripts._common import init_accelerator_and_logging  # noqa: E402


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
        import scripts._common as common
        monkeypatch.setattr(common, "_HAS_ACCELERATE", False, raising=False)
        bundle = common.init_accelerator_and_logging(_CfgForAcc())
        # Stub accelerator should still expose is_main_process + device attrs.
        assert bundle.is_main_process is True


from scripts._common import setup_checkpoint_dir  # noqa: E402


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
            [sys.executable, "scripts/dpo.py", "--help"],
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
            [sys.executable, "scripts/rlvr.py", "--help"],
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
            [sys.executable, "scripts/pretrain.py", "--help"],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0


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
            [sys.executable, "scripts/inference.py", "--help"],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0
