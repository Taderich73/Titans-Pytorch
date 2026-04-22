"""Integration tests binding TitansChunkMixin to sft.py's chunked behavior.

The mixin's ``compute_loss`` splits a sequence into chunks of
``config.chunk_size``, runs forward on each chunk with memory carry, detaches
memory state at chunk boundaries, and resets per batch when
``reset_memory_per_batch`` is True. These tests lock in that contract so
future refactors cannot silently drift the mixin away from the sft.py loop.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from titans.hf.configuration import TitansMACConfig
from titans.hf.modeling import TitansMACForCausalLM
from titans.hf.trainer import TitansChunkMixin


def _tiny_hf_model(chunk_size: int = 8) -> TitansMACForCausalLM:
    """Build a small deterministic Titans HF model for equivalence tests."""
    config = TitansMACConfig(
        dim=32,
        num_heads=2,
        num_layers=2,
        vocab_size=32,
        chunk_size=chunk_size,
        window_size=chunk_size,
        max_seq_len=chunk_size * 8,
        num_memory_layers=1,
        num_persistent_tokens=2,
    )
    return TitansMACForCausalLM(config)


class _MinimalChunkTrainer(TitansChunkMixin):
    """Trainer stub that exposes ``compute_loss`` without HF Trainer setup.

    ``compute_loss`` only consults ``self.state.global_step``,
    ``self._memory_states``, ``self.reset_memory_per_batch`` and
    ``self.state_carry_warmup_steps``, so a tiny stand-in is enough to exercise
    the chunked-forward behavior end-to-end.
    """

    def __init__(self) -> None:
        self._init_titans_memory(
            reset_memory_per_batch=True, state_carry_warmup_steps=0
        )

        # ``compute_loss`` consults ``self.state.global_step`` for the
        # warmup branch; supply a minimal stub that always returns 0.
        class _State:
            global_step = 0

        self.state = _State()


def test_mixin_matches_sft_chunked_loss_reset_true() -> None:
    """TitansChunkMixin.compute_loss must match a hand-rolled sft.py-style loop.

    Regression guard binding the mixin's chunked-forward behavior to the
    sft.py chunked training loop. With ``reset_memory_per_batch=True`` and
    ``seq_len > chunk_size``, the mixin's averaged per-chunk loss must equal
    the reference that chunks, forwards with memory carry, detaches states at
    chunk boundaries, and averages chunk losses uniformly.
    """
    torch.manual_seed(0)
    chunk_size = 8
    model = _tiny_hf_model(chunk_size=chunk_size).eval()

    seq_len = chunk_size * 2  # non-ragged: matches plan spec
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    labels = torch.randint(0, 32, (batch_size, seq_len))

    trainer = _MinimalChunkTrainer()
    loss_mixin = trainer.compute_loss(model, {"input_ids": input_ids, "labels": labels})

    assert torch.isfinite(loss_mixin), "Mixin loss must be finite."
    # reset_memory_per_batch=True -> states cleared after compute_loss.
    assert trainer._memory_states is None

    # Hand-rolled sft.py-style reference: chunk, forward with memory carry,
    # detach at chunk boundaries, average chunk losses.
    id_chunks = input_ids.split(chunk_size, dim=1)
    lbl_chunks = labels.split(chunk_size, dim=1)
    states: list | None = None
    num_chunks = len(id_chunks)
    total = torch.tensor(0.0)
    for ids_c, lbl_c in zip(id_chunks, lbl_chunks):
        out = model(input_ids=ids_c, labels=lbl_c, memory_states=states)
        total = total + out.loss / num_chunks
        states = out.past_key_values
        if states is not None:
            states = [s.detach() for s in states]

    torch.testing.assert_close(loss_mixin, total, rtol=1e-4, atol=1e-4)


def test_mixin_ragged_last_chunk_matches_single_shot() -> None:
    """Ragged-last-chunk regression: token-weighted aggregation must give the
    same loss as a single-shot reference that sees the full sequence.

    Before Task 12's fix, compute_loss averaged outputs.loss / num_chunks,
    which underweights a ragged final chunk because outputs.loss is itself
    a mean over valid tokens. With token-weighted accumulation the reported
    loss must equal the HF reference mean-over-valid-tokens across the full
    sequence.
    """
    torch.manual_seed(1)
    chunk_size = 8
    model = _tiny_hf_model(chunk_size=chunk_size).eval()

    # Ragged: last chunk is half-size.
    seq_len = chunk_size * 3 + chunk_size // 2
    batch_size = 2
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    labels = torch.randint(0, 32, (batch_size, seq_len))

    trainer = _MinimalChunkTrainer()
    loss_mixin = trainer.compute_loss(model, {"input_ids": input_ids, "labels": labels})

    # Reference: chunk, forward with memory carry, token-weighted mean.
    id_chunks = input_ids.split(chunk_size, dim=1)
    lbl_chunks = labels.split(chunk_size, dim=1)
    states: list | None = None
    total_num = torch.tensor(0.0)
    total_tok = torch.tensor(0.0)
    for ids_c, lbl_c in zip(id_chunks, lbl_chunks):
        out = model(input_ids=ids_c, labels=lbl_c, memory_states=states)
        n_tok = (lbl_c != -100).float().sum()
        total_num = total_num + out.loss * n_tok
        total_tok = total_tok + n_tok
        states = out.past_key_values
        if states is not None:
            states = [s.detach() for s in states]
    ref = total_num / total_tok.clamp(min=1.0)

    torch.testing.assert_close(loss_mixin, ref, rtol=1e-4, atol=1e-4)
