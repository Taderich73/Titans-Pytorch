"""HuggingFace Trainer with per-chunk truncated BPTT for Titans models.

Variant-agnostic: works with any Titans model that has ``config.chunk_size``
and returns memory states as ``past_key_values``.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer


class TitansChunkMixin:
    """Mixin that adds per-chunk truncated BPTT to any Trainer subclass.

    Manages memory state lifecycle: stores state between chunks within a
    batch, optionally carries state across batches, and supports warmup
    reset where memory is reset for the first N steps before carrying.

    Attrs:
        reset_memory_per_batch: If True, reset memory states after each batch.
        state_carry_warmup_steps: Reset memory for first N steps even when
            reset_memory_per_batch is False.
        _memory_states: Current memory states (list or None).
    """

    reset_memory_per_batch: bool
    state_carry_warmup_steps: int
    _memory_states: list | None
    # Provided by the concrete ``Trainer`` base class at MRO-resolution time;
    # the mixin reads ``state.global_step`` to gate the warmup reset window.
    state: Any

    def _init_titans_memory(
        self,
        reset_memory_per_batch: bool = True,
        state_carry_warmup_steps: int = 500,
    ) -> None:
        """Initialize memory management attributes.

        Args:
            reset_memory_per_batch: If True, reset memory states to None
                after each batch.
            state_carry_warmup_steps: When reset_memory_per_batch is False,
                still reset memory for the first N steps.
        """
        self.reset_memory_per_batch = reset_memory_per_batch
        self.state_carry_warmup_steps = state_carry_warmup_steps
        self._memory_states = None

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Per-chunk forward with truncated BPTT.

        Splits input_ids and labels into chunk_size pieces, runs forward
        on each chunk with memory state carry, and averages the loss.
        Detaches memory state at chunk boundaries.

        Args:
            model: The Titans model with ``config.chunk_size``.
            inputs: Dict with ``input_ids`` and ``labels`` tensors.
            return_outputs: If True, return ``(loss, last_outputs)`` tuple.
            **kwargs: Additional keyword arguments (ignored for compatibility).

        Returns:
            Averaged loss tensor, or ``(loss, last_outputs)`` if
            ``return_outputs=True``.
        """
        chunk_size = model.config.chunk_size
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        chunks = input_ids.split(chunk_size, dim=1)
        label_chunks = labels.split(chunk_size, dim=1)

        total_loss_num = torch.tensor(0.0, device=input_ids.device)
        total_loss_tokens = torch.tensor(0.0, device=input_ids.device)
        states = self._memory_states
        last_outputs = None

        for chunk_ids, chunk_labels in zip(chunks, label_chunks):
            outputs = model(
                input_ids=chunk_ids,
                labels=chunk_labels,
                memory_states=states,
            )
            # Token-weighted accumulation: HF causal-LM losses are
            # mean-over-valid-tokens (ignoring -100). Multiplying by the
            # valid-token count restores the correct per-chunk mass so a
            # ragged last chunk is not overweighted by 1/num_chunks.
            n_tok = (chunk_labels != -100).float().sum()
            total_loss_num = total_loss_num + outputs.loss * n_tok
            total_loss_tokens = total_loss_tokens + n_tok
            states = outputs.past_key_values
            last_outputs = outputs

            # Truncated BPTT: detach at chunk boundary
            if states is not None:
                states = [s.detach() if s is not None else None for s in states]

        total_loss = total_loss_num / total_loss_tokens.clamp(min=1.0)

        # Memory state lifecycle
        self._memory_states = states
        reset_this = (
            self.reset_memory_per_batch
            or self.state.global_step < self.state_carry_warmup_steps
        )
        if reset_this:
            self._memory_states = None

        if return_outputs:
            return total_loss, last_outputs
        return total_loss


class TitansTrainer(TitansChunkMixin, Trainer):
    """HF Trainer with per-chunk truncated BPTT for Titans memory models.

    Args:
        reset_memory_per_batch: If True (default), reset memory states to
            None after each batch.
        state_carry_warmup_steps: When reset_memory_per_batch is False,
            still reset memory for the first N steps.
    """

    def __init__(
        self,
        *args: Any,
        reset_memory_per_batch: bool = True,
        state_carry_warmup_steps: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_titans_memory(reset_memory_per_batch, state_carry_warmup_steps)
