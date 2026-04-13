"""HuggingFace PreTrainedModel wrapper for Titans MAC."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from titans.hf.configuration import TitansMACConfig
from titans.models import TitansMAC


class TitansMACForCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper around TitansMAC.

    Uses composition: delegates to ``self.model`` (a ``TitansMAC`` instance)
    for all forward computation. Provides ``from_pretrained()``,
    ``save_pretrained()``, and a custom ``generate()`` for chunked
    inference with memory state management.
    """

    config_class = TitansMACConfig
    base_model_prefix = "model"
    _no_split_modules = ["MACBlock"]
    supports_gradient_checkpointing = False
    # Tell transformers that model.head.weight is tied to model.embed.weight so
    # save_pretrained correctly deduplicates the shared tensor.
    _tied_weights_keys = {"model.head.weight": "model.embed.weight"}

    def __init__(self, config: TitansMACConfig) -> None:
        super().__init__(config)
        titans_config = config.to_titans_config()
        self.model = TitansMAC(titans_config)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embedding layer."""
        return self.model.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input embedding layer."""
        self.model.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        """Return the output projection (language model head)."""
        return self.model.head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Set the output projection (language model head)."""
        self.model.head = value

    def tie_weights(self, **kwargs) -> None:
        """Tie input embedding and output head weights (called by post_init and from_pretrained)."""
        self.model.head.weight = self.model.embed.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        memory_states: list | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Single-chunk forward pass.

        Args:
            input_ids: Token IDs, shape ``(B, seq_len)`` where
                ``seq_len <= config.chunk_size``.
            labels: Target token IDs for loss computation (HF convention:
                model shifts internally).
            memory_states: Per-block memory states from a previous chunk.
            attention_mask: Accepted for HF API compatibility but unused.

        Returns:
            CausalLMOutputWithPast with logits, optional loss, and
            memory states as ``past_key_values``.
        """
        logits, new_states = self.model(input_ids, states=memory_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_states,
        )
