"""HuggingFace PreTrainedModel wrapper for Titans MAC."""

from __future__ import annotations

from typing import Any, cast

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

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embedding layer."""
        assert isinstance(value, nn.Embedding), (
            "TitansMAC.set_input_embeddings expects nn.Embedding"
        )
        self.model.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        """Return the output projection (language model head)."""
        return self.model.head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Set the output projection (language model head)."""
        self.model.head = value

    def tie_weights(
        self,
        missing_keys: set[str] | None = None,
        recompute_mapping: bool = True,
    ) -> None:
        """Tie input embedding and output head weights.

        Signature matches ``transformers.PreTrainedModel.tie_weights``; we
        ignore both parameters because the single weight-tie here is
        unconditional and does not consult the HF mapping machinery.
        """
        del missing_keys, recompute_mapping
        self.model.head.weight = self.model.embed.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        memory_states: list | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
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
        logits, new_states, _gate_snapshots = self.model(
            input_ids, states=memory_states
        )

        loss: torch.FloatTensor | None = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # F.cross_entropy returns a generic float Tensor; HF's output
            # dataclass types it as FloatTensor.
            loss = cast(
                torch.FloatTensor,
                F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                ),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        memory_states: list | None = None,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """Titans-specific chunked generation with memory state management.

        Processes the prompt in chunk_size chunks (prefill), then decodes
        token-by-token with a buffer that commits memory updates when full.

        Args:
            input_ids: Prompt token IDs, shape ``(B, prompt_len)``.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 to disable).
            top_p: Nucleus sampling threshold (1.0 to disable).
            memory_states: Initial memory states for all blocks.
            do_sample: If False, use greedy decoding.

        Returns:
            Tensor of shape ``(B, prompt_len + num_generated)`` containing
            the prompt followed by generated tokens.
        """
        chunk_size = self.model.config.chunk_size
        generated = input_ids
        states = memory_states

        # Prefill: chunk the prompt since model.forward() requires
        # seq_len <= chunk_size
        prompt_chunks = generated.split(chunk_size, dim=1)
        for chunk in prompt_chunks:
            logits, states, _ = self.model(chunk, states=states)
            if states is not None:
                states = [s.detach() if s is not None else None for s in states]

        committed_states = (
            [s.detach() if s is not None else None for s in states] if states else None
        )
        buffer_start = generated.shape[1]

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove_mask = (
                    cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                )
                sorted_logits[remove_mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # torch.cat widens to Tensor; the concatenation of two LongTensors
            # is still a LongTensor at runtime, so the cast preserves the
            # declared type without changing behaviour.
            generated = cast(
                torch.LongTensor, torch.cat([generated, next_token], dim=1)
            )

            buffer = generated[:, buffer_start:]
            buffer_len = buffer.shape[1]

            if buffer_len >= chunk_size:
                chunk = buffer[:, :chunk_size]
                logits, states, _ = self.model(chunk, states=committed_states)
                if states is not None:
                    states = [s.detach() if s is not None else None for s in states]
                committed_states = (
                    [s.detach() if s is not None else None for s in states]
                    if states is not None
                    else None
                )
                buffer_start += chunk_size

                if buffer_len > chunk_size:
                    remainder = buffer[:, chunk_size:]
                    logits, states, _ = self.model(remainder, states=committed_states)
                    if states is not None:
                        states = [s.detach() if s is not None else None for s in states]
            else:
                logits, states, _ = self.model(buffer, states=committed_states)
                if states is not None:
                    states = [s.detach() if s is not None else None for s in states]

        return generated
