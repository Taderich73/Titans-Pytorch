# OpenTitans Documentation

This folder is the documentation hub for OpenTitans. The top-level
[README](../README.md) keeps a short quickstart and a feature matrix; all
deep-dives live here.

## Architecture

Neural-memory mechanics, block variants, and composable architectural flags.

- [TNT: Hierarchical Memory](tnt_hierarchical_memory.md) — global + local memories, chunk sizes, reset cadence, two-stage training.
- [Attention Residuals (AttnRes)](attention_residuals.md) — depth-wise softmax over prior block outputs; `AttnResMemoryGate`.
- [Memory Cross-Attention (MCA)](memory_cross_attention.md) — second read interface over the memory MLP's weight rows, sigmoid-gated.
- [Yaad: Huber Attentional Bias](yaad_huber_bias.md) — Miras-style Huber loss for memory updates, data-dependent δ.
- [Adaptive Window Sizing](adaptive_window_sizing.md) — per-layer learned effective window via differentiable soft falloff.
- [Proportional RoPE (p-RoPE)](proportional_rope.md) — Gemma-style partial rotation; split positional vs. semantic dims.
- [Paper Alignment and Deviations](paper_alignment.md) — equation-level tracking of which pieces are faithful, deliberate deviations, or novel extensions.

## Training and Configuration

Runtime-facing knobs, long-running inference plumbing, and state I/O.

- [Configuration Guide](configuration_guide.md) — parameter walk-through with Paper Origin Tags for every flag.
- [Memory State Persistence](memory_persistence.md) — `save_memory_states` / `load_memory_states`, `MemoryDumpManager`, `.npz` file format.
- [Memory Auto-Checkpointing](memory_auto_checkpointing.md) — novelty-triggered state capture, signal log, `MemoryCheckpointConfig`.

## Ecosystem

Interop with the surrounding ML stack.

- [HuggingFace Integration](huggingface_integration.md) — `TitansMACConfig`, `TitansMACForCausalLM`, `TitansTrainer`, `TitansChunkMixin`, checkpoint conversion.

## API Reference

- [Public API (stable surface for 0.7.x)](api.md) — curated top-level exports, deprecated shims, and the stability policy.

---

[Back to project README](../README.md)
