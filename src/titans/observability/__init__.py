"""Training observability: metrics writer + four independent feature modules."""

from titans.observability.eval_loop import (
    EvalConfig,
    is_eval_example,
    restore_memory_states,
    run_eval,
    stash_memory_states,
)
from titans.observability.gate_hooks import GateHookRegistry
from titans.observability.grad_norm import global_grad_norm
from titans.observability.layer_stats import LayerStats, collect_layer_stats
from titans.observability.metrics_writer import (
    MetricsWriter,
    NullMetricsWriter,
    build_metrics_writer,
)

__all__ = [
    "EvalConfig",
    "GateHookRegistry",
    "LayerStats",
    "MetricsWriter",
    "NullMetricsWriter",
    "build_metrics_writer",
    "collect_layer_stats",
    "global_grad_norm",
    "is_eval_example",
    "restore_memory_states",
    "run_eval",
    "stash_memory_states",
]
