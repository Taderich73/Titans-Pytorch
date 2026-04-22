# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state containers.

Lightweight dataclasses that hold the state of a
:class:`~titans.memory.NeuralLongTermMemory` (plain) or its TNT-hierarchical
variant across forward calls. Split out of the monolithic ``memory.py`` so the
dataclass-only concerns stay decoupled from the large ``nn.Module`` implementing
the actual learn-at-test-time update rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemoryState:
    """State of the neural long-term memory."""

    weights: list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> MemoryState:
        return MemoryState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        return MemoryState(
            weights=[w.detach().clone() for w in self.weights],
            momentum=[m.detach().clone() for m in self.momentum],
        )


@dataclass
class TNTMemoryState:
    """State for TNT hierarchical memory system.

    Attributes:
        global_state: MemoryState for the global memory (V)
        local_states: List of MemoryState, one per local memory (W^(i))
        qk_projections: Accumulated Q-K projection matrices (M_t^(i))
        local_step_counters: Position within shard for each local memory
    """

    global_state: MemoryState
    local_states: list[MemoryState]
    qk_projections: list[torch.Tensor]
    local_step_counters: list[int]

    def detach(self) -> TNTMemoryState:
        return TNTMemoryState(
            global_state=self.global_state.detach(),
            local_states=[s.detach() for s in self.local_states],
            qk_projections=[qk.detach() for qk in self.qk_projections],
            local_step_counters=list(self.local_step_counters),
        )

    def clone(self) -> TNTMemoryState:
        return TNTMemoryState(
            global_state=self.global_state.clone(),
            local_states=[s.clone() for s in self.local_states],
            qk_projections=[qk.detach().clone() for qk in self.qk_projections],
            local_step_counters=list(self.local_step_counters),
        )
