"""Regression test: dpo.py logs `chosen_reward` and `rejected_reward` as the
positive/negative split of their difference, which is wrong. Those should be
the actual per-sample beta * log(pi / pi_ref) terms."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_DPO_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo.py"
_spec = importlib.util.spec_from_file_location("scripts_dpo_mod", _DPO_PATH)
_dpo_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts_dpo_mod"] = _dpo_mod
_spec.loader.exec_module(_dpo_mod)


def test_dpo_loss_returns_split_rewards():
    dpo_loss = _dpo_mod.dpo_loss

    # Tiny synthetic log-probs.
    pol_chosen = torch.tensor([0.3, 0.4])
    pol_rejected = torch.tensor([0.1, 0.2])
    ref_chosen = torch.tensor([0.2, 0.3])
    ref_rejected = torch.tensor([0.15, 0.15])
    beta = 0.5

    result = dpo_loss(
        pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=beta
    )
    # New contract: return (loss, chosen_rewards, rejected_rewards).
    assert len(result) == 3, (
        f"dpo_loss must return (loss, chosen_rewards, rejected_rewards); "
        f"got tuple of length {len(result)}"
    )
    loss, chosen_rewards, rejected_rewards = result
    # chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    expected_chosen = beta * (pol_chosen - ref_chosen)
    expected_rejected = beta * (pol_rejected - ref_rejected)
    assert torch.allclose(chosen_rewards, expected_chosen)
    assert torch.allclose(rejected_rewards, expected_rejected)
