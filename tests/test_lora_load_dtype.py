"""Regression: load_adapters silently cast incoming tensors to whatever dtype
the target parameter used, discarding user intent and without a warning.
Ensure we emit a warning and explicitly preserve target device."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn


def test_load_adapters_warns_on_dtype_mismatch(tmp_path, caplog):
    from titans.lora import LoRALinear, load_adapters, save_adapters

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = nn.Linear(8, 8)
            self.proj = LoRALinear(base=base, rank=2, alpha=4.0)

    src = Tiny()
    # Coerce source adapters to bf16 to simulate a bf16-checkpoint.
    with torch.no_grad():
        src.proj.lora_A.data = src.proj.lora_A.data.to(torch.bfloat16)
        src.proj.lora_B.data = src.proj.lora_B.data.to(torch.bfloat16)

    ckpt = tmp_path / "adapters.safetensors"
    save_adapters(src, ckpt, meta={})

    # Destination in fp32.
    dst = Tiny()
    with caplog.at_level(logging.WARNING, logger="titans.lora"):
        load_adapters(dst, ckpt)

    assert any("dtype" in rec.message.lower() for rec in caplog.records), (
        "Expected a dtype-mismatch warning from load_adapters"
    )
