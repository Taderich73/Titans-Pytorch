# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans: Learning to Memorize at Test Time — PyTorch Implementation.

Usage:
    import torch
    from titans import TitansConfig, TitansMAC

    config = TitansConfig(dim=512, num_heads=8, num_layers=6)
    model = TitansMAC(config)

    x = torch.randint(0, config.vocab_size, (2, 512))
    logits, states = model(x)
"""

__version__ = "0.2.0"
