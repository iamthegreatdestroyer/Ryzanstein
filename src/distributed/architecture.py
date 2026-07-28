"""
Distributed architecture configuration.

Provides configuration dataclasses for distributed inference architecture.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from .tensor_parallel import TensorParallelConfig


@dataclass
class DistributedConfig:
    """Configuration for distributed inference architecture."""

    world_size: int = 1
    rank: int = 0
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    hidden_size: int = 1024
    num_attention_heads: int = 32
    intermediate_size: int = 4096
    num_layers: int = 2


__all__ = [
    "DistributedConfig",
    "TensorParallelConfig",
]
