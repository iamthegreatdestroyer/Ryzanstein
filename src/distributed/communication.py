"""
Communication layer for distributed inference.

Provides NCCL and fallback communicators for tensor parallel operations.
"""

import logging
from typing import List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class NCCLCommunicator:
    """
    NCCL-based communicator for distributed tensor operations.

    Provides all-reduce, all-gather, and reduce-scatter primitives.
    Falls back to single-rank no-op behavior when distributed is not initialized.
    """

    def __init__(self):
        """Initialize NCCL communicator."""
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform all-reduce sum across all ranks.

        Args:
            tensor: Input tensor (modified in-place)

        Returns:
            All-reduced tensor
        """
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensors from all ranks.

        Args:
            tensor: Local tensor to gather

        Returns:
            List of tensors from all ranks
        """
        if dist.is_initialized() and self.world_size > 1:
            gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, tensor)
            return gathered
        else:
            return [tensor]

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce-scatter across all ranks.

        Args:
            tensor: Input tensor

        Returns:
            Reduced and scattered tensor
        """
        if dist.is_initialized() and self.world_size > 1:
            output = torch.zeros(
                tensor.shape[0] // self.world_size, *tensor.shape[1:],
                dtype=tensor.dtype, device=tensor.device
            )
            dist.reduce_scatter(output, list(tensor.chunk(self.world_size)))
            return output
        else:
            return tensor


__all__ = [
    "NCCLCommunicator",
]
