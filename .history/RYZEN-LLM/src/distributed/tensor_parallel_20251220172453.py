"""
Tensor Parallelism Implementation

Implements row-wise and column-wise tensor parallelism for LLM layers.
Provides parallelized Linear, Attention, and MLP layers with NCCL communication.

Key Components:
    - RowParallelLinear: Row-wise parallel linear layer
    - ColumnParallelLinear: Column-wise parallel linear layer
    - ParallelAttention: Head-wise parallel attention
    - ParallelMLP: Parallel multi-layer perceptron
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging

from .architecture import TensorParallelLayer, CommunicationHandler
from .communication import NCCLCommunicator

logger = logging.getLogger(__name__)


class RowParallelLinear(TensorParallelLayer):
    """Row-wise parallel linear layer.

    Weights are sharded across output dimension (D_out).
    Input is replicated, output is reduced via all-reduce.

    Memory: O(D_in * D_out/TP + D_out/TP)
    Communication: all_reduce after forward
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 gather_output: bool = True,
                 comm_handler: Optional[CommunicationHandler] = None):
        """Initialize row-parallel linear layer.

        Args:
            input_size: Input dimension (replicated)
            output_size: Output dimension (sharded)
            bias: Whether to include bias term
            gather_output: Whether to gather output across ranks
            comm_handler: Communication handler for collectives
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        # Get tensor parallel configuration
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.world_size = world_size
        self.rank = rank

        # Shard output dimension
        assert output_size % world_size == 0, f"output_size ({output_size}) must be divisible by world_size ({world_size})"
        self.output_size_per_partition = output_size // world_size

        # Create local weight and bias
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            input_size,
            dtype=torch.float32  # Use float32 for CPU testing
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                dtype=torch.float32
            ))
        else:
            self.bias = None

        self.comm_handler = comm_handler or NCCLCommunicator()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with row-wise parallelism.

        Args:
            input_: Input tensor of shape (..., input_size)

        Returns:
            Output tensor of shape (..., output_size)
        """
        # Local computation: input @ weight.T + bias
        output_parallel = F.linear(input_, self.weight, self.bias)

        # All-reduce to combine partial outputs
        if self.world_size > 1:
            output = self.comm_handler.all_reduce(output_parallel, op="sum")
        else:
            output = output_parallel

        return output

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, output_size={self.output_size}, bias={self.bias is not None}"


class ColumnParallelLinear(TensorParallelLayer):
    """Column-wise parallel linear layer.

    Weights are sharded across input dimension (D_in).
    Input is sharded, output is replicated.

    Memory: O(D_in/TP * D_out + D_out)
    Communication: reduce_scatter on input (implicit in all_reduce)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 gather_output: bool = True,
                 comm_handler: Optional[CommunicationHandler] = None):
        """Initialize column-parallel linear layer.

        Args:
            input_size: Input dimension (sharded)
            output_size: Output dimension (replicated)
            bias: Whether to include bias term
            gather_output: Whether to gather output across ranks
            comm_handler: Communication handler for collectives
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        # Get tensor parallel configuration
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.world_size = world_size
        self.rank = rank

        # Shard input dimension
        assert input_size % world_size == 0, f"input_size ({input_size}) must be divisible by world_size ({world_size})"
        self.input_size_per_partition = input_size // world_size

        # Create local weight and bias
        self.weight = nn.Parameter(torch.empty(
            output_size,
            self.input_size_per_partition,
            dtype=torch.float32
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                output_size,
                dtype=torch.float32
            ))
        else:
            self.bias = None

        self.comm_handler = comm_handler or NCCLCommunicator()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with column-wise parallelism.

        Args:
            input_: Input tensor of shape (..., input_size)

        Returns:
            Output tensor of shape (..., output_size)
        """
        # Local computation: input @ weight.T + bias
        output = F.linear(input_, self.weight, self.bias)

        return output

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, output_size={self.output_size}, bias={self.bias is not None}"