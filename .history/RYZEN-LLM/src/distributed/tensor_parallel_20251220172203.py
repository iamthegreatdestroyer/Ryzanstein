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
            dtype=torch.float16
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                dtype=torch.float16
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
            dtype=torch.float16
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                output_size,
                dtype=torch.float16
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


class ParallelAttention(TensorParallelLayer):
    """Parallel multi-head attention with head-wise sharding.

    Attention heads are distributed across GPUs.
    KV-cache is sharded along head dimension.
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int = None,
                 max_position_embeddings: int = 4096,
                 comm_handler: Optional[CommunicationHandler] = None):
        """Initialize parallel attention.

        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads (for GQA)
            max_position_embeddings: Maximum sequence length
            comm_handler: Communication handler
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        # Get tensor parallel configuration
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.world_size = world_size
        self.rank = rank

        # Shard attention heads
        assert num_attention_heads % world_size == 0, f"num_attention_heads ({num_attention_heads}) must be divisible by world_size ({world_size})"
        self.num_attention_heads_per_partition = num_attention_heads // world_size

        # Head dimension
        self.head_dim = hidden_size // num_attention_heads
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        # Create parallel linear layers
        self.q_proj = ColumnParallelLinear(hidden_size, hidden_size, bias=False, comm_handler=comm_handler)
        self.k_proj = ColumnParallelLinear(hidden_size, hidden_size, bias=False, comm_handler=comm_handler)
        self.v_proj = ColumnParallelLinear(hidden_size, hidden_size, bias=False, comm_handler=comm_handler)
        self.o_proj = RowParallelLinear(hidden_size, hidden_size, bias=False, comm_handler=comm_handler)

        self.comm_handler = comm_handler or NCCLCommunicator()

        # KV cache (sharded along head dimension)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for parallel attention.

        Args:
            hidden_states: Input hidden states (B, L, H)
            attention_mask: Attention mask (B, 1, L, L)
            position_ids: Position IDs (B, L)

        Returns:
            Output tensor (B, L, H)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)  # (B, L, H)
        key_states = self.k_proj(hidden_states)    # (B, L, H)
        value_states = self.v_proj(hidden_states)  # (B, L, H)

        # Reshape for attention computation
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads_per_partition, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads_per_partition, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads_per_partition, self.head_dim)

        # Transpose for attention: (B, num_heads, L, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Update KV cache if needed
        if self.k_cache is None or self.v_cache is None:
            self.k_cache = key_states
            self.v_cache = value_states
        else:
            self.k_cache = torch.cat([self.k_cache, key_states], dim=2)
            self.v_cache = torch.cat([self.v_cache, value_states], dim=2)

        # Attention computation (local to this rank)
        attn_weights = torch.matmul(query_states, self.k_cache.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, self.v_cache)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)

        return output

    def reset_cache(self):
        """Reset KV cache."""
        self.k_cache = None
        self.v_cache = None


class ParallelMLP(TensorParallelLayer):
    """Parallel multi-layer perceptron with tensor parallelism.

    Uses column-parallel for up/gate projections, row-parallel for down projection.
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 activation_fn: str = "silu",
                 comm_handler: Optional[CommunicationHandler] = None):
        """Initialize parallel MLP.

        Args:
            hidden_size: Input/output hidden dimension
            intermediate_size: Intermediate dimension
            activation_fn: Activation function name
            comm_handler: Communication handler
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_fn = activation_fn

        # Parallel linear layers
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, comm_handler=comm_handler)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, comm_handler=comm_handler)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, comm_handler=comm_handler)

        self.comm_handler = comm_handler or NCCLCommunicator()

        # Activation function
        if activation_fn == "silu":
            self.act_fn = F.silu
        elif activation_fn == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for parallel MLP.

        Args:
            hidden_states: Input tensor (B, L, H)

        Returns:
            Output tensor (B, L, H)
        """
        # Up projection (column-parallel)
        up_states = self.up_proj(hidden_states)

        # Gate projection (column-parallel)
        gate_states = self.gate_proj(hidden_states)

        # Activation
        intermediate_states = self.act_fn(gate_states) * up_states

        # Down projection (row-parallel)
        output_states = self.down_proj(intermediate_states)

        return output_states</content>
<parameter name="filePath">c:\Users\sgbil\Ryot\RYZEN-LLM\src\distributed\tensor_parallel.py