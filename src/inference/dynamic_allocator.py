"""
Dynamic Cache Allocator with Eviction and Memory Pressure Monitoring

Provides:
- Dynamic memory allocation for KV-cache
- LRU and other eviction policies
- Memory pressure monitoring
- Access pattern tracking
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


@dataclass
class CacheAllocation:
    """Tracks a single cache allocation."""
    request_id: str
    memory_bytes: int
    num_layers: int
    num_heads: int
    head_dim: int
    seq_len: int
    compressed: bool
    priority: int = 0
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryPool:
    """A memory pool for a specific layer."""
    layer_id: int
    allocated_bytes: int = 0
    used_bytes: int = 0


class MemoryPressureMonitor:
    """Monitors system memory pressure and triggers eviction when needed."""

    def __init__(self, total_memory_bytes: int, safety_margin: float = 0.1):
        self.total_memory_bytes = total_memory_bytes
        self.safety_margin = safety_margin
        self.usable_bytes = int(total_memory_bytes * (1 - safety_margin))
        self.current_used = 0

    def can_allocate(self, requested_bytes: int) -> bool:
        """Check if allocation is possible without exceeding memory limits."""
        return (self.current_used + requested_bytes) <= self.usable_bytes

    def allocate(self, num_bytes: int) -> None:
        self.current_used += num_bytes

    def deallocate(self, num_bytes: int) -> None:
        self.current_used = max(0, self.current_used - num_bytes)

    def get_pressure(self) -> float:
        """Return current memory pressure as a fraction [0, 1]."""
        if self.usable_bytes == 0:
            return 1.0
        return self.current_used / self.usable_bytes


class DynamicCacheAllocator:
    """
    Dynamic cache memory allocator with eviction support.

    Manages memory allocation for KV-cache across requests,
    with configurable eviction policies when memory is exhausted.
    """

    def __init__(self, total_memory_gb: float = 16.0,
                 safety_margin: float = 0.1,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.total_memory_bytes = int(total_memory_gb * 1024 * 1024 * 1024)
        self.eviction_policy = eviction_policy

        self._monitor = MemoryPressureMonitor(self.total_memory_bytes, safety_margin)

        # Active allocations keyed by request_id
        self.global_allocations: Dict[str, CacheAllocation] = {}

        # Memory pools keyed by layer_id
        self.memory_pools: Dict[int, MemoryPool] = {}

        # Eviction tracking
        self._total_evictions = 0

    def calculate_memory_requirement(self, seq_len: int, num_layers: int,
                                     num_heads: int, head_dim: int,
                                     compressed: bool = False) -> int:
        """
        Calculate memory requirement in bytes.

        Uncompressed: seq_len * num_layers * num_heads * head_dim * 2 (K+V) * 2 (FP16 bytes)
        Compressed: half of uncompressed (FP8)
        """
        memory = seq_len * num_layers * num_heads * head_dim * 2 * 2
        if compressed:
            memory = memory // 2
        return memory

    def allocate_cache(self, request_id: str, seq_len: int,
                       num_layers: int, num_heads: int, head_dim: int,
                       compressed: bool, priority: int = 0) -> bool:
        """
        Allocate cache memory for a request.

        Returns True if allocation succeeded, False if memory is exhausted
        even after eviction attempts.
        """
        memory_needed = self.calculate_memory_requirement(
            seq_len, num_layers, num_heads, head_dim, compressed
        )

        # Try eviction if not enough memory
        if not self._monitor.can_allocate(memory_needed):
            self._try_evict(memory_needed)

        if not self._monitor.can_allocate(memory_needed):
            return False

        # Perform allocation
        self._monitor.allocate(memory_needed)

        allocation = CacheAllocation(
            request_id=request_id,
            memory_bytes=memory_needed,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            compressed=compressed,
            priority=priority,
            access_count=0,
            last_access=time.time(),
            created_at=time.time(),
        )
        self.global_allocations[request_id] = allocation

        # Create memory pools per layer
        for layer_id in range(num_layers):
            if layer_id not in self.memory_pools:
                self.memory_pools[layer_id] = MemoryPool(layer_id=layer_id)
            layer_bytes = memory_needed // num_layers
            self.memory_pools[layer_id].allocated_bytes += layer_bytes
            self.memory_pools[layer_id].used_bytes += layer_bytes

        return True

    def deallocate_cache(self, request_id: str) -> None:
        """Deallocate cache memory for a request."""
        if request_id not in self.global_allocations:
            return

        allocation = self.global_allocations[request_id]
        self._monitor.deallocate(allocation.memory_bytes)

        # Update pool stats
        layer_bytes = allocation.memory_bytes // allocation.num_layers
        for layer_id in range(allocation.num_layers):
            if layer_id in self.memory_pools:
                pool = self.memory_pools[layer_id]
                pool.allocated_bytes = max(0, pool.allocated_bytes - layer_bytes)
                pool.used_bytes = max(0, pool.used_bytes - layer_bytes)

        del self.global_allocations[request_id]

    def access_cache(self, request_id: str) -> None:
        """Record a cache access for eviction tracking."""
        if request_id in self.global_allocations:
            alloc = self.global_allocations[request_id]
            alloc.access_count += 1
            alloc.last_access = time.time()

    def _try_evict(self, needed_bytes: int) -> None:
        """Try to evict allocations to free the needed bytes."""
        if not self.global_allocations:
            return

        # Sort by eviction policy
        if self.eviction_policy == EvictionPolicy.LRU:
            candidates = sorted(
                self.global_allocations.keys(),
                key=lambda rid: self.global_allocations[rid].last_access
            )
        elif self.eviction_policy == EvictionPolicy.LFU:
            candidates = sorted(
                self.global_allocations.keys(),
                key=lambda rid: self.global_allocations[rid].access_count
            )
        else:  # FIFO
            candidates = sorted(
                self.global_allocations.keys(),
                key=lambda rid: self.global_allocations[rid].created_at
            )

        for rid in candidates:
            if self._monitor.can_allocate(needed_bytes):
                break
            self._total_evictions += 1
            self.deallocate_cache(rid)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory usage statistics."""
        used_bytes = self._monitor.current_used
        total_bytes = self.total_memory_bytes

        return {
            "total_memory_gb": total_bytes / (1024 ** 3),
            "used_memory_gb": used_bytes / (1024 ** 3),
            "available_memory_gb": (total_bytes - used_bytes) / (1024 ** 3),
            "utilization_percent": (used_bytes / total_bytes * 100) if total_bytes > 0 else 0.0,
            "pools": len(self.memory_pools),
            "total_evictions": self._total_evictions,
        }
