"""
Inference package for KV-Cache optimization.

Provides distributed KV-cache sharding, FP8 compression,
and dynamic memory allocation for efficient LLM inference.
"""

from src.inference.distributed_kv_cache import (
    DistributedKVCache,
    CacheShard,
    ConsistencyManager,
    KVCacheCommunicator,
    ConsistencyMode,
)
from src.inference.cache_compression import (
    FP8Compressor,
    CompressedKVCache,
    CompressionAccuracyValidator,
)
from src.inference.dynamic_allocator import (
    DynamicCacheAllocator,
    MemoryPressureMonitor,
    EvictionPolicy,
)

__all__ = [
    "DistributedKVCache",
    "CacheShard",
    "ConsistencyManager",
    "KVCacheCommunicator",
    "ConsistencyMode",
    "FP8Compressor",
    "CompressedKVCache",
    "CompressionAccuracyValidator",
    "DynamicCacheAllocator",
    "MemoryPressureMonitor",
    "EvictionPolicy",
]
