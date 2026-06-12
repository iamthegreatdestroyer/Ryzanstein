"""
Semantic KV Cache — chunk-aligned prefix caching for inference speedup.

Design: Cache KV tensors at 256-token chunk-aligned prefix boundaries.
When a new request shares a prefix with a cached entry, only the uncached
suffix is forwarded through the transformer. Common patterns (system prompts,
few-shot examples) are computed once and reused across requests.

Reference: ChunkKV (arxiv:2502.00299) — chunk-based KV compression with
layer-wise index reuse. SemantiCache (arxiv:2603.14303) — semantic clustering
for KV cache entries.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

CHUNK_SIZE = 256  # Cache at chunk-aligned token boundaries (power of 2)


@dataclass
class KVCacheEntry:
    prefix_hash: str
    kv_tensors: list        # per-layer list of (K, V) tensors
    token_count: int        # number of tokens this entry covers
    last_used: float = field(default_factory=time.monotonic)
    hit_count: int = 0


class SemanticKVCache:
    """
    Chunk-aligned KV cache for CPU inference.

    For GPU PagedAttention is the standard; on CPU the equivalent benefit
    comes from prefix-level caching at chunk-aligned boundaries. Requests
    sharing a long common prefix (system prompts, few-shot examples) pay
    the forward-pass cost only once.
    """

    def __init__(self, max_entries: int = 512, chunk_size: int = CHUNK_SIZE):
        self._cache: dict[str, KVCacheEntry] = {}
        self.max_entries = max_entries
        self.chunk_size = chunk_size

        # Stats
        self.total_lookups = 0
        self.cache_hits = 0
        self.total_stores = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def lookup(self, token_ids: list[int]) -> tuple[list, int]:
        """Return (kv_tensors, prefix_length) for the longest cached prefix.

        Checks progressively longer chunk-aligned prefixes and returns the
        deepest cache hit found. Returns ([], 0) on total miss.
        """
        self.total_lookups += 1
        best_kv: list = []
        best_len: int = 0

        boundary = self.chunk_size
        while boundary <= len(token_ids):
            h = self._hash(token_ids[:boundary])
            entry = self._cache.get(h)
            if entry is not None:
                entry.last_used = time.monotonic()
                entry.hit_count += 1
                best_kv = entry.kv_tensors
                best_len = boundary
            boundary += self.chunk_size

        if best_len > 0:
            self.cache_hits += 1

        return best_kv, best_len

    def store(self, token_ids: list[int], kv_tensors: list) -> None:
        """Store KV tensors indexed at the longest chunk-aligned boundary."""
        if not kv_tensors:
            return

        boundary = (len(token_ids) // self.chunk_size) * self.chunk_size
        if boundary == 0:
            return  # sequence too short to cache

        if len(self._cache) >= self.max_entries:
            self._evict_lru()

        h = self._hash(token_ids[:boundary])
        self._cache[h] = KVCacheEntry(
            prefix_hash=h,
            kv_tensors=kv_tensors,
            token_count=boundary,
        )
        self.total_stores += 1

    def invalidate(self, prefix_hash: Optional[str] = None) -> None:
        """Remove a specific entry or clear the entire cache."""
        if prefix_hash is not None:
            self._cache.pop(prefix_hash, None)
        else:
            self._cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        hit_rate = (self.cache_hits / self.total_lookups) if self.total_lookups else 0.0
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "hit_rate": round(hit_rate, 4),
            "total_stores": self.total_stores,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(token_ids: list[int]) -> str:
        return hashlib.sha256(
            b"".join(t.to_bytes(4, "little") for t in token_ids)
        ).hexdigest()

    def _evict_lru(self) -> None:
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].last_used)
        del self._cache[oldest_key]
