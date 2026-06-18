"""
Living Benchmarks as Probes — Glyph Benchmark Index
=====================================================

Innovation #4 from Ryot-updates.md:

    "Benchmarks stop being reports; they become glyph-encoded performance probes
    that the system consults during inference. When speculative decoding chooses
    a draft size, it queries the glyph space of known benchmarks to find 'what
    worked before in similar conditions?' The benchmark data itself is stored as
    a hierarchical glyph lattice."

Architecture
------------
Each request that completes leaves a performance record:
    (glyph_centroid_vector, {latency_ms, cache_hit_rate, tokens_per_s, ...})

The glyph centroid is a 256-dim histogram over primitive IDs — a bag-of-glyphs
representation that captures what the request was *about* semantically, not what
exact tokens it used. This makes the index transfer across paraphrases and models.

At inference time, `query_nearest(token_ids, k)` runs an L2 nearest-neighbor
search in this centroid space and returns the k most similar historical
performance snapshots. `suggest_params(token_ids)` averages the metrics of the
k-nearest neighbors and returns auto-tuning suggestions:

    {
        "suggested_draft_size": 4,
        "suggested_chunk_size": 256,
        "estimated_latency_ms": 45.2,
        "estimated_cache_hit_rate": 0.72,
    }

The serving engine calls suggest_params() every N batches to continuously
adapt without any model retraining.

Serialization
-------------
`to_glyph_lattice()` serializes the entire index as a GlyphStream — meaning
the benchmark history itself can be stored, transmitted, and queried as compact
glyph bytes. The lattice is queryable offline: load it back with from_glyph_lattice()
and run suggest_params() against it on a different machine or session.
"""

import hashlib
import json
import struct
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from sigmalang.core.primitives import GlyphStream, GlyphType, Glyph
    _SIGMALANG_AVAILABLE = True
except ImportError:
    _SIGMALANG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_Centroid = List[float]          # 256-dim primitive-ID frequency histogram
_Metrics  = Dict[str, float]     # {latency_ms, cache_hit_rate, ...}
_Entry    = Tuple[_Centroid, _Metrics, float]   # (centroid, metrics, timestamp)

# Recommended params that can be auto-tuned
_TUNABLE = {
    "latency_ms":        "estimated_latency_ms",
    "tokens_per_s":      "estimated_tokens_per_s",
    "cache_hit_rate":    "estimated_cache_hit_rate",
    "batch_size":        "suggested_batch_size",
}


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class GlyphBenchmarkIndex:
    """
    Append-only performance index keyed by glyph centroid vectors.

    Thread safety: designed for single-process async use. All mutations are
    simple list/dict appends — safe for asyncio but not multi-process.
    """

    def __init__(
        self,
        max_entries: int = 2_000,
        k_neighbors: int = 5,
    ):
        """
        Args:
            max_entries:  Maximum number of performance records to retain.
                          Oldest are evicted FIFO when the limit is reached.
            k_neighbors:  Default neighbor count for nearest-neighbor queries.
        """
        self._entries: List[_Entry] = []
        self.max_entries  = max_entries
        self.k_neighbors  = k_neighbors
        self._total_records = 0
        self._mapper = None  # lazy-init

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(self, token_ids: List[int], metrics: _Metrics) -> None:
        """
        Add a performance snapshot keyed to the token sequence.

        Args:
            token_ids: Token sequence that produced the measurement.
            metrics:   Dict of float-valued performance metrics, e.g.:
                         {"latency_ms": 42.1, "cache_hit_rate": 0.8,
                          "tokens_per_s": 1200.0, "batch_size": 8}
        """
        centroid = self._glyph_centroid(token_ids)
        ts = time.monotonic()
        self._entries.append((centroid, dict(metrics), ts))
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)   # FIFO eviction
        self._total_records += 1

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query_nearest(
        self,
        token_ids: List[int],
        k: Optional[int] = None,
    ) -> List[_Metrics]:
        """
        Return the k nearest historical metrics by L2 glyph-centroid distance.

        Args:
            token_ids: Current request token sequence.
            k:         Number of neighbors (defaults to self.k_neighbors).

        Returns:
            List of metrics dicts, sorted nearest-first. Empty if no records.
        """
        if not self._entries:
            return []
        k = k or self.k_neighbors
        query_centroid = self._glyph_centroid(token_ids)
        scored: List[Tuple[float, _Metrics]] = []
        for centroid, metrics, _ in self._entries:
            dist = _l2(query_centroid, centroid)
            scored.append((dist, metrics))
        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored[:k]]

    def suggest_params(self, token_ids: List[int]) -> Dict[str, Any]:
        """
        Auto-suggest inference params based on k-nearest historical workloads.

        Returns a dict of suggestions ready to feed directly into the serving
        engine.  Numeric metrics are averaged across neighbors. If the index is
        empty, returns an empty dict so callers can detect "no data" cleanly.

        Example return value:
            {
                "estimated_latency_ms": 38.5,
                "estimated_tokens_per_s": 1350.0,
                "estimated_cache_hit_rate": 0.74,
                "suggested_batch_size": 6.0,
                "suggested_draft_size": 4.0,
                "neighbor_count": 5,
            }
        """
        neighbors = self.query_nearest(token_ids)
        if not neighbors:
            return {}

        # Collect all numeric keys from neighbors
        all_keys = set(k for n in neighbors for k in n if isinstance(n.get(k), (int, float)))

        suggestions: Dict[str, Any] = {}
        for key in all_keys:
            vals = [float(n[key]) for n in neighbors if key in n]
            avg = sum(vals) / len(vals)
            # Map metric names to friendly suggestion labels
            out_key = _TUNABLE.get(key, f"suggested_{key}")
            suggestions[out_key] = round(avg, 4)

        suggestions["neighbor_count"] = len(neighbors)
        return suggestions

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the n most recent entries (metrics + timestamp)."""
        tail = self._entries[-n:]
        return [
            {"metrics": m, "timestamp": round(ts, 3)}
            for _, m, ts in reversed(tail)
        ]

    # ------------------------------------------------------------------
    # Serialization  (Living Benchmark Lattice)
    # ------------------------------------------------------------------

    def to_glyph_lattice(self) -> bytes:
        """
        Serialize the entire index as a compact binary lattice.

        Format (little-endian):
          4B  magic  "GBIX"
          4B  uint32 entry_count
          For each entry:
            4B  uint32 centroid_nonzero_count
            For each nonzero centroid element:
              2B uint16 dimension
              4B float32 value
            4B  uint32 metrics_json_len
            NB  UTF-8 JSON of metrics dict
            8B  float64 timestamp

        Returns raw bytes. The lattice can be stored as a file, database blob,
        or transmitted over the wire.  from_glyph_lattice() reconstructs it.
        """
        buf = bytearray(b"GBIX")
        buf += struct.pack("<I", len(self._entries))
        for centroid, metrics, ts in self._entries:
            nonzero = [(i, v) for i, v in enumerate(centroid) if v > 0.0]
            buf += struct.pack("<I", len(nonzero))
            for dim, val in nonzero:
                buf += struct.pack("<Hf", dim, val)
            metrics_json = json.dumps(metrics).encode("utf-8")
            buf += struct.pack("<I", len(metrics_json))
            buf += metrics_json
            buf += struct.pack("<d", ts)
        return bytes(buf)

    @classmethod
    def from_glyph_lattice(cls, data: bytes, **kwargs) -> "GlyphBenchmarkIndex":
        """Reconstruct a GlyphBenchmarkIndex from serialized lattice bytes."""
        idx = cls(**kwargs)
        if data[:4] != b"GBIX":
            raise ValueError("Not a GBIX lattice (bad magic bytes)")
        pos = 4
        (entry_count,) = struct.unpack_from("<I", data, pos); pos += 4
        for _ in range(entry_count):
            centroid = [0.0] * 256
            (nz_count,) = struct.unpack_from("<I", data, pos); pos += 4
            for _ in range(nz_count):
                dim, val = struct.unpack_from("<Hf", data, pos); pos += 6
                centroid[dim] = val
            (jlen,) = struct.unpack_from("<I", data, pos); pos += 4
            metrics = json.loads(data[pos:pos + jlen].decode("utf-8")); pos += jlen
            (ts,) = struct.unpack_from("<d", data, pos); pos += 8
            idx._entries.append((centroid, metrics, ts))
        idx._total_records = entry_count
        return idx

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        recent_latencies = [
            e[1]["latency_ms"]
            for e in self._entries[-50:]
            if "latency_ms" in e[1]
        ]
        return {
            "entries":          len(self._entries),
            "max_entries":      self.max_entries,
            "total_records":    self._total_records,
            "k_neighbors":      self.k_neighbors,
            "avg_latency_ms":   round(sum(recent_latencies) / len(recent_latencies), 2)
                                if recent_latencies else None,
            "sigmalang_active": _SIGMALANG_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _glyph_centroid(self, token_ids: List[int]) -> _Centroid:
        """
        256-dim normalized primitive-ID histogram (bag-of-glyphs).

        Returns a vector that represents *what the sequence is about*
        in glyph-space, independent of exact token choices. Two semantically
        similar sequences produce similar centroids and thus small L2 distance.
        """
        vec = [0.0] * 256
        if not token_ids:
            return vec
        if _SIGMALANG_AVAILABLE:
            try:
                mapper = self._get_mapper()
                glyphs = mapper.tokens_to_glyphs(token_ids)
                for g in glyphs:
                    vec[g.primitive_id] += 1.0
            except Exception:
                self._fallback_centroid(token_ids, vec)
        else:
            self._fallback_centroid(token_ids, vec)

        total = sum(vec) or 1.0
        return [v / total for v in vec]

    @staticmethod
    def _fallback_centroid(token_ids: List[int], vec: List[float]) -> None:
        """Bucket tokens into 256 bins when sigmalang is absent."""
        for tid in token_ids:
            vec[tid % 256] += 1.0

    def _get_mapper(self):
        if self._mapper is None:
            from .glyph_kv_cache import TokenGlyphMapper
            self._mapper = TokenGlyphMapper()
        return self._mapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
