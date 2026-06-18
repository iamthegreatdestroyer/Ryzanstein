"""
Innovation #4: Living Benchmarks as Probes — GlyphBenchmarkIndex Tests
=======================================================================

Validates:
  - record() adds entries and respects max_entries FIFO eviction
  - query_nearest() returns k nearest by glyph centroid distance
  - suggest_params() averages metrics from k-nearest neighbors
  - Semantically similar sequences (same glyph primitives) are nearest-neighbors
  - Serialization round-trip: to_glyph_lattice() → from_glyph_lattice()
  - DistributedServingEngine integration: benchmark_index in get_stats()
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.recycler.glyph_benchmark_index import GlyphBenchmarkIndex, _l2
from src.recycler.glyph_kv_cache import _SIGMALANG_AVAILABLE

sigmalang_required = pytest.mark.skipif(
    not _SIGMALANG_AVAILABLE, reason="sigmalang not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(latency=10.0, hit_rate=0.5, batch=4.0):
    return {"latency_ms": latency, "cache_hit_rate": hit_rate, "batch_size": batch}


# ---------------------------------------------------------------------------
# Basic record / query
# ---------------------------------------------------------------------------

class TestGlyphBenchmarkIndexRecord:

    def test_empty_index(self):
        idx = GlyphBenchmarkIndex()
        assert idx.stats()["entries"] == 0
        assert idx.query_nearest([1, 2, 3]) == []

    def test_record_increments_count(self):
        idx = GlyphBenchmarkIndex()
        idx.record([1, 2, 3, 4], _metrics())
        assert idx.stats()["entries"] == 1
        assert idx.stats()["total_records"] == 1

    def test_multiple_records(self):
        idx = GlyphBenchmarkIndex(max_entries=10)
        for i in range(5):
            idx.record([i, i+1, i+2, i+3], _metrics(latency=float(i)))
        assert idx.stats()["entries"] == 5

    def test_fifo_eviction_at_max(self):
        idx = GlyphBenchmarkIndex(max_entries=3)
        for i in range(5):
            idx.record([i, i+1, i+2, i+3], _metrics())
        assert idx.stats()["entries"] == 3
        assert idx.stats()["total_records"] == 5

    def test_record_empty_tokens(self):
        idx = GlyphBenchmarkIndex()
        idx.record([], _metrics())  # should not crash
        assert idx.stats()["entries"] == 1


class TestGlyphBenchmarkIndexQuery:

    def test_query_returns_at_most_k(self):
        idx = GlyphBenchmarkIndex(k_neighbors=3)
        for i in range(10):
            idx.record([i, i+1, i+2, i+3], _metrics())
        results = idx.query_nearest([0, 1, 2, 3])
        assert len(results) == 3

    def test_query_with_fewer_than_k_entries(self):
        idx = GlyphBenchmarkIndex(k_neighbors=5)
        idx.record([0, 1, 2, 3], _metrics())
        idx.record([4, 5, 6, 7], _metrics())
        results = idx.query_nearest([0, 1, 2, 3])
        assert len(results) == 2

    @sigmalang_required
    def test_identical_sequence_is_nearest(self):
        idx = GlyphBenchmarkIndex(k_neighbors=1)
        target_tokens  = [0, 1, 2, 3]
        distant_tokens = [100, 200, 300, 400]

        idx.record(target_tokens,  _metrics(latency=10.0))
        idx.record(distant_tokens, _metrics(latency=999.0))

        result = idx.query_nearest(target_tokens, k=1)
        assert len(result) == 1
        assert result[0]["latency_ms"] == 10.0, (
            "Identical sequence should be the nearest neighbor"
        )

    @sigmalang_required
    def test_semantically_similar_is_near(self):
        """
        Tokens 256 and 384 map to the same glyph primitive (0x80).
        [0, 1, 2, 256] and [0, 1, 2, 384] should have nearly equal centroids
        and thus tiny L2 distance.
        """
        idx = GlyphBenchmarkIndex()
        tokens_a = [0, 1, 2, 256]
        tokens_b = [0, 1, 2, 384]

        idx.record(tokens_a, _metrics(latency=20.0))
        results = idx.query_nearest(tokens_b, k=1)
        assert len(results) == 1
        assert results[0]["latency_ms"] == 20.0


# ---------------------------------------------------------------------------
# suggest_params
# ---------------------------------------------------------------------------

class TestGlyphBenchmarkIndexSuggest:

    def test_suggest_empty_index(self):
        idx = GlyphBenchmarkIndex()
        assert idx.suggest_params([1, 2, 3]) == {}

    def test_suggest_averages_latency(self):
        idx = GlyphBenchmarkIndex(k_neighbors=2)
        idx.record([0, 1, 2, 3], _metrics(latency=10.0))
        idx.record([0, 1, 2, 3], _metrics(latency=20.0))
        suggestions = idx.suggest_params([0, 1, 2, 3])
        assert "estimated_latency_ms" in suggestions
        assert abs(suggestions["estimated_latency_ms"] - 15.0) < 0.5

    def test_suggest_includes_neighbor_count(self):
        idx = GlyphBenchmarkIndex(k_neighbors=3)
        for _ in range(5):
            idx.record([0, 1, 2, 3], _metrics())
        s = idx.suggest_params([0, 1, 2, 3])
        assert "neighbor_count" in s
        assert s["neighbor_count"] == 3

    def test_suggest_maps_cache_hit_rate(self):
        idx = GlyphBenchmarkIndex(k_neighbors=2)
        idx.record([0, 1, 2, 3], _metrics(hit_rate=0.8))
        idx.record([0, 1, 2, 3], _metrics(hit_rate=0.6))
        s = idx.suggest_params([0, 1, 2, 3])
        assert "estimated_cache_hit_rate" in s
        assert abs(s["estimated_cache_hit_rate"] - 0.7) < 0.01


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestGlyphBenchmarkIndexSerialization:

    def test_empty_round_trip(self):
        idx = GlyphBenchmarkIndex()
        data = idx.to_glyph_lattice()
        assert data[:4] == b"GBIX"
        idx2 = GlyphBenchmarkIndex.from_glyph_lattice(data)
        assert idx2.stats()["entries"] == 0

    def test_populated_round_trip(self):
        idx = GlyphBenchmarkIndex()
        idx.record([0, 1, 2, 3], _metrics(latency=42.0, hit_rate=0.75))
        idx.record([4, 5, 6, 7], _metrics(latency=18.0, hit_rate=0.90))

        data = idx.to_glyph_lattice()
        idx2 = GlyphBenchmarkIndex.from_glyph_lattice(data)

        assert idx2.stats()["entries"] == 2
        results = idx2.query_nearest([0, 1, 2, 3], k=1)
        assert len(results) == 1
        assert results[0]["latency_ms"] == 42.0

    def test_many_entries_round_trip(self):
        idx = GlyphBenchmarkIndex(max_entries=100)
        for i in range(50):
            idx.record([i % 32, (i+1) % 32, (i+2) % 32, (i+3) % 32], _metrics(latency=float(i)))
        data = idx.to_glyph_lattice()
        idx2 = GlyphBenchmarkIndex.from_glyph_lattice(data)
        assert idx2.stats()["entries"] == 50

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="GBIX"):
            GlyphBenchmarkIndex.from_glyph_lattice(b"XXXX\x00\x00\x00\x00")

    def test_lattice_compresses_well(self):
        """Sparse centroids (few unique primitives) should produce small lattices."""
        idx = GlyphBenchmarkIndex()
        for _ in range(5):
            idx.record([0, 1, 2, 3], _metrics())   # only 4 unique primitives

        data = idx.to_glyph_lattice()
        # Header (8B) + 5 entries × ~few bytes each → far less than 5 × 256 × 4 B
        # 5 entries × ~4 nonzero dims × (2+4)B + JSON per entry ≈ 550 bytes worst-case
        # much less than 5 × 256 floats × 4 bytes = 5120 bytes (dense encoding)
        assert len(data) < 600, f"Expected compact lattice, got {len(data)} bytes"


# ---------------------------------------------------------------------------
# Serving engine integration
# ---------------------------------------------------------------------------

class _MockModel(nn.Module):
    def __init__(self, vocab=64):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(1, vocab, bias=False)

    def forward(self, tokens):
        return self.proj(tokens.float().mean(dim=-1, keepdim=True))


@pytest.mark.asyncio
async def test_engine_has_benchmark_index():
    from src.serving.distributed_serving import DistributedServingEngine
    from src.recycler.glyph_benchmark_index import GlyphBenchmarkIndex
    engine = DistributedServingEngine(_MockModel(), num_gpus=1)
    assert hasattr(engine, "benchmark_index")
    assert isinstance(engine.benchmark_index, GlyphBenchmarkIndex)


@pytest.mark.asyncio
async def test_get_stats_includes_benchmark_index():
    from src.serving.distributed_serving import DistributedServingEngine
    engine = DistributedServingEngine(_MockModel(), num_gpus=1)
    stats = await engine.get_stats()
    assert "benchmark_index" in stats
    assert "entries" in stats["benchmark_index"]


@pytest.mark.asyncio
async def test_benchmark_index_populated_after_inference():
    from src.serving.distributed_serving import (
        DistributedServingEngine, InferenceRequest, RequestPriority,
    )
    from src.recycler.glyph_kv_cache import HybridKVCache

    engine = DistributedServingEngine(_MockModel(), num_gpus=1)
    engine.kv_cache = HybridKVCache(max_entries=64, chunk_size=4)

    req = InferenceRequest(
        request_id="bench-test",
        prompt_tokens=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
        max_tokens=4,
        priority=RequestPriority.NORMAL,
    )
    await engine.submit_request(req)
    reqs = await engine.request_queue.dequeue(count=1)
    await engine.batcher.add_requests(reqs)
    batches = await engine.batcher.form_batches()
    with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
        for batch in batches:
            await engine._process_batch(batch)

    assert engine.benchmark_index.stats()["entries"] > 0


@pytest.mark.asyncio
async def test_suggest_params_after_two_batches():
    from src.serving.distributed_serving import (
        DistributedServingEngine, InferenceRequest, RequestPriority,
    )
    from src.recycler.glyph_kv_cache import HybridKVCache

    engine = DistributedServingEngine(_MockModel(), num_gpus=1)
    engine.kv_cache = HybridKVCache(max_entries=64, chunk_size=4)

    tokens = [0, 1, 2, 3, 4, 5, 6, 7]
    for rid in ["r1", "r2"]:
        req = InferenceRequest(
            request_id=rid,
            prompt_tokens=torch.tensor(tokens, dtype=torch.long),
            max_tokens=4,
            priority=RequestPriority.NORMAL,
        )
        await engine.submit_request(req)
        reqs = await engine.request_queue.dequeue(count=1)
        await engine.batcher.add_requests(reqs)
        batches = await engine.batcher.form_batches()
        with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
            for batch in batches:
                await engine._process_batch(batch)

    suggestions = engine.benchmark_index.suggest_params(tokens)
    assert "estimated_latency_ms" in suggestions
    assert suggestions["estimated_latency_ms"] >= 0
