"""
Sprint 1 E2E Integration Test: HybridKVCache ↔ DistributedServingEngine
========================================================================

Validates the full integration path:
  - DistributedServingEngine holds a HybridKVCache
  - get_stats() exposes kv_cache sub-dict with engine hit counters
  - Identical prompts hit the cache on second submission
  - Semantic dedup (glyph path): tokens that share a Σ-glyph share cache entries
  - Cache invalidation resets lookup to miss

All tests run on CPU; torch.Tensor.cuda() is patched to identity so the engine's
GPU selection step works without actual CUDA hardware.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.serving.distributed_serving import (
    DistributedServingEngine,
    InferenceRequest,
    RequestPriority,
)
from src.recycler.glyph_kv_cache import HybridKVCache, GlyphKVCache


# ---------------------------------------------------------------------------
# Mock model — deterministic CPU inference, no GPU needed
# ---------------------------------------------------------------------------

class MockModel(nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        torch.manual_seed(42)
        self.proj = nn.Linear(1, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, seq_len] → one logit row per request
        mean_val = tokens.float().mean(dim=-1, keepdim=True)
        return self.proj(mean_val)  # [batch, vocab_size]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(token_ids, req_id: str = None, max_tokens: int = 4) -> InferenceRequest:
    rid = req_id or f"r{hash(tuple(token_ids)) & 0xFFFF:04x}"
    return InferenceRequest(
        request_id=rid,
        prompt_tokens=torch.tensor(token_ids, dtype=torch.long),
        max_tokens=max_tokens,
        priority=RequestPriority.NORMAL,
    )


def _make_engine(chunk_size: int = 4, max_entries: int = 256) -> DistributedServingEngine:
    engine = DistributedServingEngine(MockModel(), num_gpus=1)
    engine.kv_cache = HybridKVCache(max_entries=max_entries, chunk_size=chunk_size)
    return engine


async def _run_batch(engine, requests):
    """Submit requests and drain one full serving loop cycle."""
    ids = [await engine.submit_request(r) for r in requests]
    reqs = await engine.request_queue.dequeue(count=len(requests))
    await engine.batcher.add_requests(reqs)
    batches = await engine.batcher.form_batches()
    # Patch cuda() to identity so the test runs on CPU
    with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
        for batch in batches:
            await engine._process_batch(batch)
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_has_hybrid_kv_cache():
    engine = _make_engine()
    assert isinstance(engine.kv_cache, HybridKVCache)
    assert hasattr(engine, "_cache_hits")
    assert hasattr(engine, "_cache_misses")


@pytest.mark.asyncio
async def test_get_stats_kv_cache_section():
    engine = _make_engine()
    stats = await engine.get_stats()
    assert "kv_cache" in stats
    kv = stats["kv_cache"]
    assert "engine_hit_rate" in kv
    assert "engine_cache_hits" in kv
    assert "engine_cache_misses" in kv


@pytest.mark.asyncio
async def test_first_request_is_miss():
    engine = _make_engine(chunk_size=4)
    await _run_batch(engine, [_req([1, 2, 3, 4, 5, 6, 7, 8])])
    assert engine._cache_misses == 1
    assert engine._cache_hits == 0


@pytest.mark.asyncio
async def test_second_identical_request_hits():
    """
    First submission: cache miss, stored.
    Second submission: cache hit because tokens match a stored entry.
    Uses chunk_size=4 so a 8-token sequence creates a cacheable chunk.
    """
    engine = _make_engine(chunk_size=4)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    await _run_batch(engine, [_req(tokens, req_id="r001")])
    hits_after_first = engine._cache_hits

    await _run_batch(engine, [_req(tokens, req_id="r002")])
    hits_after_second = engine._cache_hits

    assert hits_after_second > hits_after_first, (
        f"Expected a cache hit on second identical request. "
        f"hits before={hits_after_first}, after={hits_after_second}"
    )


@pytest.mark.asyncio
async def test_engine_hit_rate_nonzero_after_repeated_prompts():
    engine = _make_engine(chunk_size=4)
    tokens = [0, 1, 2, 3, 4, 5, 6, 7]
    await _run_batch(engine, [_req(tokens, req_id="r-a")])
    await _run_batch(engine, [_req(tokens, req_id="r-b")])

    stats = await engine.get_stats()
    assert stats["kv_cache"]["engine_hit_rate"] > 0.0


@pytest.mark.asyncio
async def test_semantic_dedup_glyph_cache_direct():
    """
    Unit-level: tokens 256 and 384 both map to primitive_id 0x80.
    GlyphKVCache must return a hit when the second sequence is looked up.
    """
    cache = GlyphKVCache(max_entries=64, chunk_size=4)
    kv = [[1.0, 2.0, 3.0]]

    tokens_a = [0, 1, 2, 256]   # 256 → (256-128)%128 + 0x80 = 0x80
    tokens_b = [0, 1, 2, 384]   # 384 → (384-128)%128 + 0x80 = 0x80

    cache.store(tokens_a, kv)
    _, hit_len = cache.lookup(tokens_b)

    assert hit_len == 4, (
        f"Tokens 256 and 384 should collide in the glyph cache (both primitive 0x80). "
        f"Got hit_len={hit_len}"
    )


@pytest.mark.asyncio
async def test_cache_invalidate_causes_miss():
    engine = _make_engine(chunk_size=4)
    tokens = [0, 1, 2, 3, 4, 5, 6, 7]

    await _run_batch(engine, [_req(tokens, req_id="r-x")])

    engine.kv_cache.invalidate()
    engine._cache_hits = 0
    engine._cache_misses = 0

    await _run_batch(engine, [_req(tokens, req_id="r-y")])

    assert engine._cache_hits == 0, "After invalidation, lookup should miss"
    assert engine._cache_misses == 1


@pytest.mark.asyncio
async def test_multiple_unique_requests_all_miss():
    engine = _make_engine(chunk_size=4)
    requests = [_req([i, i+1, i+2, i+3, i+4, i+5, i+6, i+7], req_id=f"u{i}") for i in range(5)]
    await _run_batch(engine, requests)

    assert engine._cache_hits == 0
    assert engine._cache_misses == 5


@pytest.mark.asyncio
async def test_responses_stored_in_processed_requests():
    engine = _make_engine(chunk_size=4)
    requests = [_req([i, i+1, i+2, i+3], req_id=f"resp-{i}") for i in range(3)]
    ids = await _run_batch(engine, requests)

    async with engine.lock:
        present = set(engine.processed_requests.keys())

    assert set(ids).issubset(present), (
        f"Missing responses for: {set(ids) - present}"
    )
