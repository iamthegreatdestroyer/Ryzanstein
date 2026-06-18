"""
Sprint 1: Glyph-Native Serialization — Test Suite
==================================================

Tests for:
  - TokenGlyphMapper (token → glyph deterministic mapping)
  - GlyphKVCache (glyph-hashed KV cache)
  - HybridKVCache (dual-path: token-exact + glyph-semantic)
  - Compression ratio assertions
  - Cache hit/miss semantics
  - Round-trip glyph stream serialization (requires sigmalang)

Run with:
    cd S:\\repos\\Layer-4-Storage\\Ryot
    python -m pytest tests/test_glyph_kv_cache.py -v
"""

import hashlib
import sys
import pytest

# ---------------------------------------------------------------------------
# Conditional sigmalang import — tests are split into sigmalang-required and
# sigmalang-optional so CI can run the fallback path without the library.
# ---------------------------------------------------------------------------
try:
    from sigmalang.core.primitives import (
        Glyph,
        GlyphStream,
        GlyphType,
        ExistentialPrimitive,
    )
    SIGMALANG_AVAILABLE = True
except ImportError:
    SIGMALANG_AVAILABLE = False

sigmalang_required = pytest.mark.skipif(
    not SIGMALANG_AVAILABLE, reason="sigmalang not installed"
)

from src.recycler.glyph_kv_cache import (
    TokenGlyphMapper,
    GlyphKVCache,
    HybridKVCache,
    _SIGMALANG_AVAILABLE,
)
from src.recycler.semantic_kv_cache import SemanticKVCache


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mapper():
    return TokenGlyphMapper()


@pytest.fixture
def glyph_cache():
    return GlyphKVCache(max_entries=64, chunk_size=4)


@pytest.fixture
def hybrid_cache():
    return HybridKVCache(max_entries=64, chunk_size=4)


def _make_kv(n_layers: int = 2):
    """Dummy KV tensors (lists of tuples, no torch dependency)."""
    return [([1.0] * 8, [2.0] * 8) for _ in range(n_layers)]


# ===========================================================================
# TokenGlyphMapper
# ===========================================================================

class TestTokenGlyphMapper:

    @sigmalang_required
    def test_tier0_tokens_map_to_existential(self, mapper):
        """Tokens 0-15 map to the 16 ExistentialPrimitives."""
        glyphs = mapper.tokens_to_glyphs(list(range(16)))
        assert len(glyphs) == 16
        for i, g in enumerate(glyphs):
            assert g.glyph_type == GlyphType.PRIMITIVE
            assert g.primitive_id == i

    @sigmalang_required
    def test_tier1_tokens_offset_correctly(self, mapper):
        """Tokens 16-127 offset into domain primitive range 0x10-0x7F."""
        glyphs = mapper.tokens_to_glyphs([16, 32, 127])
        assert glyphs[0].primitive_id == 0x10
        assert glyphs[1].primitive_id == 0x20
        assert glyphs[2].primitive_id == 0x7F

    @sigmalang_required
    def test_high_tokens_wrap_predictably(self, mapper):
        """Token IDs >= 256 wrap without raising errors."""
        glyphs = mapper.tokens_to_glyphs([256, 512, 1000, 50000])
        assert len(glyphs) == 4
        for g in glyphs:
            assert 0 <= g.primitive_id <= 0xFF

    @sigmalang_required
    def test_mapping_is_deterministic(self, mapper):
        """Same input always produces same glyph bytes."""
        tokens = [0, 10, 42, 100, 200, 500]
        b1 = mapper.tokens_to_bytes(tokens)
        b2 = mapper.tokens_to_bytes(tokens)
        assert b1 == b2

    @sigmalang_required
    def test_glyph_bytes_smaller_than_raw(self, mapper):
        """Glyph encoding should be more compact than int32 tokens."""
        # 64 tokens in tier-0/1 range → each glyph is 1 byte (no payload, no ext)
        tokens = list(range(64))
        glyph_bytes = mapper.tokens_to_bytes(tokens)
        raw_bytes = len(tokens) * 4
        # GlyphStream has 4-byte header + 2-byte CRC → 6 bytes overhead
        # Each tier-0/1 glyph = 1 byte inline → total = 6 + 64 = 70 bytes
        # raw = 256 bytes
        assert len(glyph_bytes) < raw_bytes, (
            f"Expected glyph bytes ({len(glyph_bytes)}) < raw bytes ({raw_bytes})"
        )

    @sigmalang_required
    def test_round_trip_glyph_stream(self, mapper):
        """GlyphStream.to_bytes() → from_bytes() preserves glyph count and types."""
        tokens = [0, 1, 15, 16, 100, 255, 300]
        glyphs = mapper.tokens_to_glyphs(tokens)
        stream = GlyphStream(glyphs=glyphs)
        serialized = stream.to_bytes()
        recovered = GlyphStream.from_bytes(serialized)
        assert len(recovered.glyphs) == len(glyphs)
        for orig, rec in zip(glyphs, recovered.glyphs):
            assert orig.glyph_type == rec.glyph_type
            assert orig.primitive_id == rec.primitive_id


# ===========================================================================
# GlyphKVCache
# ===========================================================================

class TestGlyphKVCache:

    def test_empty_cache_miss(self, glyph_cache):
        kv, length = glyph_cache.lookup([1, 2, 3, 4])
        assert kv == []
        assert length == 0

    def test_store_and_lookup_exact(self, glyph_cache):
        tokens = [1, 2, 3, 4]
        kv = _make_kv()
        glyph_cache.store(tokens, kv)

        result_kv, result_len = glyph_cache.lookup(tokens)
        assert result_len == 4
        assert result_kv == kv

    def test_sequence_too_short_not_cached(self, glyph_cache):
        """Sequences shorter than chunk_size (4) should not be stored."""
        glyph_cache.store([1, 2, 3], _make_kv())
        _, length = glyph_cache.lookup([1, 2, 3])
        assert length == 0

    def test_prefix_hit_on_longer_query(self, glyph_cache):
        """Storing 4 tokens → lookup of 8 tokens still returns the 4-token prefix."""
        tokens4 = [1, 2, 3, 4]
        tokens8 = [1, 2, 3, 4, 5, 6, 7, 8]
        kv = _make_kv()
        glyph_cache.store(tokens4, kv)

        result_kv, result_len = glyph_cache.lookup(tokens8)
        assert result_len == 4
        assert result_kv == kv

    def test_cache_hit_rate_tracking(self, glyph_cache):
        tokens = [0, 1, 2, 3]
        glyph_cache.store(tokens, _make_kv())
        glyph_cache.lookup(tokens)   # hit
        glyph_cache.lookup([9, 9, 9, 9])  # miss

        stats = glyph_cache.stats()
        assert stats["cache_hits"] == 1
        assert stats["total_lookups"] == 2
        assert stats["hit_rate"] == 0.5

    def test_lru_eviction(self):
        cache = GlyphKVCache(max_entries=2, chunk_size=4)
        cache.store([0, 1, 2, 3], _make_kv())
        cache.store([4, 5, 6, 7], _make_kv())
        cache.store([8, 9, 10, 11], _make_kv())  # should evict oldest

        stats = cache.stats()
        assert stats["entries"] <= 2

    def test_invalidate_all(self, glyph_cache):
        glyph_cache.store([0, 1, 2, 3], _make_kv())
        glyph_cache.invalidate()
        _, length = glyph_cache.lookup([0, 1, 2, 3])
        assert length == 0
        assert glyph_cache.stats()["entries"] == 0

    @sigmalang_required
    def test_glyph_bytes_saved_positive(self, glyph_cache):
        """After storing tier-0/1 tokens, glyph_bytes_saved should be > 0."""
        tokens = list(range(64))  # 64 tier-0/1 tokens
        glyph_cache.store(tokens, _make_kv())
        stats = glyph_cache.stats()
        assert stats["glyph_bytes_saved"] > 0

    @sigmalang_required
    def test_semantic_dedup_same_glyph_range(self):
        """
        Two token sequences that map to the same glyph chunk should share
        cache entries — the core semantic dedup property.

        Tokens 0 and 16 both produce distinct glyphs, so they won't collide,
        but two sequences that differ only in tokens >= 256 (which wrap to
        the same primitive_id) should.
        """
        cache = GlyphKVCache(max_entries=64, chunk_size=4)
        kv = _make_kv()

        # tokens 256 and 384 both wrap to primitive_id 0x80
        tokens_a = [0, 1, 2, 256]
        tokens_b = [0, 1, 2, 384]

        cache.store(tokens_a, kv)
        result_kv, result_len = cache.lookup(tokens_b)

        # If wrapping is correct, both hash the same → cache hit
        assert result_len == 4, (
            "Expected semantic dedup: tokens_b should hit tokens_a cache entry"
        )


# ===========================================================================
# HybridKVCache
# ===========================================================================

class TestHybridKVCache:

    def test_store_lookup_token_path(self, hybrid_cache):
        tokens = [1, 2, 3, 4]
        kv = _make_kv()
        hybrid_cache.store(tokens, kv)
        result_kv, result_len = hybrid_cache.lookup(tokens)
        assert result_len == 4

    def test_stats_structure(self, hybrid_cache):
        stats = hybrid_cache.stats()
        assert "token_cache" in stats
        assert "glyph_cache" in stats
        assert "hit_rate" in stats["token_cache"]
        assert "hit_rate" in stats["glyph_cache"]

    def test_invalidate_clears_both(self, hybrid_cache):
        hybrid_cache.store([0, 1, 2, 3], _make_kv())
        hybrid_cache.invalidate()
        _, tlen = hybrid_cache.token_cache.lookup([0, 1, 2, 3])
        _, glen = hybrid_cache.glyph_cache.lookup([0, 1, 2, 3])
        assert tlen == 0
        assert glen == 0

    def test_token_cache_hit_takes_priority(self, hybrid_cache):
        """Token-exact path should win when both caches have entries."""
        tokens = [5, 6, 7, 8]
        kv_token = _make_kv(n_layers=2)
        kv_glyph = _make_kv(n_layers=3)

        # Store different KV in each cache manually
        hybrid_cache.token_cache.store(tokens, kv_token)
        hybrid_cache.glyph_cache.store(tokens, kv_glyph)

        result_kv, _ = hybrid_cache.lookup(tokens)
        # Should return token cache result (first path)
        assert result_kv == kv_token


# ===========================================================================
# Compression ratio baseline
# ===========================================================================

class TestCompressionRatio:

    @sigmalang_required
    def test_tier0_compression_meets_target(self):
        """
        Pure tier-0 sequence: each glyph = 1 byte inline.
        GlyphStream overhead = 6 bytes (4 header + 2 CRC).
        16 tokens × 4 bytes = 64 bytes raw.
        16 glyphs × 1 byte + 6 overhead = 22 bytes.
        Ratio ≥ 2.5× (target from patent claims).
        """
        mapper = TokenGlyphMapper()
        tokens = list(range(16))
        glyph_bytes = mapper.tokens_to_bytes(tokens)
        raw_bytes = len(tokens) * 4

        ratio = raw_bytes / len(glyph_bytes)
        assert ratio >= 2.5, f"Compression ratio {ratio:.2f} below 2.5x target"

    @sigmalang_required
    def test_large_sequence_compression(self):
        """256-token sequence should maintain >2x compression."""
        mapper = TokenGlyphMapper()
        tokens = list(range(256))
        glyph_bytes = mapper.tokens_to_bytes(tokens)
        raw_bytes = len(tokens) * 4

        ratio = raw_bytes / len(glyph_bytes)
        assert ratio > 2.0, f"Compression ratio {ratio:.2f} below 2.0x"
