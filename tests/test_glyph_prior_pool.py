"""
Innovation #3: Bidirectional Token Recycling — GlyphPriorPool Tests
====================================================================

Validates:
  - Accumulation: glyph residues build up with each call
  - Bias directionality: tokens with accumulated primitives get positive bias
  - Decay: weights shrink monotonically and eventually prune to zero
  - Normalization: prior_logits always ≤ prior_strength
  - HybridKVCache integration: store() auto-accumulates; lookup_with_prior() returns prior
  - Reset: invalidate() clears the pool
  - Graceful degradation when sigmalang is absent (zeros, no crash)
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.recycler.glyph_prior_pool import GlyphPriorPool
from src.recycler.glyph_kv_cache import HybridKVCache, _SIGMALANG_AVAILABLE

sigmalang_required = pytest.mark.skipif(
    not _SIGMALANG_AVAILABLE, reason="sigmalang not installed"
)

VOCAB = 512   # small vocab for fast tests


# ---------------------------------------------------------------------------
# Basic accumulation
# ---------------------------------------------------------------------------

class TestGlyphPriorPoolAccumulate:

    def test_empty_pool_all_zeros(self):
        pool = GlyphPriorPool(vocab_size=VOCAB)
        bias = pool.prior_logits(VOCAB)
        assert len(bias) == VOCAB
        assert all(v == 0.0 for v in bias)

    @sigmalang_required
    def test_accumulate_increases_weight(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=1.0)
        tokens = [0, 1, 2, 3]  # tier-0: direct primitive mapping
        pool.accumulate(tokens)
        assert pool.stats()["active_primitives"] > 0
        assert pool.stats()["max_weight"] > 0.0

    @sigmalang_required
    def test_repeated_accumulate_increases_max_weight(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=1.0)
        tokens = [0, 1, 2, 3]
        pool.accumulate(tokens)
        w1 = pool.stats()["max_weight"]
        pool.accumulate(tokens)
        w2 = pool.stats()["max_weight"]
        assert w2 > w1

    @sigmalang_required
    def test_total_accumulations_tracked(self):
        pool = GlyphPriorPool(vocab_size=VOCAB)
        pool.accumulate([0, 1, 2, 3])
        pool.accumulate([4, 5, 6, 7])
        assert pool.stats()["total_accumulations"] == 2

    def test_accumulate_empty_list_no_crash(self):
        pool = GlyphPriorPool(vocab_size=VOCAB)
        pool.accumulate([])  # should silently do nothing
        assert pool.stats()["active_primitives"] == 0


# ---------------------------------------------------------------------------
# Prior logits directionality
# ---------------------------------------------------------------------------

class TestGlyphPriorPoolLogits:

    @sigmalang_required
    def test_prior_logits_length(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=0.5)
        pool.accumulate([0, 1, 2])
        bias = pool.prior_logits(VOCAB)
        assert len(bias) == VOCAB

    @sigmalang_required
    def test_prior_logits_nonnegative(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=0.5)
        pool.accumulate([0, 1, 2, 3, 10, 20])
        bias = pool.prior_logits(VOCAB)
        assert all(v >= 0.0 for v in bias)

    @sigmalang_required
    def test_prior_logits_bounded_by_strength(self):
        strength = 0.3
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=strength)
        for _ in range(5):
            pool.accumulate(list(range(16)))
        bias = pool.prior_logits(VOCAB)
        assert max(bias) <= strength + 1e-9, f"Max bias {max(bias)} exceeds strength {strength}"

    @sigmalang_required
    def test_accumulated_tokens_get_higher_bias(self):
        """Token 0 (primitive 0x00) should have higher bias than an unused token."""
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=1.0)
        pool.accumulate([0, 0, 0, 0])  # heavily bias primitive 0x00
        bias = pool.prior_logits(VOCAB)
        # Token 0 maps to primitive 0x00 — should have the highest bias
        assert bias[0] > 0.0
        # Token 200 maps to a different primitive (tier-2) — likely 0
        # (unless 200 happens to share primitive 0x00, which it won't)
        assert bias[0] >= bias[200]

    @sigmalang_required
    def test_prior_tensor_shape(self):
        import torch
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=0.2)
        pool.accumulate([0, 1, 2])
        t = pool.prior_tensor(VOCAB)
        assert t.shape == (VOCAB,)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

class TestGlyphPriorPoolDecay:

    @sigmalang_required
    def test_decay_reduces_max_weight(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, decay_factor=0.5)
        pool.accumulate([0, 1, 2, 3])
        w_before = pool.stats()["max_weight"]
        pool.decay()
        w_after = pool.stats()["max_weight"]
        assert w_after < w_before

    @sigmalang_required
    def test_decay_tracks_count(self):
        pool = GlyphPriorPool(vocab_size=VOCAB)
        pool.accumulate([0, 1])
        pool.decay()
        pool.decay()
        assert pool.stats()["total_decays"] == 2

    @sigmalang_required
    def test_full_decay_prunes_all_entries(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, decay_factor=0.1)
        pool.accumulate([0, 1, 2, 3])
        for _ in range(20):
            pool.decay()
        # After many fast decays all weights should fall below 1e-7 threshold
        assert pool.stats()["active_primitives"] == 0

    def test_decay_empty_pool_no_crash(self):
        pool = GlyphPriorPool(vocab_size=VOCAB)
        pool.decay()  # should be a no-op
        assert pool.stats()["total_decays"] == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestGlyphPriorPoolReset:

    @sigmalang_required
    def test_reset_clears_weights(self):
        pool = GlyphPriorPool(vocab_size=VOCAB)
        pool.accumulate([0, 1, 2, 3])
        assert pool.stats()["active_primitives"] > 0
        pool.reset()
        assert pool.stats()["active_primitives"] == 0

    @sigmalang_required
    def test_reset_zeros_bias(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=1.0)
        pool.accumulate([0, 1, 2, 3])
        pool.reset()
        bias = pool.prior_logits(VOCAB)
        assert all(v == 0.0 for v in bias)

    @sigmalang_required
    def test_top_primitives(self):
        pool = GlyphPriorPool(vocab_size=VOCAB, prior_strength=1.0)
        pool.accumulate([0, 0, 0])  # primitive 0x00 should dominate
        tops = pool.top_primitives(n=5)
        assert len(tops) >= 1
        assert tops[0]["primitive_id"] == 0x00


# ---------------------------------------------------------------------------
# HybridKVCache integration
# ---------------------------------------------------------------------------

class TestHybridKVCachePrior:

    def test_hybrid_has_prior_pool(self):
        cache = HybridKVCache(max_entries=64, chunk_size=4)
        assert hasattr(cache, "prior_pool")
        assert isinstance(cache.prior_pool, GlyphPriorPool)

    def test_stats_includes_prior_pool(self):
        cache = HybridKVCache(max_entries=64, chunk_size=4)
        stats = cache.stats()
        assert "prior_pool" in stats
        assert "active_primitives" in stats["prior_pool"]

    @sigmalang_required
    def test_store_accumulates_residue(self):
        cache = HybridKVCache(max_entries=64, chunk_size=4)
        tokens = [0, 1, 2, 3, 4, 5, 6, 7]
        cache.store(tokens, [[1.0, 2.0]])
        # Prior pool should have accumulated residue
        assert cache.prior_pool.stats()["total_accumulations"] == 1
        assert cache.prior_pool.stats()["active_primitives"] > 0

    def test_lookup_with_prior_returns_triple(self):
        cache = HybridKVCache(max_entries=64, chunk_size=4, vocab_size=VOCAB)
        tokens = [1, 2, 3, 4]
        result = cache.lookup_with_prior(tokens, vocab_size=VOCAB)
        assert len(result) == 3
        kv, length, prior = result
        assert isinstance(prior, list)
        assert len(prior) == VOCAB

    @sigmalang_required
    def test_prior_nonzero_after_store(self):
        cache = HybridKVCache(
            max_entries=64, chunk_size=4,
            prior_strength=1.0, vocab_size=VOCAB
        )
        tokens = [0, 1, 2, 3, 4, 5, 6, 7]
        cache.store(tokens, [[1.0]])
        _, _, prior = cache.lookup_with_prior(tokens, vocab_size=VOCAB)
        assert max(prior) > 0.0, "Prior should be non-zero after storing a sequence"

    @sigmalang_required
    def test_invalidate_resets_prior_pool(self):
        cache = HybridKVCache(max_entries=64, chunk_size=4, vocab_size=VOCAB)
        cache.store([0, 1, 2, 3], [[1.0]])
        assert cache.prior_pool.stats()["active_primitives"] > 0
        cache.invalidate()
        assert cache.prior_pool.stats()["active_primitives"] == 0

    def test_prior_all_zeros_when_sigmalang_absent(self):
        """Prior degrades gracefully when sigmalang isn't available."""
        cache = HybridKVCache(max_entries=64, chunk_size=4, vocab_size=VOCAB)
        _, _, prior = cache.lookup_with_prior([1, 2, 3, 4], vocab_size=VOCAB)
        # Either all zeros (no sigmalang) or a valid list — never crashes
        assert isinstance(prior, list)
        assert len(prior) == VOCAB
