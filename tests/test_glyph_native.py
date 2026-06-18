"""
Innovation #1: Glyph-Native Serialization — Unit Tests
======================================================

Validates:
  - GlyphNativeStream: from_token_ids, from_text, from_primitive_ids
  - Wire format round-trips: FLAG_NONE, FLAG_DELTAS, FLAG_RLE
  - Compression ratios vs JSON baseline
  - ZeroCopyGlyphOutput: all ecosystem connectors (prior_pool, centroid,
    negative_space_descriptor, mamba_token_ids, cache_key)
  - GlyphNativePipeline: single-pass ingestion, stats
  - GlyphNativeCodec: Accept header negotiation, encode/decode
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.serialization.glyph_native import (
    FLAG_DELTAS,
    FLAG_NONE,
    FLAG_RLE,
    GNFS_MAGIC,
    GlyphNativeCodec,
    GlyphNativePipeline,
    GlyphNativeStream,
    ZeroCopyGlyphOutput,
)

torch.manual_seed(0)


# ============================================================================
# GlyphNativeStream — construction
# ============================================================================

class TestGlyphNativeStreamConstruction:

    def test_from_primitive_ids(self):
        s = GlyphNativeStream.from_primitive_ids([0, 1, 2, 3])
        assert s.primitive_ids == [0, 1, 2, 3]
        assert s.glyph_count == 4

    def test_from_token_ids(self):
        s = GlyphNativeStream.from_token_ids([0, 1, 2, 3])
        assert s.glyph_count == 4
        # Tier-0 tokens 0-15: prim_id == token_id
        assert s.primitive_ids[0] == 0
        assert s.primitive_ids[1] == 1

    def test_from_text(self):
        s = GlyphNativeStream.from_text("hi")
        assert s.glyph_count == 2
        # "h"=104, "i"=105
        assert s.primitive_ids == [104, 105]

    def test_primitive_ids_clamped_to_255(self):
        s = GlyphNativeStream([256, 512])
        assert all(0 <= p <= 255 for p in s.primitive_ids)

    def test_with_deltas(self):
        s = GlyphNativeStream([0, 1, 2], deltas=[0.1, 0.2, 0.3])
        assert s.deltas is not None
        assert len(s.deltas) == 3

    def test_delta_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            GlyphNativeStream([0, 1], deltas=[0.1])

    def test_unique_primitives(self):
        s = GlyphNativeStream([0, 0, 1, 1, 2])
        assert s.unique_primitives == frozenset([0, 1, 2])

    def test_sparsity(self):
        s = GlyphNativeStream(list(range(128)))
        assert s.sparsity == 0.5

    def test_empty_stream(self):
        s = GlyphNativeStream([])
        assert s.glyph_count == 0
        assert s.sparsity == 0.0


# ============================================================================
# GlyphNativeStream — serialisation
# ============================================================================

class TestGlyphNativeStreamSerialization:

    def test_magic_header(self):
        s = GlyphNativeStream([0, 1, 2])
        data = s.to_bytes()
        assert data[:4] == GNFS_MAGIC

    def test_flag_none_round_trip(self):
        original = [0, 1, 2, 100, 200, 255]
        s = GlyphNativeStream(original)
        data = s.to_bytes()
        s2 = GlyphNativeStream.from_bytes(data)
        assert s2.primitive_ids == original

    def test_flag_deltas_round_trip(self):
        pids   = [0, 1, 2, 3]
        deltas = [0.1, 0.2, 0.3, 0.4]
        s = GlyphNativeStream(pids, deltas=deltas)
        data = s.to_bytes()
        s2 = GlyphNativeStream.from_bytes(data)
        assert s2.primitive_ids == pids
        assert s2.deltas is not None
        for a, b in zip(s2.deltas, deltas):
            assert abs(a - b) < 1e-5

    def test_flag_rle_round_trip(self):
        pids = [5, 5, 5, 10, 10, 20]
        s = GlyphNativeStream(pids)
        data_rle = s.to_bytes(rle=True)
        s2 = GlyphNativeStream.from_bytes(data_rle)
        assert s2.primitive_ids == pids

    def test_rle_smaller_than_flat_for_repeated_primitives(self):
        """Repeated primitives → RLE should compress."""
        pids = [42] * 100
        s = GlyphNativeStream(pids)
        flat = s.to_bytes(rle=False)
        rle  = s.to_bytes(rle=True)
        assert len(rle) < len(flat), "RLE should compress 100 identical glyphs"

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="GNFS"):
            GlyphNativeStream.from_bytes(b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00CRCX")

    def test_crc_corruption_raises(self):
        s = GlyphNativeStream([1, 2, 3])
        data = bytearray(s.to_bytes())
        data[-1] ^= 0xFF  # corrupt last CRC byte
        with pytest.raises(ValueError, match="CRC"):
            GlyphNativeStream.from_bytes(bytes(data))

    def test_empty_stream_round_trip(self):
        s = GlyphNativeStream([])
        s2 = GlyphNativeStream.from_bytes(s.to_bytes())
        assert s2.primitive_ids == []

    def test_compression_ratio_vs_json(self):
        """GNFS should be smaller than equivalent JSON for medium sequences."""
        s = GlyphNativeStream(list(range(256)) * 2)  # 512 glyphs, 256 unique
        ratio = s.compression_ratio_vs_json(rle=False)
        assert ratio > 1.0, f"Expected GNFS < JSON, got ratio {ratio:.2f}"

    def test_rle_compression_ratio_better_than_flat(self):
        """RLE should beat flat for repetitive streams."""
        s = GlyphNativeStream([0] * 100 + [1] * 100)
        flat_ratio = s.compression_ratio_vs_json(rle=False)
        rle_ratio  = s.compression_ratio_vs_json(rle=True)
        assert rle_ratio >= flat_ratio

    def test_cache_key_deterministic(self):
        s1 = GlyphNativeStream([0, 1, 2])
        s2 = GlyphNativeStream([0, 1, 2])
        assert s1.cache_key() == s2.cache_key()

    def test_different_streams_different_cache_key(self):
        s1 = GlyphNativeStream([0, 1, 2])
        s2 = GlyphNativeStream([3, 4, 5])
        assert s1.cache_key() != s2.cache_key()

    def test_centroid_sums_to_one(self):
        s = GlyphNativeStream([0, 1, 2, 3])
        c = s.to_centroid()
        assert len(c) == 256
        assert abs(sum(c) - 1.0) < 1e-9

    def test_centroid_nonzero_at_present_prims(self):
        s = GlyphNativeStream([5, 10, 15])
        c = s.to_centroid()
        assert c[5]  > 0
        assert c[10] > 0
        assert c[15] > 0
        assert c[0]  == 0


# ============================================================================
# ZeroCopyGlyphOutput — ecosystem connectors
# ============================================================================

class TestZeroCopyGlyphOutput:

    def _make_output(self, token_ids=None) -> ZeroCopyGlyphOutput:
        token_ids = token_ids or [0, 1, 2, 3, 4, 5, 6, 7]
        return ZeroCopyGlyphOutput.from_token_ids(token_ids, request_id="test-001")

    def test_from_token_ids(self):
        out = self._make_output()
        assert out.stream.glyph_count == 8
        assert out.request_id == "test-001"

    def test_from_bytes(self):
        out = self._make_output()
        data = out.as_bytes()
        out2 = ZeroCopyGlyphOutput.from_bytes(data)
        assert out2.stream.primitive_ids == out.stream.primitive_ids

    def test_cache_key_consistent(self):
        out = self._make_output([0, 1, 2])
        assert out.as_cache_key() == out.as_cache_key()

    def test_as_negative_space_descriptor(self):
        out = self._make_output([0, 1, 2])
        ns = out.as_negative_space_descriptor()
        if ns is None:
            pytest.skip("consensus module not importable")
        # Only primitives 0, 1, 2 are present → 253 are absent
        assert len(ns.absent) == 256 - len(out.stream.unique_primitives)

    def test_as_prior_pool(self):
        out = self._make_output([0, 1, 2])
        pool = out.as_prior_pool(vocab_size=256)
        if pool is None:
            pytest.skip("recycler module not importable")
        assert pool.stats()["total_accumulations"] > 0

    def test_as_benchmark_centroid(self):
        out = self._make_output([0, 1, 2])
        centroid = out.as_benchmark_centroid()
        assert len(centroid) == 256
        assert abs(sum(centroid) - 1.0) < 1e-9

    def test_as_mamba_token_ids(self):
        out = self._make_output([0, 1, 2])
        ids = out.as_mamba_token_ids()
        assert isinstance(ids, list)
        assert len(ids) == 3

    def test_as_json_dict_structure(self):
        out = self._make_output()
        d = out.as_json_dict()
        assert "glyph_count" in d
        assert "primitive_ids" in d
        assert "cache_key" in d
        assert d["request_id"] == "test-001"

    def test_as_json_dict_round_trips_via_codec(self):
        out = self._make_output([10, 20, 30])
        codec = GlyphNativeCodec()
        # JSON path
        body, ctype = codec.encode(out, accept_header="application/json")
        out2 = GlyphNativeCodec.decode(body, ctype)
        assert out2.stream.primitive_ids == out.stream.primitive_ids

    def test_as_bytes_binary_round_trip(self):
        out = self._make_output([10, 20, 30])
        codec = GlyphNativeCodec()
        body, ctype = codec.encode(out, accept_header="application/x-glyph-native")
        assert ctype == GlyphNativeCodec.CONTENT_TYPE
        out2 = GlyphNativeCodec.decode(body, ctype)
        assert out2.stream.primitive_ids == out.stream.primitive_ids


# ============================================================================
# GlyphNativePipeline
# ============================================================================

class TestGlyphNativePipeline:

    def test_ingest_returns_result(self):
        pipe = GlyphNativePipeline()
        out  = ZeroCopyGlyphOutput.from_token_ids([0, 1, 2, 3, 4])
        r = pipe.ingest(out)
        assert r.glyph_count == 5
        assert r.wire_bytes > 0
        assert r.compression_ratio > 0

    def test_pipeline_with_prior_pool(self):
        from src.recycler.glyph_prior_pool import GlyphPriorPool
        pool = GlyphPriorPool(vocab_size=256)
        pipe = GlyphNativePipeline(prior_pool=pool)
        out  = ZeroCopyGlyphOutput.from_token_ids([0, 1, 2])
        r = pipe.ingest(out)
        assert r.prior_pool_seeded

    def test_pipeline_with_benchmark_index(self):
        from src.recycler.glyph_benchmark_index import GlyphBenchmarkIndex
        idx  = GlyphBenchmarkIndex()
        pipe = GlyphNativePipeline(benchmark_index=idx)
        out  = ZeroCopyGlyphOutput.from_token_ids([0, 1, 2])
        pipe.ingest(out, latency_ms=5.0)
        assert idx.stats()["entries"] == 1

    def test_pipeline_absent_count(self):
        pipe = GlyphNativePipeline()
        out  = ZeroCopyGlyphOutput.from_token_ids([0, 1, 2])  # 3 unique prims
        r = pipe.ingest(out)
        # 256 - 3 unique prims = 253 absent (if NS module available)
        if r.absent_count > 0:
            assert r.absent_count == 256 - len(out.stream.unique_primitives)

    def test_ingest_count_increments(self):
        pipe = GlyphNativePipeline()
        out  = ZeroCopyGlyphOutput.from_token_ids([0, 1, 2])
        for _ in range(5):
            pipe.ingest(out)
        assert pipe._ingest_count == 5

    def test_rle_smaller_than_flat_in_result(self):
        pipe = GlyphNativePipeline()
        out  = ZeroCopyGlyphOutput(GlyphNativeStream([7] * 200))
        r    = pipe.ingest(out)
        assert r.rle_wire_bytes <= r.wire_bytes


# ============================================================================
# GlyphNativeCodec
# ============================================================================

class TestGlyphNativeCodec:

    def test_encode_glyph_native_accept(self):
        codec = GlyphNativeCodec()
        out   = ZeroCopyGlyphOutput.from_token_ids([0, 1, 2])
        body, ctype = codec.encode(out, accept_header="application/x-glyph-native")
        assert ctype == "application/x-glyph-native"
        assert body[:4] == GNFS_MAGIC

    def test_encode_json_default(self):
        codec = GlyphNativeCodec()
        out   = ZeroCopyGlyphOutput.from_token_ids([0, 1, 2])
        body, ctype = codec.encode(out, accept_header="")
        assert ctype == "application/json"
        d = json.loads(body)
        assert "glyph_count" in d

    def test_decode_glyph_native(self):
        codec = GlyphNativeCodec()
        out   = ZeroCopyGlyphOutput.from_token_ids([5, 10, 15])
        body, ctype = codec.encode(out, accept_header="application/x-glyph-native")
        out2 = codec.decode(body, ctype)
        assert out2.stream.primitive_ids == out.stream.primitive_ids

    def test_decode_json(self):
        codec = GlyphNativeCodec()
        out   = ZeroCopyGlyphOutput.from_token_ids([5, 10, 15])
        body, ctype = codec.encode(out, accept_header="application/json")
        out2 = codec.decode(body, ctype)
        assert out2.stream.primitive_ids == out.stream.primitive_ids
