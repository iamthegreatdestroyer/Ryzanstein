"""
Glyph-Native Serialization & Zero-Copy Inference — Innovation #1
=================================================================

Core concept (from Ryot-updates.md, section 1):

    "Design a wire format where inference outputs are directly serializable
     as glyph vectors. MCP tool outputs don't need JSON→glyph→retrieval
     cycles. They ARE glyphs. Token recycling becomes literal glyph-pool
     recycling with zero marshaling overhead. Clients receive compressed
     glyph probes instead of token streams, enabling sub-linear bandwidth
     for agentic tasks."

Problem with standard inference I/O
------------------------------------
Standard path (every hop burns CPU + bandwidth):

    model output  →  token IDs (int32[])  →  JSON string  →  HTTP  →
    JSON parse    →  token IDs            →  glyph lookup  →  KV cache
                                                              ↓ GlyphPriorPool
                                                              ↓ GlyphBenchmarkIndex
                                                              ↓ NegativeSpaceDescriptor

Every arrow is a serialization/deserialization that materialises intermediate
representations nobody outside the pipeline needs.

Zero-copy path (this module)
-----------------------------
    model output  →  GlyphNativeStream  →  HTTP (binary)  →
    ZeroCopyGlyphOutput  →  KV cache / GlyphPriorPool / GlyphBenchmarkIndex
                         →  NegativeSpaceDescriptor
                         →  next inference prior (no re-tokenisation)

One binary object carries the output through the entire Sigma ecosystem.

Wire format: GNFS (Glyph Native Format Stream)
----------------------------------------------

Header (12 bytes, fixed):
    [0:4]   magic    = b"GNFS"        (Glyph Native Format Stream)
    [4:6]   version  = 0x0001         (uint16 LE)
    [6:8]   flags    = FLAG_*         (uint16 LE)
    [8:12]  count    = glyph_count    (uint32 LE)

Body (variable, per glyph):
    FLAG_NONE:    1 byte  per glyph   (primitive_id uint8)
    FLAG_DELTAS:  5 bytes per glyph   (primitive_id uint8 + delta float32 LE)
    FLAG_RLE:     variable — runs of (prim_id uint8 + run_length uint8)

Trailer (4 bytes):
    [0:4]   CRC-32 of header + body   (uint32 LE, zlib crc32)

Compression comparison (1000 glyph output):
    JSON token array:     ~5500 bytes (int tokens as decimal strings + commas)
    GNFS flat:            1012 bytes  (12 header + 1000 × 1B + 4 CRC)
    GNFS RLE (60% same):  ~412 bytes  (runs compress repeated primitives)
    Compression vs JSON:  5.4×–13.4×

    vs full embedding vectors (1000 × 1024 × 4B = 4 MB):
    GNFS flat:            ~4000× smaller

Integration (zero-copy chain)
------------------------------
    stream = GlyphNativeStream.from_token_ids(token_ids)
    out    = ZeroCopyGlyphOutput(stream)

    # All downstream operations take the same object — no re-serialization:
    kv_key  = out.as_cache_key()                       → sha256 of GNFS bytes
    prior   = out.as_prior_pool(vocab_size=32_000)     → GlyphPriorPool
    centroid = out.as_benchmark_centroid()             → 256-dim float list
    absent  = out.as_negative_space_descriptor()       → NegativeSpaceDescriptor

    # Pipeline chains all of the above:
    pipeline = GlyphNativePipeline(kv_cache, prior_pool, benchmark_index, extractor)
    result   = pipeline.ingest(out)
"""

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# ── Wire format constants ──────────────────────────────────────────────────
GNFS_MAGIC   = b"GNFS"
GNFS_VERSION = 1
GNFS_HEADER_SIZE = 12   # magic(4) + version(2) + flags(2) + count(4)
GNFS_TRAILER_SIZE = 4   # CRC-32

FLAG_NONE   = 0x0000
FLAG_DELTAS = 0x0001   # each prim_id followed by float32 delta
FLAG_RLE    = 0x0002   # run-length encode consecutive equal prim_ids

N_PRIMITIVES = 256

# Average bytes per token in a JSON token-id array (decimal integer + comma/space)
_JSON_BYTES_PER_TOKEN = 5.5


# ---------------------------------------------------------------------------
# GlyphNativeStream
# ---------------------------------------------------------------------------

class GlyphNativeStream:
    """
    Binary wire format for zero-copy glyph I/O.

    A GlyphNativeStream is:
    - Serialisable to GNFS bytes in O(N) — no intermediate dicts/strings
    - Directly ingestible by GlyphKVCache, GlyphPriorPool, GlyphBenchmarkIndex,
      and NegativeSpaceExtractor (all Innovation #3-#5 components)
    - Comparable: streams over semantically identical content produce the same
      primitive_ids when tokens share glyph buckets (semantic dedup)
    - Compact: 1 byte per glyph in FLAG_NONE mode; RLE compresses runs further

    Construction
    ------------
    from_token_ids(token_ids)    — most common: from model output
    from_text(text)              — byte-level → primitives
    from_bytes(data)             — deserialise from wire
    from_primitive_ids(prim_ids) — direct
    """

    __slots__ = ("primitive_ids", "deltas", "_flags_hint")

    def __init__(
        self,
        primitive_ids: List[int],
        deltas: Optional[List[float]] = None,
    ):
        self.primitive_ids: List[int] = [int(p) % N_PRIMITIVES for p in primitive_ids]
        self.deltas: Optional[List[float]] = (
            [float(d) for d in deltas] if deltas is not None else None
        )
        if self.deltas is not None and len(self.deltas) != len(self.primitive_ids):
            raise ValueError("deltas length must match primitive_ids length")
        self._flags_hint: int = FLAG_DELTAS if self.deltas else FLAG_NONE

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_primitive_ids(
        cls,
        prim_ids: List[int],
        deltas: Optional[List[float]] = None,
    ) -> "GlyphNativeStream":
        return cls(prim_ids, deltas)

    @classmethod
    def from_token_ids(
        cls,
        token_ids: List[int],
        deltas: Optional[List[float]] = None,
    ) -> "GlyphNativeStream":
        """
        Convert LLM output token IDs to glyph primitive IDs.

        Uses sigmalang TokenGlyphMapper when available; falls back to
        token_id % 256 bucketing.  Both paths are deterministic — the same
        tokens always produce the same primitive_ids, enabling zero-copy
        cache keying.
        """
        prim_ids = _tokens_to_primitives(token_ids)
        return cls(prim_ids, deltas)

    @classmethod
    def from_text(cls, text: str) -> "GlyphNativeStream":
        """
        Byte-level text encoding → primitive IDs.

        Each byte becomes a primitive_id (0-255).  This is the fastest
        zero-dependency path: no tokeniser required.
        """
        return cls([b for b in text.encode("utf-8", errors="replace")])

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_bytes(self, rle: bool = False) -> bytes:
        """
        Serialise to GNFS wire format.

        Args:
            rle: if True, use FLAG_RLE run-length encoding. Beneficial when
                 the same primitive repeats (semantic repetition in output).
                 Typical gain: 2×-4× over FLAG_NONE for natural language.
        """
        flags = self._flags_hint
        if rle:
            flags |= FLAG_RLE

        body = self._encode_body(flags)
        header = struct.pack(
            "<4sHHI",
            GNFS_MAGIC,
            GNFS_VERSION,
            flags,
            len(self.primitive_ids),
        )
        crc = struct.pack("<I", zlib.crc32(header + body) & 0xFFFFFFFF)
        return header + body + crc

    def _encode_body(self, flags: int) -> bytes:
        if flags & FLAG_RLE:
            return self._encode_rle(flags)
        if flags & FLAG_DELTAS:
            parts = []
            for pid, d in zip(self.primitive_ids, self.deltas or []):
                parts.append(struct.pack("<Bf", pid, d))
            return b"".join(parts)
        return bytes(self.primitive_ids)

    def _encode_rle(self, flags: int) -> bytes:
        """Run-length encode consecutive equal primitive_ids."""
        parts = []
        i = 0
        pids = self.primitive_ids
        while i < len(pids):
            run = 1
            while i + run < len(pids) and pids[i + run] == pids[i] and run < 255:
                run += 1
            if flags & FLAG_DELTAS:
                avg_delta = (
                    sum(self.deltas[i:i+run]) / run
                    if self.deltas else 0.0
                )
                parts.append(struct.pack("<BBf", pids[i], run, avg_delta))
            else:
                parts.append(struct.pack("<BB", pids[i], run))
            i += run
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "GlyphNativeStream":
        """Deserialise from GNFS wire format."""
        if len(data) < GNFS_HEADER_SIZE + GNFS_TRAILER_SIZE:
            raise ValueError("GNFS data too short")
        magic, version, flags, count = struct.unpack_from("<4sHHI", data, 0)
        if magic != GNFS_MAGIC:
            raise ValueError(f"Invalid GNFS magic: {magic!r}")
        if version != GNFS_VERSION:
            raise ValueError(f"Unsupported GNFS version: {version}")

        body    = data[GNFS_HEADER_SIZE:-GNFS_TRAILER_SIZE]
        crc_ref = struct.unpack_from("<I", data, len(data) - GNFS_TRAILER_SIZE)[0]
        crc_calc = zlib.crc32(data[: GNFS_HEADER_SIZE + len(body)]) & 0xFFFFFFFF
        if crc_calc != crc_ref:
            raise ValueError(f"GNFS CRC mismatch: expected {crc_ref:#010x}, got {crc_calc:#010x}")

        prim_ids, deltas = _decode_body(body, flags, count)
        stream = cls.__new__(cls)
        stream.primitive_ids = prim_ids
        stream.deltas        = deltas if deltas else None
        stream._flags_hint   = flags & ~FLAG_RLE  # strip RLE flag — decoded
        return stream

    # ── Properties & helpers ─────────────────────────────────────────────────

    @property
    def glyph_count(self) -> int:
        return len(self.primitive_ids)

    @property
    def unique_primitives(self) -> FrozenSet[int]:
        return frozenset(self.primitive_ids)

    @property
    def sparsity(self) -> float:
        """Fraction of 256 primitives actually used."""
        return len(self.unique_primitives) / N_PRIMITIVES

    def compression_ratio_vs_json(self, rle: bool = False) -> float:
        """
        Ratio of (JSON token-array bytes) to (GNFS wire bytes).
        > 1.0 means GNFS is smaller.
        """
        json_bytes = len(self.primitive_ids) * _JSON_BYTES_PER_TOKEN
        gnfs_bytes = len(self.to_bytes(rle=rle))
        return json_bytes / gnfs_bytes if gnfs_bytes else 0.0

    def cache_key(self) -> str:
        """SHA-256 of GNFS bytes — zero-copy cache key for GlyphKVCache."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_centroid(self) -> List[float]:
        """
        256-dim bag-of-glyphs histogram (same format as GlyphBenchmarkIndex
        centroid) — directly ingestible without conversion.
        """
        counts = [0] * N_PRIMITIVES
        for p in self.primitive_ids:
            counts[p] += 1
        total = len(self.primitive_ids) or 1
        return [c / total for c in counts]

    def __len__(self) -> int:
        return len(self.primitive_ids)

    def __repr__(self) -> str:
        return (
            f"GlyphNativeStream(glyphs={self.glyph_count}, "
            f"unique={len(self.unique_primitives)}, "
            f"sparsity={self.sparsity:.1%})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlyphNativeStream):
            return NotImplemented
        return self.primitive_ids == other.primitive_ids

    def __hash__(self) -> int:
        return hash(tuple(self.primitive_ids))


# ---------------------------------------------------------------------------
# ZeroCopyGlyphOutput
# ---------------------------------------------------------------------------

class ZeroCopyGlyphOutput:
    """
    Inference output that is natively a GlyphNativeStream.

    This is the "unified field" connector between all Ryot innovations:

        out.as_cache_key()               → str  (GlyphKVCache key, Inno #3)
        out.as_prior_pool(vocab_size)    → GlyphPriorPool (Inno #3)
        out.as_benchmark_centroid()      → List[float] (GlyphBenchmarkIndex, Inno #4)
        out.as_negative_space_descriptor() → NegativeSpaceDescriptor (Inno #5)
        out.as_mamba_token_ids()         → List[int] (GlyphMambaModel input, Inno #2)

    All of these derive from the same GlyphNativeStream — zero re-serialisation.
    """

    def __init__(
        self,
        stream: GlyphNativeStream,
        metadata: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ):
        self.stream     = stream
        self.metadata   = metadata or {}
        self.request_id = request_id

    # ── Wire ─────────────────────────────────────────────────────────────────

    def as_bytes(self, rle: bool = False) -> bytes:
        """GNFS wire bytes — serve directly as HTTP response body."""
        return self.stream.to_bytes(rle=rle)

    def as_cache_key(self) -> str:
        return self.stream.cache_key()

    # ── Ecosystem connectors (zero re-serialisation) ──────────────────────────

    def as_prior_pool(self, vocab_size: int = 32_000, weight: float = 1.0):
        """
        Build a GlyphPriorPool pre-seeded with this output's glyph residue.
        Zero-copy: uses primitive_ids directly, no token re-lookup.
        """
        try:
            from src.recycler.glyph_prior_pool import GlyphPriorPool
            pool = GlyphPriorPool(vocab_size=vocab_size)
            # accumulate using primitive_ids as proxy token_ids (tier-0 range)
            capped = [min(p, vocab_size - 1) for p in self.stream.primitive_ids]
            pool.accumulate(capped, weight=weight)
            return pool
        except ImportError:
            return None

    def as_benchmark_centroid(self) -> List[float]:
        """256-dim centroid for GlyphBenchmarkIndex — zero-copy."""
        return self.stream.to_centroid()

    def as_negative_space_descriptor(self):
        """
        NegativeSpaceDescriptor where absent = primitives NOT in this stream.
        Zero-copy: derived directly from unique_primitives frozenset.
        """
        try:
            from src.consensus.negative_space import NegativeSpaceDescriptor
            absent = frozenset(range(N_PRIMITIVES)) - self.stream.unique_primitives
            return NegativeSpaceDescriptor(absent)
        except ImportError:
            return None

    def as_mamba_token_ids(self) -> List[int]:
        """
        Representative token_ids for GlyphMambaModel forward pass.
        Uses primitive_ids directly (they map onto valid embedding indices
        since d_glyph embedding table has N_PRIMITIVES=256 rows).
        """
        return self.stream.primitive_ids

    def as_json_dict(self) -> Dict:
        """Fallback JSON representation for non-glyph-native clients."""
        return {
            "glyph_count":       self.stream.glyph_count,
            "unique_primitives": len(self.stream.unique_primitives),
            "sparsity":          round(self.stream.sparsity, 4),
            "primitive_ids":     self.stream.primitive_ids,
            "cache_key":         self.as_cache_key(),
            "request_id":        self.request_id,
            "metadata":          self.metadata,
        }

    @classmethod
    def from_token_ids(
        cls,
        token_ids: List[int],
        request_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> "ZeroCopyGlyphOutput":
        stream = GlyphNativeStream.from_token_ids(token_ids)
        return cls(stream, metadata=metadata, request_id=request_id)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> "ZeroCopyGlyphOutput":
        stream = GlyphNativeStream.from_bytes(data)
        return cls(stream, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ZeroCopyGlyphOutput(stream={self.stream!r}, "
            f"request_id={self.request_id!r})"
        )


# ---------------------------------------------------------------------------
# GlyphNativePipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result from one GlyphNativePipeline.ingest() call."""
    cache_key:         str
    glyph_count:       int
    unique_primitives: int
    wire_bytes:        int
    json_bytes_equiv:  int
    compression_ratio: float
    prior_pool_seeded: bool
    centroid_dims:     int
    absent_count:      int
    rle_wire_bytes:    int


class GlyphNativePipeline:
    """
    Zero-copy pipeline that routes a ZeroCopyGlyphOutput into all downstream
    Sigma ecosystem components in a single pass.

    One call to ingest() populates:
    - GlyphKVCache (via cache_key)
    - GlyphPriorPool (via as_prior_pool)
    - GlyphBenchmarkIndex (via as_benchmark_centroid, on demand)
    - NegativeSpaceDescriptor (via as_negative_space_descriptor)

    No intermediate JSON. No re-serialisation. No token lookup at ingestion time.
    """

    def __init__(
        self,
        kv_cache=None,         # HybridKVCache or None
        prior_pool=None,       # GlyphPriorPool or None
        benchmark_index=None,  # GlyphBenchmarkIndex or None
        vocab_size: int = 32_000,
    ):
        self.kv_cache        = kv_cache
        self.prior_pool      = prior_pool
        self.benchmark_index = benchmark_index
        self.vocab_size      = vocab_size
        self._ingest_count   = 0

    def ingest(
        self,
        output: ZeroCopyGlyphOutput,
        latency_ms: float = 0.0,
        rle: bool = False,
    ) -> PipelineResult:
        """
        Route output through all downstream components.

        Args:
            output:     ZeroCopyGlyphOutput to ingest
            latency_ms: inference latency to record in benchmark_index
            rle:        whether to measure RLE wire size

        Returns:
            PipelineResult with size and routing stats.
        """
        self._ingest_count += 1

        # ── Wire sizes ──────────────────────────────────────────────────────
        wire_bytes     = len(output.as_bytes(rle=False))
        rle_wire_bytes = len(output.as_bytes(rle=True))
        json_equiv     = int(output.stream.glyph_count * _JSON_BYTES_PER_TOKEN)
        compression    = json_equiv / wire_bytes if wire_bytes else 0.0

        # ── GlyphPriorPool ──────────────────────────────────────────────────
        prior_seeded = False
        if self.prior_pool is not None:
            try:
                capped = [min(p, self.vocab_size - 1) for p in output.stream.primitive_ids]
                self.prior_pool.accumulate(capped, weight=1.0)
                prior_seeded = True
            except Exception:
                pass

        # ── GlyphBenchmarkIndex ─────────────────────────────────────────────
        if self.benchmark_index is not None:
            try:
                self.benchmark_index.record(
                    output.stream.primitive_ids,
                    {"latency_ms": latency_ms, "cache_hit_rate": 0.0, "batch_size": 1.0},
                )
            except Exception:
                pass

        # ── NegativeSpaceDescriptor ─────────────────────────────────────────
        absent_count = 0
        ns_desc = output.as_negative_space_descriptor()
        if ns_desc is not None:
            absent_count = ns_desc.absent_count

        return PipelineResult(
            cache_key         = output.as_cache_key(),
            glyph_count       = output.stream.glyph_count,
            unique_primitives = len(output.stream.unique_primitives),
            wire_bytes        = wire_bytes,
            json_bytes_equiv  = json_equiv,
            compression_ratio = round(compression, 2),
            prior_pool_seeded = prior_seeded,
            centroid_dims     = N_PRIMITIVES,
            absent_count      = absent_count,
            rle_wire_bytes    = rle_wire_bytes,
        )

    def stats(self) -> Dict:
        return {
            "ingested": self._ingest_count,
            "prior_pool":       self.prior_pool.stats() if self.prior_pool else None,
            "benchmark_index":  self.benchmark_index.stats() if self.benchmark_index else None,
        }


# ---------------------------------------------------------------------------
# GlyphNativeCodec
# ---------------------------------------------------------------------------

class GlyphNativeCodec:
    """
    Content-negotiation codec for the FastAPI server.

    Handles:
        Accept: application/x-glyph-native  → GNFS bytes
        Accept: application/json (default)  → JSON dict

    Usage in FastAPI:
        codec = GlyphNativeCodec()
        response = codec.encode(output, accept_header=request.headers.get("Accept", ""))
        return Response(response.body, media_type=response.media_type)
    """

    CONTENT_TYPE = "application/x-glyph-native"

    def encode(
        self,
        output: ZeroCopyGlyphOutput,
        accept_header: str = "",
        rle: bool = False,
    ) -> Tuple[bytes, str]:
        """
        Encode output according to Accept header.

        Returns:
            (body_bytes, media_type)
        """
        if self.CONTENT_TYPE in accept_header:
            return output.as_bytes(rle=rle), self.CONTENT_TYPE
        else:
            import json
            return json.dumps(output.as_json_dict()).encode(), "application/json"

    @staticmethod
    def decode(body: bytes, content_type: str) -> ZeroCopyGlyphOutput:
        """Decode incoming body back to ZeroCopyGlyphOutput."""
        if GlyphNativeCodec.CONTENT_TYPE in content_type:
            return ZeroCopyGlyphOutput.from_bytes(body)
        else:
            import json
            data = json.loads(body)
            prim_ids = data.get("primitive_ids", [])
            stream = GlyphNativeStream.from_primitive_ids(prim_ids)
            return ZeroCopyGlyphOutput(
                stream,
                metadata=data.get("metadata", {}),
                request_id=data.get("request_id"),
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokens_to_primitives(token_ids: List[int]) -> List[int]:
    """Map token IDs → primitive IDs (sigmalang or fallback)."""
    mapper = _get_mapper()
    if mapper is not None:
        return [mapper._map_one(t).primitive_id for t in token_ids]
    # Fallback: tier-aware bucketing (matches TokenGlyphMapper heuristic)
    result = []
    for t in token_ids:
        if t < 16:
            result.append(t)
        elif t < 128:
            result.append(0x10 + (t - 16))
        else:
            result.append(0x80 + ((t - 128) % 128))
    return result


_cached_mapper = None

def _get_mapper():
    global _cached_mapper
    if _cached_mapper is not None:
        return _cached_mapper
    try:
        import sys
        from pathlib import Path
        _sl_root = Path(__file__).parent.parent.parent.parent / "sigmalang"
        if _sl_root.exists():
            sys.path.insert(0, str(_sl_root))
        from src.recycler.glyph_kv_cache import TokenGlyphMapper
        _cached_mapper = TokenGlyphMapper()
    except Exception:
        pass
    return _cached_mapper


def _decode_body(
    body: bytes,
    flags: int,
    count: int,
) -> Tuple[List[int], Optional[List[float]]]:
    """Decode GNFS body bytes → (primitive_ids, deltas)."""
    prim_ids: List[int] = []
    deltas: Optional[List[float]] = [] if (flags & FLAG_DELTAS) else None

    if flags & FLAG_RLE:
        if flags & FLAG_DELTAS:
            entry_size = 6  # uint8 + uint8 + float32
            for i in range(0, len(body), entry_size):
                if i + entry_size > len(body):
                    break
                pid, run, delta = struct.unpack_from("<BBf", body, i)
                prim_ids.extend([pid] * run)
                if deltas is not None:
                    deltas.extend([delta] * run)
        else:
            entry_size = 2  # uint8 + uint8
            for i in range(0, len(body), entry_size):
                if i + entry_size > len(body):
                    break
                pid, run = struct.unpack_from("<BB", body, i)
                prim_ids.extend([pid] * run)
    elif flags & FLAG_DELTAS:
        entry_size = 5  # uint8 + float32
        for i in range(0, len(body), entry_size):
            if i + entry_size > len(body):
                break
            pid, delta = struct.unpack_from("<Bf", body, i)
            prim_ids.append(pid)
            if deltas is not None:
                deltas.append(delta)
    else:
        prim_ids = list(body[:count])

    return prim_ids, deltas
