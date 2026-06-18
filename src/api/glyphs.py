"""
POST /v1/glyphs  — Token → Glyph encoding endpoint.
POST /v1/glyphs/decode — Glyph bytes → token IDs (round-trip verification).
GET  /v1/glyphs/stats  — Cache and encoder statistics.

Mount this router in the main Ryot FastAPI app:

    from src.api.glyphs import router as glyphs_router
    app.include_router(glyphs_router)
"""

import hashlib
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from sigmalang.core.primitives import (
        Glyph,
        GlyphStream,
        GlyphType,
    )
    from ..recycler.glyph_kv_cache import TokenGlyphMapper
    _SIGMALANG_AVAILABLE = True
except ImportError:
    _SIGMALANG_AVAILABLE = False

router = APIRouter(prefix="/v1/glyphs", tags=["glyphs"])

# Module-level mapper instance (stateless, safe to share)
_mapper = TokenGlyphMapper() if _SIGMALANG_AVAILABLE else None

# Simple in-memory stats counters (reset on restart)
_stats = {
    "encode_calls": 0,
    "decode_calls": 0,
    "total_tokens_encoded": 0,
    "total_glyphs_produced": 0,
    "total_bytes_raw": 0,
    "total_bytes_glyph": 0,
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EncodeRequest(BaseModel):
    tokens: List[int] = Field(..., description="Token IDs from the model tokenizer")
    include_hex: bool = Field(False, description="Include raw hex bytes in response")


class GlyphInfo(BaseModel):
    glyph_type: str
    primitive_id: int
    primitive_hex: str
    payload_bytes: Optional[int] = None


class EncodeResponse(BaseModel):
    glyphs: List[GlyphInfo]
    token_count: int
    glyph_count: int
    raw_bytes: int           # tokens × 4 (int32 baseline)
    glyph_bytes: int         # compact binary size
    compression_ratio: float
    stream_hex: Optional[str] = None
    latency_ms: float


class DecodeRequest(BaseModel):
    stream_hex: str = Field(..., description="Hex-encoded GlyphStream bytes from /v1/glyphs")


class DecodeResponse(BaseModel):
    glyph_count: int
    glyphs: List[GlyphInfo]
    latency_ms: float


class StatsResponse(BaseModel):
    sigmalang_available: bool
    encode_calls: int
    decode_calls: int
    total_tokens_encoded: int
    total_glyphs_produced: int
    overall_compression_ratio: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _glyph_to_info(g: "Glyph") -> GlyphInfo:
    return GlyphInfo(
        glyph_type=g.glyph_type.name,
        primitive_id=g.primitive_id,
        primitive_hex=f"0x{g.primitive_id:02X}",
        payload_bytes=len(g.payload) if g.payload is not None else None,
    )


def _require_sigmalang():
    if not _SIGMALANG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="sigmalang is not installed. Install it to enable glyph encoding.",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=EncodeResponse, summary="Encode token IDs to glyphs")
async def encode_tokens(req: EncodeRequest) -> EncodeResponse:
    """
    Convert a sequence of token IDs into their Σ-glyph representation.

    Each token maps deterministically to a glyph primitive:
    - Tokens 0-15   → Existential primitives (ENTITY, ACTION, …)
    - Tokens 16-127 → Domain primitives
    - Tokens 128+   → Learned/Reference primitives (wrapped)

    The returned `stream_hex` can be passed to POST /v1/glyphs/decode
    for round-trip verification.
    """
    _require_sigmalang()

    t0 = time.perf_counter()

    glyphs: List[Glyph] = _mapper.tokens_to_glyphs(req.tokens)
    stream = GlyphStream(glyphs=glyphs)
    stream_bytes = stream.to_bytes()

    raw_bytes = len(req.tokens) * 4
    glyph_bytes = len(stream_bytes)
    compression_ratio = raw_bytes / glyph_bytes if glyph_bytes > 0 else 1.0

    latency_ms = (time.perf_counter() - t0) * 1000

    # Update stats
    _stats["encode_calls"] += 1
    _stats["total_tokens_encoded"] += len(req.tokens)
    _stats["total_glyphs_produced"] += len(glyphs)
    _stats["total_bytes_raw"] += raw_bytes
    _stats["total_bytes_glyph"] += glyph_bytes

    return EncodeResponse(
        glyphs=[_glyph_to_info(g) for g in glyphs],
        token_count=len(req.tokens),
        glyph_count=len(glyphs),
        raw_bytes=raw_bytes,
        glyph_bytes=glyph_bytes,
        compression_ratio=round(compression_ratio, 3),
        stream_hex=stream_bytes.hex() if req.include_hex else None,
        latency_ms=round(latency_ms, 3),
    )


@router.post("/decode", response_model=DecodeResponse, summary="Decode a GlyphStream back to glyph info")
async def decode_glyphs(req: DecodeRequest) -> DecodeResponse:
    """
    Deserialize a hex-encoded GlyphStream and return its glyph breakdown.

    This is primarily for round-trip verification and debugging.
    Token IDs are NOT recovered here — glyph→token mapping requires
    a learned inverse table (future sprint work).
    """
    _require_sigmalang()

    t0 = time.perf_counter()

    try:
        raw = bytes.fromhex(req.stream_hex)
        stream = GlyphStream.from_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid GlyphStream bytes: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    _stats["decode_calls"] += 1

    return DecodeResponse(
        glyph_count=len(stream.glyphs),
        glyphs=[_glyph_to_info(g) for g in stream.glyphs],
        latency_ms=round(latency_ms, 3),
    )


@router.get("/stats", response_model=StatsResponse, summary="Encoding statistics")
async def get_stats() -> StatsResponse:
    raw = _stats["total_bytes_raw"]
    glyph = _stats["total_bytes_glyph"]
    ratio = round(raw / glyph, 3) if glyph > 0 else 1.0

    return StatsResponse(
        sigmalang_available=_SIGMALANG_AVAILABLE,
        encode_calls=_stats["encode_calls"],
        decode_calls=_stats["decode_calls"],
        total_tokens_encoded=_stats["total_tokens_encoded"],
        total_glyphs_produced=_stats["total_glyphs_produced"],
        overall_compression_ratio=ratio,
    )
