"""
Ryzanstein LLM — FastAPI Server
================================

OpenAI-compatible inference API for the Ryzanstein CPU-first LLM engine.

Endpoints
---------
POST /v1/chat/completions   — OpenAI-compat chat (streaming + non-streaming)
POST /v1/embeddings         — 1024-dim embedding vectors from last hidden state
GET  /v1/models             — list available models
GET  /health                — readiness / liveness probe
GET  /mcp/tools             — MCP tool-use manifest (for mcp-mesh registration)
/v1/glyphs/*                — Σ-Glyph encoding API (Sprint 1)

Environment
-----------
MODEL_PATH          Path to model weights (safetensors). Optional; stub mode if absent.
RYZANSTEIN_PORT     Listen port (default 8000).
RYZANSTEIN_HOST     Listen host (default 0.0.0.0).
EMBED_DIM           Embedding dimension (default 1024).
"""

import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .glyphs import router as glyphs_router
from .security import rate_limit, require_auth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "ryzanstein-bitnet-7b")
MODEL_PATH = os.getenv("MODEL_PATH", "")
EMBED_DIM  = int(os.getenv("EMBED_DIM", "1024"))
BACKEND    = os.getenv("RYZANSTEIN_BACKEND", "stub")  # stub | ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwythos-9b")
# Separate from OLLAMA_MODEL: chat models (qwythos-9b, phi4-mini, ...) are not
# embedding-capable, and Ollama's /v1/embeddings 501s if asked to embed with
# one. This was the actual reason every /v1/embeddings caller (sigma-compress,
# sigma-index, sigma-diff) always fell back to a local/non-semantic path --
# not that Ryzanstein was unreachable, but that every real call it forwarded
# errored upstream. nomic-embed-text is already pulled on this box.
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Per-request chat-model selection (roadmap C1) -------------------------
# /v1/chat/completions historically pinned OLLAMA_MODEL and ignored request.model.
# Allow an ALLOWLISTED per-request model so callers can opt into a stronger
# "quality" tier (gemma3:4b-it-qat) over the fast default. The recycler cache
# keys on the RESOLVED model (lookup/store), so a quality-tier answer is never
# served to a fast-tier request for the same prompt.
ALLOWED_CHAT_MODELS = set(
    m.strip() for m in os.getenv(
        "GATEWAY_ALLOWED_MODELS",
        "phi4-mini,gemma3:4b,gemma3:4b-it-qat,qwythos-9b,granite4:1b,qwen3:30b",
    ).split(",") if m.strip()
)

def _resolve_chat_model(requested: str) -> str:
    r = (requested or "").strip()
    return r if r in ALLOWED_CHAT_MODELS else OLLAMA_MODEL

# Small model dedicated to the MCP tool/agent layer (roadmap C2): Granite 4.0
# leads structured function-calling in its size class.
MCP_TOOL_MODEL = os.getenv("MCP_TOOL_MODEL", "granite4:1b")

_MODEL_CREATED_TS = 1_700_000_000   # stable epoch for /v1/models

# ---------------------------------------------------------------------------
# Stub model — used when MODEL_PATH is absent or loading fails
# ---------------------------------------------------------------------------

class _StubModel(nn.Module):
    """Lightweight stub used when no real weights are available."""
    def __init__(self, embed_dim: int = EMBED_DIM, vocab_size: int = 32_000):
        super().__init__()
        self.embed_dim = embed_dim
        torch.manual_seed(0)
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.out_proj  = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [batch, vocab_size]."""
        h = self.token_emb(input_ids).mean(dim=1)  # mean-pool → [batch, dim]
        return self.out_proj(h)

    def last_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled embeddings [batch, embed_dim]."""
        return self.token_emb(input_ids).mean(dim=1)


def _load_model() -> nn.Module:
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        try:
            from safetensors.torch import load_file
            state = load_file(MODEL_PATH)
            logger.info(f"Loaded model weights from {MODEL_PATH}")
            # Attach weights to stub (real impl would use proper architecture)
            model = _StubModel()
            model.load_state_dict(state, strict=False)
            return model
        except Exception as e:
            logger.warning(f"Could not load {MODEL_PATH}: {e} — using stub")
    logger.info("No MODEL_PATH set — running in stub mode")
    return _StubModel()


_model: Optional[nn.Module] = None

def _get_model() -> nn.Module:
    global _model
    if _model is None:
        _model = _load_model()
        _model.eval()
    return _model


# ---------------------------------------------------------------------------
# Simple tokenizer shim (BPE not required for stub mode)
# ---------------------------------------------------------------------------

def _tokenize(text: str, max_len: int = 512) -> List[int]:
    """Byte-level tokenizer shim. Clamped to vocab 0-255."""
    ids = [b % 256 for b in text.encode("utf-8")][:max_len]
    return ids or [0]


# ---------------------------------------------------------------------------
# Ollama proxy backend
# ---------------------------------------------------------------------------

async def _ollama_chat(messages: list, model: str, max_tokens: int,
                       temperature: float, top_p: float, stream: bool) -> dict:
    """Forward a non-streaming chat completion to Ollama's OpenAI-compat endpoint.

    NOTE: this always requests stream=False from Ollama regardless of the
    `stream` parameter -- by design, callers that want a real stream must use
    _ollama_chat_stream() below instead. (Previously this function silently
    ignored `stream` entirely and always hit the non-streaming path even when
    a caller asked for streaming; see chat_completions() for the fix.)
    """
    import httpx
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()


async def _ollama_chat_stream(messages: list, model: str, max_tokens: int,
                              temperature: float, top_p: float):
    """Forward a streaming chat completion to Ollama's OpenAI-compat endpoint.

    Yields each SSE "data: ..." line from Ollama's response as it arrives
    (stripped of trailing newlines -- caller re-adds framing). Purely a
    passthrough generator; it does not itself accumulate or cache anything.
    See _gw_stream_and_tee_to_cache() for the wrapper that re-frames these
    lines for the client AND accumulates the answer text in the background so
    it can be stored into the Token Recycler once the stream completes,
    without buffering or delaying the client's real-time stream.
    """
    import httpx

    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    # _GW_UPSTREAM_TIMEOUT is defined later in this module (gateway section) but
    # resolved at call time, not def time, so this is safe: by the time any
    # request reaches here the module has finished importing.
    async with httpx.AsyncClient(timeout=_GW_UPSTREAM_TIMEOUT) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                yield line


async def _ollama_embed(texts: List[str], model: str) -> dict:
    """Forward embedding request to Ollama's OpenAI-compat endpoint."""
    import httpx
    payload = {"model": model, "input": texts}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/v1/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class _Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[_Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int  = Field(default=256, ge=1, le=4096)
    top_p: float     = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool     = False


class EmbeddingRequest(BaseModel):
    input: Any          # str | List[str]
    model: str = MODEL_NAME
    encoding_format: str = "float"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ryzanstein LLM API",
    version="3.1.0",
    description="CPU-First LLM Inference — OpenAI-compatible REST API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount glyph API (Sprint 1)
app.include_router(glyphs_router)

_start_time = time.time()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _start_time, 1),
        "model": OLLAMA_MODEL if BACKEND == "ollama" else MODEL_NAME,
        "backend": BACKEND,
        "stub_mode": BACKEND == "stub" and (MODEL_PATH == "" or not os.path.exists(MODEL_PATH)),
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": _MODEL_CREATED_TS,
                "owned_by": "ryzanstein",
                "capabilities": {
                    "chat_completions": True,
                    "embeddings": True,
                    "glyph_encoding": True,
                },
            }
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if model_id != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return (await list_models())["data"][0]


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

def _generate_tokens(input_ids: List[int], max_tokens: int) -> List[int]:
    """
    Greedy token generation using the loaded/stub model.
    Returns at most max_tokens new token IDs.
    """
    model = _get_model()
    generated: List[int] = []
    current_ids = list(input_ids[-512:])  # context window cap

    with torch.no_grad():
        for _ in range(max_tokens):
            t = torch.tensor([current_ids], dtype=torch.long)
            logits = model(t)   # [1, vocab_size]
            next_tok = int(torch.argmax(logits[0]).item())
            generated.append(next_tok)
            current_ids.append(next_tok)
            # Stop on EOS (token 0 in stub mode) or padding
            if next_tok == 0 and len(generated) > 4:
                break

    return generated


def _decode_tokens(token_ids: List[int]) -> str:
    """Byte-level decode back to text. Filters non-printable bytes."""
    raw = bytes([t & 0xFF for t in token_ids])
    return raw.decode("utf-8", errors="replace")


def _build_completion_response(
    request_id: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_completion(
    request_id: str,
    content: str,
    chunk_size: int = 4,
) -> AsyncIterator[str]:
    """Yield SSE chunks character-by-character (chunked)."""
    import json as _json
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i + chunk_size]
        delta = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }
        yield f"data: {_json.dumps(delta)}\n\n"

    # Final chunk with finish_reason
    final = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {_json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    req_id = uuid.uuid4().hex[:12]

    if BACKEND == "ollama":
        return await _gw_openai_chat_completions(req_id, request)

    # Stub/local backend: build flat prompt from messages
    prompt_parts = []
    for msg in request.messages:
        prefix = {"system": "<<SYS>>", "user": "User:", "assistant": "Asst:"}.get(msg.role, "")
        prompt_parts.append(f"{prefix} {msg.content}")
    prompt = "\n".join(prompt_parts)

    prompt_ids   = _tokenize(prompt)
    gen_ids      = _generate_tokens(prompt_ids, request.max_tokens)
    content      = _decode_tokens(gen_ids)

    if request.stream:
        return StreamingResponse(
            _stream_completion(req_id, content),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    return JSONResponse(
        _build_completion_response(req_id, content, len(prompt_ids), len(gen_ids))
    )


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """
    Generate embedding vectors from input text(s).

    Returns embedding vectors derived from the model's last hidden state
    (mean-pool over token positions). This is the interface consumed by:
      - sigma-compress  (semantic deduplication)
      - sigma-index     (HNSW approximate search)
      - sigma-diff      (similarity scoring)

    Input:
        {"input": "text" | ["text1", "text2", ...], "model": "ryzanstein-bitnet-7b"}

    Output:
        {"object": "list", "data": [{"embedding": [...floats], "index": 0}], ...}
    """
    inputs: List[str] = [request.input] if isinstance(request.input, str) else list(request.input)

    if BACKEND == "ollama":
        return JSONResponse(await _ollama_embed(inputs, OLLAMA_EMBED_MODEL))

    model = _get_model()

    if not inputs:
        raise HTTPException(status_code=400, detail="'input' must be a non-empty string or list")

    embeddings = []
    total_tokens = 0

    with torch.no_grad():
        for i, text in enumerate(inputs):
            token_ids = _tokenize(text, max_len=512)
            total_tokens += len(token_ids)
            t = torch.tensor([token_ids], dtype=torch.long)

            # Use last_hidden_state method if available (real model), else token emb mean
            if hasattr(model, "last_hidden_state"):
                vec = model.last_hidden_state(t)  # [1, embed_dim]
            elif hasattr(model, "token_emb"):
                vec = model.token_emb(t).mean(dim=1)
            else:
                # Generic: run forward and use the logits as a proxy embedding
                # (falls back gracefully for arbitrary nn.Module)
                vec = model(t)

            # L2-normalise so cosine similarity = dot product
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
            emb = vec[0].tolist()

            # Pad or truncate to EMBED_DIM
            if len(emb) < EMBED_DIM:
                emb = emb + [0.0] * (EMBED_DIM - len(emb))
            else:
                emb = emb[:EMBED_DIM]

            embeddings.append({"object": "embedding", "index": i, "embedding": emb})

    return {
        "object": "list",
        "model": MODEL_NAME,
        "data": embeddings,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


# ---------------------------------------------------------------------------
# /mcp/tools  (Sprint 3 — MCP protocol adapter)
# ---------------------------------------------------------------------------

_MCP_TOOLS = [
    {
        "name": "generate",
        "description": "Generate text from a prompt using Ryzanstein LLM.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt":     {"type": "string",  "description": "Input prompt"},
                "max_tokens": {"type": "integer", "description": "Max tokens to generate", "default": 256},
                "temperature":{"type": "number",  "description": "Sampling temperature", "default": 0.7},
            },
            "required": ["prompt"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "text":             {"type": "string"},
                "tokens_generated": {"type": "integer"},
            },
        },
    },
    {
        "name": "embed",
        "description": "Compute a 1024-dim embedding vector for a text snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to embed"},
            },
            "required": ["text"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "embedding": {"type": "array", "items": {"type": "number"}},
                "dim":       {"type": "integer"},
            },
        },
    },
    {
        "name": "encode_glyphs",
        "description": "Encode token IDs into compact Sigma-Glyph byte sequences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of token IDs to encode",
                },
            },
            "required": ["tokens"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "glyph_hex":        {"type": "string"},
                "compression_ratio":{"type": "number"},
                "glyph_count":      {"type": "integer"},
            },
        },
    },
]


@app.get("/mcp/tools")
async def mcp_list_tools():
    """
    MCP tool manifest — consumed by mcp-mesh to register Ryzanstein as an agent.

    The mesh reads this endpoint at startup and exposes each tool to the
    broader Sigma Ecosystem agent network via the MCP protocol.
    """
    return {
        "schema_version": "1.0",
        "server_name": "ryzanstein-llm",
        "server_version": "3.1.0",
        "tools": _MCP_TOOLS,
    }


class _McpToolCallRequest(BaseModel):
    name: str
    input: Dict[str, Any]


@app.post("/mcp/tools/call")
async def mcp_call_tool(
    request: _McpToolCallRequest,
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """
    Execute an MCP tool call.  Dispatches to the matching FastAPI handler.
    """
    if request.name == "generate":
        prompt     = request.input.get("prompt", "")
        max_tokens = int(request.input.get("max_tokens", 256))
        # Real generation via the tool-calling model (Granite 4.0, roadmap C2)
        # instead of the former local token stub. Allowlisted; falls back to the
        # default model on a bad/absent model.
        model = request.input.get("model") or MCP_TOOL_MODEL
        if model not in ALLOWED_CHAT_MODELS:
            model = OLLAMA_MODEL
        try:
            # _ollama_chat reads msg.role/msg.content (objects, not dicts)
            from types import SimpleNamespace
            _msg = SimpleNamespace(role="user", content=prompt)
            result = await _ollama_chat([_msg], model, max_tokens, 0.2, 1.0, False)
            text = ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            usage = result.get("usage") or {}
            return {"text": text, "model": model,
                    "tokens_generated": usage.get("completion_tokens", 0)}
        except Exception as e:
            logger.warning(f"mcp generate via {model} failed: {e}")
            return {"text": "", "model": model, "error": str(e), "tokens_generated": 0}

    elif request.name == "embed":
        text = request.input.get("text", "")
        emb_resp = await create_embeddings(EmbeddingRequest(input=text))
        vec = emb_resp["data"][0]["embedding"]
        return {"embedding": vec, "dim": len(vec)}

    elif request.name == "encode_glyphs":
        tokens = request.input.get("tokens", [])
        if not tokens:
            raise HTTPException(status_code=400, detail="'tokens' list is required")
        # Delegate to glyphs router logic inline
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "sigmalang"))
            from sigmalang.core.primitives import Glyph, GlyphStream, GlyphType, ExistentialPrimitive

            def _map(tok: int) -> Glyph:
                if tok < 16:
                    return Glyph(GlyphType.PRIMITIVE, tok, None)
                elif tok < 128:
                    return Glyph(GlyphType.PRIMITIVE, 0x10 + (tok - 16), None)
                else:
                    pid = 0x80 + ((tok - 128) % 128)
                    gt = GlyphType.REFERENCE if tok >= 256 else GlyphType.PRIMITIVE
                    return Glyph(gt, pid, None)

            glyphs = [_map(t) for t in tokens]
            stream = GlyphStream(glyphs=glyphs)
            glyph_bytes = stream.to_bytes()
            raw_bytes   = len(tokens) * 4
            ratio       = raw_bytes / len(glyph_bytes) if glyph_bytes else 1.0
            return {
                "glyph_hex":         glyph_bytes.hex(),
                "compression_ratio": round(ratio, 3),
                "glyph_count":       len(glyphs),
            }
        except ImportError:
            raise HTTPException(status_code=503, detail="sigmalang not installed")

    else:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {request.name!r}")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": type(exc).__name__}},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("RYZANSTEIN_HOST", "0.0.0.0")
    port = int(os.getenv("RYZANSTEIN_PORT", "8000"))
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)



# === Ollama-compatible gateway + Token Recycler answer cache (Tier 0.1/0.2, 2026-07-01) =====
# Transparently reverse-proxies Ollama's native /api/* to the real ollama daemon so
# Ryzanstein :8000 is the single LLM entry point every consumer routes through (superset
# of ollama's API plus this server's /v1). For POST /api/generate and /api/chat with
# stream=false, checks Ryot's own Token Recycling System [REF:TR-006] (RSU semantic
# cache over Qdrant, RYZEN-LLM/src/recycler) before forwarding, and stores the answer
# on a miss. Streaming requests pass through uncached in this MVP.
import os as _gw_os
import sys as _gw_sys
import json as _gw_json
import time as _gw_time
import httpx as _gw_httpx
from urllib.parse import urlparse as _gw_urlparse
from fastapi import Request as _GwRequest
from fastapi.responses import StreamingResponse as _GwStreaming, JSONResponse as _GwJSON
from starlette.background import BackgroundTask as _GwBg

_GW_OLLAMA_URL = _gw_os.getenv(
    "GATEWAY_OLLAMA_URL", _gw_os.getenv("OLLAMA_URL", "http://localhost:11434")
)
# Upstream request timeout for the cache-miss path (generate/chat). CPU-only
# inference for longer prompts on this box routinely takes 300-320s, which
# exactly matched the old 300.0s default and caused reproducible 500s for
# real content-generation workloads (confirmed via review-roundup-automator
# and saas-alternatives-directory). Bounded rather than unbounded so a truly
# hung upstream still eventually fails instead of holding the connection open
# forever.
_GW_UPSTREAM_TIMEOUT = float(_gw_os.getenv("GATEWAY_UPSTREAM_TIMEOUT", "900.0"))

# --- wire in Ryot's own Token Recycling System (RYZEN-LLM/src/recycler) ---
_RYZEN_LLM_SRC = _gw_os.path.abspath(
    _gw_os.path.join(_gw_os.path.dirname(__file__), "..", "..", "RYZEN-LLM", "src")
)
if _RYZEN_LLM_SRC not in _gw_sys.path:
    _gw_sys.path.insert(0, _RYZEN_LLM_SRC)
try:
    from recycler import SemanticCompressor, VectorBank, SelectiveRetriever
    from sigma_core import SigmalangClient
    _RECYCLER_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - fail-open if the package can't load
    _RECYCLER_IMPORT_ERROR = str(_e)

_RECYCLER_ENABLED = _gw_os.getenv("RECYCLER_ENABLED", "true").lower() not in ("0", "false", "no")
_RECYCLER_EMBED_MODEL = _gw_os.getenv("RECYCLER_EMBED_MODEL", "nomic-embed-text")
_RECYCLER_THRESHOLD = float(_gw_os.getenv("RECYCLER_THRESHOLD", "0.97"))
_RECYCLER_TTL = int(_gw_os.getenv("RECYCLER_TTL_SECONDS", "86400"))
# sigmalang: an ADDITIONAL similarity signal, checked only AFTER the primary
# embedding threshold above already accepted a hit. Deliberately permissive
# default -- see sigma_core.sigmalang's docstring for the calibration that set
# 0.4 (its cosine doesn't reliably discriminate topic at this dimension; the
# goal here is structural readiness + a logged score, not a strict filter).
# Fails open: if the sigmalang service is unreachable, the gate is skipped and
# the primary signal's decision stands unchanged.
_SIGMALANG_ENABLED = _gw_os.getenv("SIGMALANG_GATE_ENABLED", "true").lower() not in ("0", "false", "no")
_SIGMALANG_THRESHOLD = float(_gw_os.getenv("SIGMALANG_THRESHOLD", "0.4"))
_QDRANT_URL = _gw_os.getenv("QDRANT_URL", "http://localhost:6333")
_SIGMA_INDEX_URL = _gw_os.getenv("SIGMA_INDEX_URL", "http://localhost:8200")
_SIGMA_INDEX_DUALWRITE = _gw_os.getenv("SIGMA_INDEX_DUALWRITE", "true").lower() not in (
    "0", "false", "no"
)


def _gw_parse_host_port(url: str, default_port: int):
    u = _gw_urlparse(url)
    return u.hostname or "localhost", u.port or default_port


# Hand-rolled counters (module-level dict, same rationale as /metrics below:
# no prometheus_client dependency). Previously store() had no observability at
# all beyond a couple of log lines at warning/debug level -- a failing cache
# write was otherwise invisible. These three counters are surfaced via both
# /metrics (Prometheus text exposition) and /v1/recycler/stats (JSON).
_GW_METRICS = {
    "stores_total": 0,          # successful recycler.store() calls (RSU + embed OK)
    "store_failures_total": 0,  # recycler.store() calls where compress/embed/Qdrant write failed
    "passthrough_total": 0,     # /v1/chat/completions requests handled by the ollama backend
                                # (cache hit or miss, streaming or not) -- i.e. traffic that went
                                # through the Token Recycler-aware path at all.
}


class _TokenRecyclerCache:
    """Glues SemanticCompressor + VectorBank + SelectiveRetriever into the
    answer-level cache the Ryzanstein gateway checks on generate/chat calls."""

    def __init__(self):
        qhost, qport = _gw_parse_host_port(_QDRANT_URL, default_port=6333)
        self.bank = VectorBank(host=qhost, port=qport, collection_name="rsu_bank", vector_size=768)
        self.retriever = SelectiveRetriever(self.bank, top_k=1)
        self.compressor = SemanticCompressor(embed_fn=self._embed)
        self.sigmalang = SigmalangClient()
        self.hits = 0
        self.misses = 0
        self.sigmalang_rejected = 0
        self.sigmalang_last_score = None

    async def _embed(self, text: str):
        result = await _ollama_embed([text], _RECYCLER_EMBED_MODEL)
        return result["data"][0]["embedding"]

    async def lookup(self, prompt: str, model: str):
        try:
            query_vec = await self.compressor.embed_query(prompt)
            hit = await self.retriever.retrieve(
                query_vec,
                score_threshold=_RECYCLER_THRESHOLD,
                # Filtering on "model" alone is not enough once a second
                # backend can serve the same model name (e.g. a future
                # llama.cpp/BitNet backend alongside "ollama") -- without the
                # "backend" filter too, a hit stored by one backend could be
                # served as a cache hit for a request routed to a different
                # backend, even though the two may answer differently for
                # the same prompt. BACKEND is bound into both this filter and
                # store()'s payload (see below), so lookups only ever match
                # RSUs written by the same backend.
                filter_dict={"must": [
                    {"key": "model", "match": {"value": model}},
                    {"key": "backend", "match": {"value": BACKEND}},
                ]},
            )
        except Exception as e:
            logger.warning(f"Token Recycler lookup failed (degrading to miss): {e}")
            return None
        if hit is None:
            self.misses += 1
            return None
        created_ts = hit.metadata.get("_created_ts", 0) or 0
        if (_gw_time.time() - created_ts) > _RECYCLER_TTL:
            self.misses += 1
            return None
        # sigmalang: an ADDITIONAL signal checked only after the primary
        # embedding threshold above already accepted this candidate. Fails
        # open -- a sigmalang outage/error never turns an accepted hit into a
        # miss; it only skips the extra check and logs why.
        if _SIGMALANG_ENABLED:
            try:
                sig_score = await self.sigmalang.asimilarity(prompt, hit.prompt)
                self.sigmalang_last_score = sig_score
                if sig_score < _SIGMALANG_THRESHOLD:
                    self.sigmalang_rejected += 1
                    self.misses += 1
                    return None
            except Exception as e:
                logger.debug(f"sigmalang gate check failed (failing open, hit stands): {e}")
        self.hits += 1
        return hit.answer

    async def store(self, prompt: str, model: str, answer: str) -> None:
        try:
            # "backend" rides in metadata exactly like "_created_ts" already
            # does -- VectorBank.store() spreads rsu.metadata into the
            # top-level Qdrant payload, so this becomes a real, filterable
            # "backend" field lookup() can match against (see lookup()'s
            # filter_dict above for why this matters).
            rsu = await self.compressor.compress(
                prompt, answer, model,
                metadata={"_created_ts": _gw_time.time(), "backend": BACKEND},
            )
            await self.bank.store(rsu)
        except Exception as e:
            # Previously this was the only trace of a failed cache write: a
            # single warning log, no counter. A failing store here means every
            # future semantically-identical request re-runs full inference
            # instead of being served from cache -- worth being able to see in
            # Grafana/curl /metrics, not just grep the logs after the fact.
            _GW_METRICS["store_failures_total"] += 1
            logger.warning(f"Token Recycler store failed (ignored, cache write lost "
                          f"for this answer -- store_failures_total={_GW_METRICS['store_failures_total']}): {e}")
            return
        _GW_METRICS["stores_total"] += 1
        # Shadow dual-write: sigma-index is being built up in parallel as a
        # future Qdrant replacement (see Tier 0.2 plan). Best-effort, fail-open
        # -- never let sigma-index affect the primary cache path. Previously
        # this failure was logged at debug level only (i.e. invisible by
        # default) and uncounted; raised to warning + counted so a persistent
        # dual-write outage is actually noticeable, while still never raising
        # or affecting the primary Qdrant store above.
        if _SIGMA_INDEX_DUALWRITE:
            try:
                async with _gw_httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        f"{_SIGMA_INDEX_URL}/add",
                        json={
                            "namespace": "token_recycler",
                            "id": rsu.id,
                            "vector": rsu.embedding,
                            "text": rsu.prompt,
                        },
                    )
                    resp.raise_for_status()
            except Exception as e:
                logger.warning(f"sigma-index shadow dual-write failed (ignored, primary "
                              f"Qdrant cache unaffected): {e}")


_recycler_cache = None


def _get_recycler():
    global _recycler_cache
    if not _RECYCLER_ENABLED or _RECYCLER_IMPORT_ERROR:
        return None
    if _recycler_cache is None:
        _recycler_cache = _TokenRecyclerCache()
    return _recycler_cache


def _gw_extract_prompt(path: str, body: dict) -> str:
    if path == "generate":
        return body.get("prompt", "") or ""
    if path == "chat":
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in body.get("messages", [])
        )
    return ""


def _gw_extract_answer(path: str, resp_json: dict) -> str:
    if path == "generate":
        return resp_json.get("response", "") or ""
    if path == "chat":
        return (resp_json.get("message") or {}).get("content", "") or ""
    return ""


def _gw_wrap_cached_answer(path: str, model: str, answer: str) -> dict:
    base = {"model": model, "done": True, "done_reason": "stop"}
    if path == "generate":
        base["response"] = answer
    else:
        base["message"] = {"role": "assistant", "content": answer}
    return base


# ---------------------------------------------------------------------------
# OpenAI-shaped Token Recycler adapter for /v1/chat/completions
# ---------------------------------------------------------------------------
# The /api/{path} gateway above caches against Ollama's native {response:...} /
# {message:{content:...}} shape via _gw_extract_answer / _gw_wrap_cached_answer.
# /v1/chat/completions speaks OpenAI's {choices:[{message:{...}}], usage:{...}}
# shape instead, so it needs its own extract/wrap pair rather than reusing
# those directly -- the underlying cache (RSU prompt/answer/model triples) is
# the same, only the request/response envelope differs.

def _gw_extract_openai_prompt(messages: list) -> str:
    """Build the recycler lookup/store key text from ChatCompletionRequest.messages.

    Mirrors _gw_extract_prompt's "chat" branch (role: content per line) but
    takes the Pydantic _Message objects /v1/chat/completions already validated
    into, rather than a raw dict body.
    """
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _gw_wrap_openai_cached_answer(request_id: str, model: str, answer: str) -> dict:
    """Cache-hit response in OpenAI chat.completion shape (see _build_completion_response)."""
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(_gw_time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        # Cache hits skip real inference, so there's no real token accounting
        # to report -- 0 rather than a fabricated estimate. Consumers that key
        # off X-Cache/X-Served-By already know this was a cache hit.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _gw_stream_openai_cached_answer(request_id: str, model: str, answer: str,
                                          chunk_size: int = 4) -> AsyncIterator[str]:
    """SSE synthesis for a cache HIT under stream=true -- same chunked-delta
    framing _stream_completion uses for the stub backend's real streaming, so
    OpenAI-compatible clients see an identical chunk shape whether the answer
    came from cache or a live generation."""
    for i in range(0, len(answer), chunk_size):
        chunk_text = answer[i:i + chunk_size]
        delta = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(_gw_time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }
        yield f"data: {_gw_json.dumps(delta)}\n\n"

    final = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(_gw_time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {_gw_json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _gw_stream_and_tee_to_cache(request_id: str, model: str, prompt_text: str,
                                      messages: list, max_tokens: int, temperature: float,
                                      top_p: float, recycler) -> AsyncIterator[str]:
    """Real streaming passthrough for a cache MISS under stream=true.

    Forwards Ollama's own OpenAI-compat SSE lines to the client as they arrive
    (no buffering -- each line is yielded the moment it's read off the
    upstream response), while accumulating the assistant's `delta.content`
    fragments in the background. Once the upstream stream completes
    successfully (the [DONE] sentinel is observed), the accumulated full
    answer is stored into the Token Recycler -- this happens AFTER the client
    has already received every chunk, so it never adds latency to the
    client's real-time streaming experience. If accumulation or the store
    call fails, the client-visible stream is unaffected (the exception is
    caught and logged, matching _TokenRecyclerCache.store's own fail-open
    contract).

    Deliberately does NOT store on an aborted/partial stream (client
    disconnect, upstream error mid-stream, generator closed early): storing a
    truncated accumulation as if it were the complete answer would poison the
    cache with a cut-off response that a future exact-prompt hit would then
    serve as if it were whole. completed_ok tracks this explicitly rather than
    inferring completeness from "accumulated is non-empty".
    """
    accumulated: List[str] = []
    completed_ok = False
    try:
        async for line in _ollama_chat_stream(messages, model, max_tokens, temperature, top_p):
            yield f"{line}\n\n"
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                completed_ok = True
                continue
            try:
                chunk = _gw_json.loads(payload)
                delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                content = delta.get("content")
                if content:
                    accumulated.append(content)
            except Exception as e:
                logger.debug(f"chat/completions stream-tee: could not parse chunk for "
                             f"accumulation (client stream unaffected): {e}")
    finally:
        if recycler is not None and completed_ok and accumulated:
            full_answer = "".join(accumulated)
            try:
                await recycler.store(prompt_text, model, full_answer)
            except Exception as e:
                # store() already fails open/logs internally; this is a second
                # layer of defense specific to the streaming tee path so a
                # failure here can never surface to the (already-completed)
                # client stream.
                logger.warning(f"chat/completions stream-tee: cache store failed "
                              f"(ignored, client stream already completed): {e}")


async def _gw_openai_chat_completions(req_id: str, request: "ChatCompletionRequest"):
    """BACKEND=="ollama" path for POST /v1/chat/completions.

    Routes through the same Token Recycler cache the /api/{path} gateway uses,
    instead of unconditionally forwarding to Ollama. Covers all four
    stream x cache-state combinations:
      - stream=false, hit:  synchronous OpenAI-shaped JSON from cache
      - stream=false, miss: real Ollama call, store on success, return JSON
      - stream=true,  hit:  synthesized SSE from the cached answer
      - stream=true,  miss: real SSE passthrough, tee accumulated text into
                            the cache once the stream completes

    Model cache key: uses OLLAMA_MODEL (the model actually served) rather than
    request.model. Verified real callers across the ecosystem send differing
    model labels for what all resolve to the same served model on this box
    (myceloforge default "ryzanstein-bitnet-7b", NEURECTOMY spectrum-workspace
    default "ryot-bitnet-7b", sigma-index's Go inference_client default
    "ryzanstein-bitnet-3b", vs. this server's own MODEL_NAME default
    "ryzanstein-bitnet-7b") -- none of that is honored today; this handler
    already ignored request.model entirely and always forwarded OLLAMA_MODEL
    (see _ollama_chat call below). Keying the cache on request.model as-sent
    would silently fragment the cache across labels that all hit the exact
    same upstream model, without ever actually selecting a different model.
    Keying on OLLAMA_MODEL instead makes the cache key match what is actually
    served, and stays consistent with the /api/{path} gateway's own contract
    of keying on the model string that request is actually forwarded with.
    """
    recycler = _get_recycler()
    prompt_text = _gw_extract_openai_prompt(request.messages)
    model = _resolve_chat_model(getattr(request, "model", ""))

    if not request.stream:
        if recycler is not None and prompt_text:
            # _TokenRecyclerCache.lookup() already has its own internal
            # try/except (degrades to a miss on failure), but this call site
            # wraps it too: /v1/chat/completions must keep working even if a
            # *different* recycler implementation (or a future refactor) ever
            # lets an exception through lookup() itself.
            try:
                cached = await recycler.lookup(prompt_text, model)
            except Exception as e:
                logger.warning(f"chat/completions: recycler.lookup() raised, treating as "
                              f"miss (client response unaffected): {e}")
                cached = None
            if cached is not None:
                _GW_METRICS["passthrough_total"] += 1
                return JSONResponse(
                    _gw_wrap_openai_cached_answer(req_id, model, cached),
                    headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "hit"},
                )
        result = await _ollama_chat(
            request.messages, model, request.max_tokens,
            request.temperature, request.top_p, request.stream,
        )
        if recycler is not None and prompt_text:
            answer_text = ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            if answer_text:
                # Same reasoning as the lookup() wrapper above: store() is
                # expected to fail open internally, but the already-successful
                # Ollama answer must reach the client even if a cache-store
                # exception somehow escapes it anyway.
                try:
                    await recycler.store(prompt_text, model, answer_text)
                except Exception as e:
                    logger.warning(f"chat/completions: recycler.store() raised (ignored, "
                                  f"client response unaffected): {e}")
        _GW_METRICS["passthrough_total"] += 1
        return JSONResponse(result, headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "miss"})

    # stream=true
    if recycler is not None and prompt_text:
        cached = await recycler.lookup(prompt_text, model)
        if cached is not None:
            _GW_METRICS["passthrough_total"] += 1
            return StreamingResponse(
                _gw_stream_openai_cached_answer(req_id, model, cached),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "X-Served-By": "ryzanstein-gateway", "X-Cache": "hit"},
            )

    _GW_METRICS["passthrough_total"] += 1
    return StreamingResponse(
        _gw_stream_and_tee_to_cache(
            req_id, model, prompt_text, request.messages,
            request.max_tokens, request.temperature, request.top_p, recycler,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "X-Served-By": "ryzanstein-gateway", "X-Cache": "miss"},
    )


@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE"])
async def _ollama_api_gateway(path: str, request: _GwRequest):
    body_bytes = await request.body()

    # Cache-aware handling for non-streaming generate/chat (the Token Recycler)
    if path in ("generate", "chat") and request.method == "POST":
        try:
            body = _gw_json.loads(body_bytes) if body_bytes else {}
        except ValueError:
            body = {}
        if isinstance(body, dict) and not body.get("stream", True):
            recycler = _get_recycler()
            prompt_text = _gw_extract_prompt(path, body)
            model = body.get("model", "")
            if recycler is not None and prompt_text:
                cached = await recycler.lookup(prompt_text, model)
                if cached is not None:
                    _GW_METRICS["passthrough_total"] += 1
                    return _GwJSON(
                        _gw_wrap_cached_answer(path, model, cached),
                        headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "hit"},
                    )
                async with _gw_httpx.AsyncClient(timeout=_GW_UPSTREAM_TIMEOUT) as client:
                    upstream = await client.post(f"{_GW_OLLAMA_URL}/api/{path}", content=body_bytes)
                    upstream.raise_for_status()
                    resp_json = upstream.json()
                answer_text = _gw_extract_answer(path, resp_json)
                if answer_text:
                    await recycler.store(prompt_text, model, answer_text)
                _GW_METRICS["passthrough_total"] += 1
                return _GwJSON(
                    resp_json,
                    headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "miss"},
                )

    # Generic transparent passthrough for everything else (tags, embed, ps, pull, streaming, ...)
    target = f"{_GW_OLLAMA_URL}/api/{path}"
    fwd = {k: v for k, v in request.headers.items()
           if k.lower() not in ("host", "content-length", "connection")}
    client = _gw_httpx.AsyncClient(timeout=None)
    upstream = await client.send(
        client.build_request(request.method, target, content=body_bytes, headers=fwd,
                             params=dict(request.query_params)),
        stream=True,
    )
    out = {k: v for k, v in upstream.headers.items()
           if k.lower() not in ("content-length", "transfer-encoding",
                                "content-encoding", "connection")}
    out["X-Served-By"] = "ryzanstein-gateway"

    async def _cleanup():
        await upstream.aclose()
        await client.aclose()

    return _GwStreaming(upstream.aiter_raw(), status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type"),
                        headers=out, background=_GwBg(_cleanup))


@app.get("/v1/recycler/stats")
async def _recycler_stats():
    recycler = _get_recycler()
    if recycler is None:
        return {"enabled": False, "reason": _RECYCLER_IMPORT_ERROR or "disabled via env"}
    total = recycler.hits + recycler.misses
    # Previously unguarded: a Qdrant outage made this whole diagnostic endpoint
    # 500 (via the global exception handler) instead of reporting -1/degraded
    # -- the one moment this endpoint is most useful (Qdrant is down) was
    # exactly when it stopped responding. Discovered while manually verifying
    # the new store_failures_total counter against an intentionally-broken
    # QDRANT_URL on a throwaway instance; pre-existing, not introduced by the
    # /v1/chat/completions cache wiring in this change.
    try:
        qdrant_count = await recycler.bank.count()
    except Exception as e:
        logger.warning(f"/v1/recycler/stats: Qdrant count() failed (reporting -1): {e}")
        qdrant_count = -1
    return {
        "enabled": True,
        "hits": recycler.hits,
        "misses": recycler.misses,
        "hit_rate": round(recycler.hits / total, 3) if total else 0.0,
        "rsu_count": qdrant_count,
        "threshold": _RECYCLER_THRESHOLD,
        "ttl_seconds": _RECYCLER_TTL,
        "embed_model": _RECYCLER_EMBED_MODEL,
        "sigmalang_gate_enabled": _SIGMALANG_ENABLED,
        "sigmalang_threshold": _SIGMALANG_THRESHOLD,
        "sigmalang_rejected": recycler.sigmalang_rejected,
        "sigmalang_last_score": recycler.sigmalang_last_score,
        "stores_total": _GW_METRICS["stores_total"],
        "store_failures_total": _GW_METRICS["store_failures_total"],
        "passthrough_total": _GW_METRICS["passthrough_total"],
    }


@app.get("/metrics")
async def _prometheus_metrics():
    # Hand-rolled Prometheus text exposition (no prometheus_client dependency --
    # the box's system Python is externally-managed; avoids a new pip package).
    # Achieves sigma-telemetry's intended purpose (Grafana-scraped observability
    # for Ryzanstein) without the PyO3/HTTP-wrapper work needed to link that Rust
    # crate directly into this Python service -- see memory notes, 2026-07-02.
    from fastapi.responses import PlainTextResponse

    recycler = _get_recycler()
    lines = [
        "# HELP ryzanstein_up Whether the Ryzanstein gateway is serving requests.",
        "# TYPE ryzanstein_up gauge",
        "ryzanstein_up 1",
        "# HELP ryzanstein_recycler_enabled Whether the Token Recycler cache is enabled.",
        "# TYPE ryzanstein_recycler_enabled gauge",
        f"ryzanstein_recycler_enabled {1 if recycler is not None else 0}",
    ]
    if recycler is not None:
        # Same fix as /v1/recycler/stats above: don't let a Qdrant outage take
        # down the /metrics scrape entirely -- that's precisely the moment a
        # Prometheus/Grafana consumer most needs ryzanstein_up=1 to still be
        # visible alongside a degraded recycler reading.
        try:
            rsu_count = await recycler.bank.count()
        except Exception as e:
            logger.warning(f"/metrics: Qdrant count() failed (reporting -1): {e}")
            rsu_count = -1
        lines += [
            "# HELP ryzanstein_recycler_hits_total Token Recycler cache hits.",
            "# TYPE ryzanstein_recycler_hits_total counter",
            f"ryzanstein_recycler_hits_total {recycler.hits}",
            "# HELP ryzanstein_recycler_misses_total Token Recycler cache misses.",
            "# TYPE ryzanstein_recycler_misses_total counter",
            f"ryzanstein_recycler_misses_total {recycler.misses}",
            "# HELP ryzanstein_recycler_rsu_count Recyclable Semantic Units stored in Qdrant.",
            "# TYPE ryzanstein_recycler_rsu_count gauge",
            f"ryzanstein_recycler_rsu_count {rsu_count}",
        ]
    lines += [
        "# HELP ryzanstein_recycler_stores_total Successful Token Recycler cache writes.",
        "# TYPE ryzanstein_recycler_stores_total counter",
        f"ryzanstein_recycler_stores_total {_GW_METRICS['stores_total']}",
        "# HELP ryzanstein_recycler_store_failures_total Token Recycler cache writes "
        "that raised an exception (compress/embed/Qdrant) and were dropped.",
        "# TYPE ryzanstein_recycler_store_failures_total counter",
        f"ryzanstein_recycler_store_failures_total {_GW_METRICS['store_failures_total']}",
        "# HELP ryzanstein_gateway_passthrough_total Requests handled by the "
        "Token-Recycler-aware gateway path (/v1/chat/completions and /api/{generate,chat} "
        "with the ollama backend), regardless of cache hit/miss or streaming mode.",
        "# TYPE ryzanstein_gateway_passthrough_total counter",
        f"ryzanstein_gateway_passthrough_total {_GW_METRICS['passthrough_total']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
