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

import json
import logging
import os
import time
import uuid

# sigma-telemetry (Rust/pyo3): real latency histograms (p50/p95/p99) for the
# serving path. Fail-open -- if the extension is not importable, _TEL stays None
# and every use below is a guarded no-op, so the gateway is never affected.
try:
    import sigma_telemetry as _sigtel
    _TEL = _sigtel.PyMetrics()
except Exception:
    _TEL = None
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

# --- Per-model upstream routing (NUC offload) -------------------------------
# GATEWAY_MODEL_UPSTREAMS maps model name -> base URL of the Ollama host that
# serves it, e.g. '{"ling-mini-2.0":"http://10.88.0.6:11434"}' routes that
# model to sigma-infer over wg0 while everything else stays on OLLAMA_URL.
# Defensive parse: a malformed value must never crash the gateway at import
# time -- it disables routing loudly instead. An empty map reproduces the
# legacy single-upstream behavior exactly, so unsetting the env var is the
# kill switch.
try:
    MODEL_UPSTREAMS: Dict[str, str] = {
        str(k).strip(): str(v).strip().rstrip("/")
        for k, v in json.loads(os.getenv("GATEWAY_MODEL_UPSTREAMS", "{}")).items()
    }
except (ValueError, TypeError, AttributeError):
    logger.exception(
        "GATEWAY_MODEL_UPSTREAMS is not a valid JSON object; "
        "per-model upstream routing DISABLED"
    )
    MODEL_UPSTREAMS = {}
if MODEL_UPSTREAMS:
    logger.info("Per-model upstream routing active: %s", MODEL_UPSTREAMS)

def _resolve_upstream(model: str, default: Optional[str] = None) -> str:
    """Base URL of the Ollama upstream serving `model`.

    Falls back to `default` (or module-wide OLLAMA_URL) when the model is
    not explicitly mapped -- an empty/absent GATEWAY_MODEL_UPSTREAMS thus
    reproduces the old single-upstream behavior exactly.
    """
    return MODEL_UPSTREAMS.get((model or "").strip()) or (default or OLLAMA_URL)

# --- Per-request chat-model selection (roadmap C1) -------------------------
# /v1/chat/completions historically pinned OLLAMA_MODEL and ignored request.model.
# Allow an ALLOWLISTED per-request model so callers can opt into a stronger
# "quality" tier (gemma3:4b-it-qat) over the fast default. The recycler cache
# keys on the RESOLVED model (lookup/store), so a quality-tier answer is never
# served to a fast-tier request for the same prompt.
ALLOWED_CHAT_MODELS = set(
    m.strip() for m in os.getenv(
        "GATEWAY_ALLOWED_MODELS",
        "phi4-mini,gemma3:4b,gemma3:4b-it-qat,qwythos-9b,granite4:1b,ling-mini-2.0",
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
    _t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{_resolve_upstream(model)}/v1/chat/completions", json=payload
        )
        resp.raise_for_status()
        _result = resp.json()
    if _TEL is not None:
        try:
            _TEL.record_histogram("ryzanstein.chat.latency_ms", (time.perf_counter() - _t0) * 1000.0)
            _TEL.increment("ryzanstein.chat.requests_total")
        except Exception:
            pass
    return _result


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
        async with client.stream(
            "POST", f"{_resolve_upstream(model)}/v1/chat/completions", json=payload
        ) as resp:
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
import hashlib as _gw_hashlib
import re as _gw_re
from collections import OrderedDict as _GwOrderedDict
from dataclasses import dataclass as _gw_dataclass, field as _gw_field

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

# /api/{path} passthrough: only inference/read paths may transit the gateway.
# Ollama's admin verbs (pull/push/delete/create/copy) are unauthenticated on
# the upstream, and with per-model routing an upstream may be a remote NUC --
# letting any gateway consumer mutate models there is not acceptable.
_GW_API_ALLOWED_PATHS = frozenset(
    {"chat", "generate", "embeddings", "embed", "show", "tags", "ps", "version"}
)

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
_RECYCLER_THRESHOLD = float(_gw_os.getenv("RECYCLER_THRESHOLD", "0.99"))
_RECYCLER_TTL = int(_gw_os.getenv("RECYCLER_TTL_SECONDS", "86400"))
# Step-4 retrieval-semantics knobs (candidate gate over the semantic L2 path).
_RECYCLER_TOPK = int(_gw_os.getenv("RECYCLER_TOPK", "5"))            # candidates fetched per lookup
_RECYCLER_MARGIN = float(_gw_os.getenv("RECYCLER_MARGIN", "0.01"))  # min score gap #1 vs first DIFFERENT-answer
_RECYCLER_EXACT_TWIN = float(_gw_os.getenv("RECYCLER_EXACT_TWIN", "0.999"))  # accept regardless of margin at/above this
_RECYCLER_MAX_DELETES = int(_gw_os.getenv("RECYCLER_MAX_DELETES", "8"))      # cap delete-on-expire per lookup
# Param-awareness (step 3, folded in): bucket sizes for the stored/filtered params.
_RECYCLER_TEMP_BUCKET = float(_gw_os.getenv("RECYCLER_TEMP_BUCKET", "0.1"))
_RECYCLER_TOPP_BUCKET = float(_gw_os.getenv("RECYCLER_TOPP_BUCKET", "0.05"))
# sigmalang: an ADDITIONAL similarity signal, checked only AFTER the primary
# gate already accepted a hit. It can only REJECT a hit (turn it into a miss),
# never cause a wrong serve. Its cosine "doesn't reliably discriminate topic at
# this dimension" (per sigma_core.sigmalang) and 0.4 is permissive -- it rarely
# rejects anything. OFF BY DEFAULT (2026-07-23): after step-4 the primary gate
# (0.99 threshold + digit guard + different-answer margin + param buckets) is
# strong enough that this backstop adds no discrimination, only a per-hit network
# call to the sigmalang service on the hot path. Set SIGMALANG_GATE_ENABLED=true
# to re-enable it as a logged extra signal.
_SIGMALANG_ENABLED = _gw_os.getenv("SIGMALANG_GATE_ENABLED", "false").lower() not in ("0", "false", "no")
_SIGMALANG_THRESHOLD = float(_gw_os.getenv("SIGMALANG_THRESHOLD", "0.4"))
_QDRANT_URL = _gw_os.getenv("QDRANT_URL", "http://localhost:6333")
_SIGMA_INDEX_URL = _gw_os.getenv("SIGMA_INDEX_URL", "http://localhost:8200")
# sigma-index shadow dual-write: OFF BY DEFAULT (2026-07-23). This mirrored every
# stored RSU into sigma-index's "token_recycler" namespace as a future-Qdrant-
# replacement experiment, but nothing ever read it back (lookup() reads Qdrant
# only; an ecosystem-wide search found zero runtime consumers of that namespace)
# and sigma-index has no delete path so the shadow grew unbounded. Disabled to
# drop a per-store HTTP call + failure surface from the hot path. Set
# SIGMA_INDEX_DUALWRITE=true to restore it if it is ever promoted to the read path.
_SIGMA_INDEX_DUALWRITE = _gw_os.getenv("SIGMA_INDEX_DUALWRITE", "false").lower() not in (
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
    # step-4 L2 retrieval-gate observability (A1: make the gate calibratable).
    "l2_expired_filtered": 0,   # candidates skipped as expired (age > TTL)
    "l2_expired_deleted": 0,    # expired RSUs deleted from Qdrant (delete-on-expire, bounded)
    "l2_rejected_lexical": 0,   # candidates rejected by the digit/date guard (R3)
    "l2_rejected_maxtok": 0,    # candidates rejected as max_tokens-incompatible (R1)
    "l2_rejected_ambiguous": 0, # lookups rejected as an ambiguous cloud (R2)
    "dedup_skipped_total": 0,   # store() calls skipped: a near-exact same-answer twin already existed (P1)
}


# ---------------------------------------------------------------------------
# L1 exact-match cache + completeness gate (Token Recycler, tier 1)
# ---------------------------------------------------------------------------
# The semantic L2 (Qdrant) recycler below matches prompts FUZZILY and keys only
# on (model, backend) -- so it can serve an answer generated under different
# sampling params, and its nearest neighbour can be a near-miss twin. This L1
# sits IN FRONT of it: an in-process, param-aware, EXACT-match cache. A request
# identical in (resolved model, backend, full message list, all sampling params)
# to a prior one is served from the byte-identical tuple, with zero embedding
# round-trip. Two wins: correctness (never a cross-param / fuzzy answer for an
# exact repeat -- the dominant automation pattern) and latency (skips the
# 0.4-1s embed HTTP that dominates L2 hit time). Pure Python + bounded LRU/TTL,
# so exact repeats keep being served even when Qdrant/embeddings are down.
_L1_ENABLED = _gw_os.getenv("RECYCLER_L1_ENABLED", "true").lower() not in ("0", "false", "no")
_L1_MAX_ENTRIES = int(_gw_os.getenv("RECYCLER_L1_MAX_ENTRIES", "2048"))
_L1_TTL = int(_gw_os.getenv("RECYCLER_L1_TTL_SECONDS", str(_RECYCLER_TTL)))


def _gw_answer_is_complete(finish_reason) -> bool:
    """Whether a generated answer is complete enough to cache.

    Only a natural stop -- or a backend that omits finish_reason entirely
    (null-equivalent) -- counts. Any early-termination reason must NOT be
    cached: most importantly "length" (max_tokens hit), where storing the
    accumulated text as if it were the whole answer would let a later
    exact-prompt hit replay the cut-off response (truncation-replay). Other
    non-"stop" reasons ("content_filter", "tool_calls", ...) are likewise not
    a plain complete text answer and are excluded. Applied at EVERY store site
    (L1 put, L2 store on the non-stream path, the streaming tee, and the
    /api/{generate,chat} path) so no truncated answer enters either layer.
    """
    return finish_reason in ("stop", None)


class _ExactMatchL1Cache:
    """In-process, param-aware, exact-match answer cache (Token Recycler L1).

    Keyed on sha256(resolved model + backend + full messages + all sampling
    params); see _gw_l1_key(). Bounded LRU with per-entry TTL and
    delete-on-read-expiry (an expired entry is removed, never used to veto a
    lookup). No external service -- this is the latency win and the
    Qdrant-independent availability win.

    Lock-free by design: every method is synchronous and contains no `await`,
    so under the single-threaded asyncio event loop a get()/put() read-modify-
    write can never interleave with another request handler's.
    """

    def __init__(self, max_entries: int, ttl: int):
        self._d = _GwOrderedDict()   # key -> (answer, stored_ts)
        self._max = max(1, max_entries)
        self._ttl = ttl
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired = 0

    def get(self, key: str):
        item = self._d.get(key)
        if item is None:
            self.misses += 1
            return None
        answer, ts = item
        if self._ttl > 0 and (_gw_time.time() - ts) > self._ttl:
            del self._d[key]            # delete-on-expire (no veto -- it's just gone)
            self.expired += 1
            self.misses += 1
            return None
        self._d.move_to_end(key)        # LRU: mark most-recently-used
        self.hits += 1
        return answer

    def put(self, key: str, answer: str) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = (answer, _gw_time.time())
        while len(self._d) > self._max:
            self._d.popitem(last=False)  # evict least-recently-used
            self.evictions += 1

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": True,
            "entries": len(self._d),
            "max_entries": self._max,
            "ttl_seconds": self._ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "evictions": self.evictions,
            "expired": self.expired,
        }


_L1_CACHE = _ExactMatchL1Cache(_L1_MAX_ENTRIES, _L1_TTL)


# ---------------------------------------------------------------------------
# L2 (semantic) retrieval-selection policy -- step 4 (Kimi-reviewed)
# ---------------------------------------------------------------------------
def _gw_bucket(value: float, size: float) -> str:
    """Canonical STRING bucket key for a sampling param, so near-identical values
    share a cache partition (temperature 0.70 and 0.72 -> '0.7'). Returned as a
    string, NOT a float, on purpose: Qdrant's payload match-value supports
    keyword/integer/bool but not float equality, so a float bucket would silently
    never match and collapse the L2 hit rate. store() and lookup() both key
    through this function, so their bucket strings are always identical."""
    if size <= 0:
        return f"{value:g}"
    return f"{round(round(value / size) * size, 6):g}"


_GW_DIGIT_RE = _gw_re.compile(r"\d+")


def _gw_digit_multiset(text: str):
    """Sorted multiset of digit runs (years, dates, quantities, versions)."""
    return tuple(sorted(_GW_DIGIT_RE.findall(text or "")))


def _gw_digits_compatible(a: str, b: str) -> bool:
    """R3 deterministic guard: embeddings are structurally blind to digit swaps
    ('won 2020' vs 'won 2024' can score >0.99), so a candidate whose prompt has a
    different digit/date multiset than the query is rejected regardless of score."""
    return _gw_digit_multiset(a) == _gw_digit_multiset(b)


@_gw_dataclass
class _GwSelectionOutcome:
    chosen: Any
    expired_ids: List[str]
    reason: str                                   # hit | no_candidates | no_survivors | ambiguous
    counts: Dict[str, int] = _gw_field(default_factory=dict)

    @property
    def allow_l2_store(self) -> bool:
        # On an ambiguous-reject the caller still populates L1 (this request's own
        # exact answer) but must NOT add a point to a cloud it couldn't disambiguate.
        return self.reason != "ambiguous"


def _gw_select_recyclable(candidates, query_prompt, now, ttl, exact_twin, margin,
                          request_max_tokens=None, max_deletes=8):
    """Choose the RSU to serve from a score-descending candidate list, or reject.

    Replaces the old top_k=1 + TTL-veto path:
      * EXPIRED candidates (age > ttl) are filtered out, not used to veto -- a
        fresh runner-up is still servable; their ids are returned (bounded) for
        delete-on-expire. Legacy RSUs default _created_ts=0 -> always expired.
      * R3 lexical guard: candidate prompt digit/date multiset must match query.
      * R1 max_tokens: a stored COMPLETE answer under budget S serves a request
        only if request budget >= S (else a fresh gen might truncate shorter).
      * R2 margin: accept best iff score >= exact_twin OR no survivor has a
        DIFFERENT answer OR best.score - first_different_answer.score >= margin.
        Measuring against the first *different-answer* candidate (not survivors[1])
        is essential -- same-answer duplicate twins must not read as ambiguous.
    """
    counts = {"expired_filtered": 0, "lexical_rejected": 0, "maxtok_rejected": 0}
    if not candidates:
        return _GwSelectionOutcome(None, [], "no_candidates", counts)

    def _budget(v):
        # Comparable token budget, or None for "absent -> no constraint". These
        # are distinct cases and must not be conflated:
        #   None    -> None   absent max_tokens: DON'T filter (legacy / param-agnostic
        #                     /api store with no budget recorded -- serve as before).
        #   <= 0    -> +inf   Ollama num_predict -1 "unlimited" / -2 "fill context":
        #                     an EXPLICIT unlimited budget (serve any complete answer;
        #                     an unlimited-stored answer is served to nothing smaller).
        #   non-int -> +inf   garbage value: never raise a TypeError on the `<` below.
        #   > 0     -> the int.
        # A raw `<` on the -1 sentinel would treat it as a tiny ordinal and INVERT
        # the gate (collapse hit-rate for unlimited requests; over-serve an
        # unlimited-stored answer to a small budget).
        if v is None:
            return None
        return float(v) if (isinstance(v, int) and not isinstance(v, bool) and v > 0) else float("inf")

    req_budget = _budget(request_max_tokens)
    expired_ids: List[str] = []
    survivors: List[Any] = []
    for c in candidates:
        meta = getattr(c, "metadata", None) or {}
        ts = meta.get("_created_ts", 0) or 0
        if ttl > 0 and (now - ts) > ttl:
            counts["expired_filtered"] += 1
            if len(expired_ids) < max_deletes:
                expired_ids.append(c.rsu_id)
            continue
        if not _gw_digits_compatible(query_prompt, getattr(c, "prompt", "")):
            counts["lexical_rejected"] += 1
            continue
        # R1 truncation-safety: a stored COMPLETE answer generated under budget S
        # is served only to a request whose budget >= S. Applied only when BOTH
        # budgets are known -- an absent budget on either side means "no constraint".
        stored_budget = _budget(meta.get("max_tokens"))
        if req_budget is not None and stored_budget is not None and req_budget < stored_budget:
            counts["maxtok_rejected"] += 1
            continue
        survivors.append(c)
    if not survivors:
        return _GwSelectionOutcome(None, expired_ids, "no_survivors", counts)
    best = survivors[0]
    if best.score >= exact_twin:
        return _GwSelectionOutcome(best, expired_ids, "hit", counts)
    diff = next((c for c in survivors[1:] if getattr(c, "answer", None) != best.answer), None)
    if diff is None or (best.score - diff.score) >= margin:
        return _GwSelectionOutcome(best, expired_ids, "hit", counts)
    return _GwSelectionOutcome(None, expired_ids, "ambiguous", counts)


@_gw_dataclass
class _GwRecyclerLookup:
    """lookup() result: answer (None on miss); score = cosine of the served
    candidate (for X-Cache-Score); allow_l2_store is False only on an
    ambiguous-reject (L1 is still populated, L2 is not)."""
    answer: Any = None
    score: Any = None
    allow_l2_store: bool = True


class _TokenRecyclerCache:
    """Glues SemanticCompressor + VectorBank + SelectiveRetriever into the
    answer-level cache the Ryzanstein gateway checks on generate/chat calls."""

    def __init__(self):
        qhost, qport = _gw_parse_host_port(_QDRANT_URL, default_port=6333)
        self.bank = VectorBank(host=qhost, port=qport, collection_name="rsu_bank", vector_size=768)
        self.retriever = SelectiveRetriever(self.bank, top_k=_RECYCLER_TOPK)
        self.compressor = SemanticCompressor(embed_fn=self._embed)
        self.sigmalang = SigmalangClient()
        self.hits = 0
        self.misses = 0
        self.sigmalang_rejected = 0
        self.sigmalang_last_score = None
        self.last_hit_score = None      # cosine of the most recent served L2 candidate (X-Cache-Score)

    async def _embed(self, text: str):
        result = await _ollama_embed([text], _RECYCLER_EMBED_MODEL)
        return result["data"][0]["embedding"]

    async def lookup(self, prompt: str, model: str,
                     temperature: Optional[float] = None, top_p: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> "_GwRecyclerLookup":
        """Semantic L2 lookup -> _GwRecyclerLookup (answer None on miss).

        Fetches top-K candidates (param-bucket-filtered in Qdrant) instead of the
        single nearest, then applies the step-4 gate via _gw_select_recyclable:
        expired-filter (+ bounded delete-on-expire), R3 digit guard, R1 max_tokens
        compat, R2 different-answer margin, EXACT_TWIN bypass. This replaces the
        old top_k=1 path where an expired-or-fuzzy nearest either vetoed the
        lookup or was served outright.
        """
        try:
            query_vec = await self.compressor.embed_query(prompt)
            # model + backend bind the served model/runtime; the param buckets
            # (step 3) keep a temp=0 deterministic answer from being served to a
            # temp=1 request. Absent params add no bucket clause -- legacy/native
            # callers still match on model+backend alone (and legacy RSUs without
            # a temp_bucket field simply won't match a param-filtered lookup).
            must = [
                {"key": "model", "match": {"value": model}},
                {"key": "backend", "match": {"value": BACKEND}},
            ]
            if temperature is not None:
                must.append({"key": "temp_bucket",
                             "match": {"value": _gw_bucket(temperature, _RECYCLER_TEMP_BUCKET)}})
            if top_p is not None:
                must.append({"key": "top_p_bucket",
                             "match": {"value": _gw_bucket(top_p, _RECYCLER_TOPP_BUCKET)}})
            candidates = await self.retriever.retrieve_candidates(
                query_vec, score_threshold=_RECYCLER_THRESHOLD,
                filter_dict={"must": must}, limit=_RECYCLER_TOPK,
            )
        except Exception as e:
            logger.warning(f"Token Recycler lookup failed (degrading to miss): {e}")
            return _GwRecyclerLookup(None)

        # Gate + selection run AFTER retrieval; guard them too so any
        # post-retrieval error (a malformed candidate, an unexpected type)
        # degrades to a miss instead of escaping lookup() -- lookup()'s "degrades
        # to a miss" contract must cover the whole operation, and the /api call
        # site historically relied on lookup() never raising.
        try:
            outcome = _gw_select_recyclable(
                candidates, prompt, _gw_time.time(), _RECYCLER_TTL,
                _RECYCLER_EXACT_TWIN, _RECYCLER_MARGIN,
                request_max_tokens=max_tokens, max_deletes=_RECYCLER_MAX_DELETES,
            )
            _GW_METRICS["l2_expired_filtered"] += outcome.counts.get("expired_filtered", 0)
            _GW_METRICS["l2_rejected_lexical"] += outcome.counts.get("lexical_rejected", 0)
            _GW_METRICS["l2_rejected_maxtok"] += outcome.counts.get("maxtok_rejected", 0)
            # delete-on-expire: bounded (max_deletes), best-effort, never fails lookup.
            for rid in outcome.expired_ids:
                try:
                    await self.bank.delete(rid)
                    _GW_METRICS["l2_expired_deleted"] += 1
                except Exception as e:
                    logger.debug(f"Token Recycler delete-on-expire failed (ignored): {e}")

            if outcome.chosen is None:
                self.misses += 1
                if outcome.reason == "ambiguous":
                    _GW_METRICS["l2_rejected_ambiguous"] += 1
                return _GwRecyclerLookup(None, allow_l2_store=outcome.allow_l2_store)

            chosen = outcome.chosen
            # sigmalang: an ADDITIONAL signal checked only after the gate above has
            # already accepted this candidate. Fails open -- an outage/error never
            # turns an accepted hit into a miss.
            if _SIGMALANG_ENABLED:
                try:
                    sig_score = await self.sigmalang.asimilarity(prompt, chosen.prompt)
                    self.sigmalang_last_score = sig_score
                    if sig_score < _SIGMALANG_THRESHOLD:
                        self.sigmalang_rejected += 1
                        self.misses += 1
                        return _GwRecyclerLookup(None)
                except Exception as e:
                    logger.debug(f"sigmalang gate check failed (failing open, hit stands): {e}")
            self.hits += 1
            self.last_hit_score = chosen.score
            return _GwRecyclerLookup(chosen.answer, score=chosen.score)
        except Exception as e:
            logger.warning(f"Token Recycler gate/selection failed (degrading to miss): {e}")
            return _GwRecyclerLookup(None)

    async def store(self, prompt: str, model: str, answer: str,
                    temperature: Optional[float] = None, top_p: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> None:
        try:
            # "backend" rides in metadata like "_created_ts"; VectorBank.store()
            # spreads rsu.metadata into the top-level Qdrant payload, so these
            # become real filterable fields lookup() matches against. The param
            # buckets (step 3) partition the cache so a temp=0 answer isn't served
            # to a temp=1 request; max_tokens is stored so lookup() can enforce
            # truncation-safety (serve a complete answer only to an
            # equal-or-larger budget). Absent params are simply not stored, so
            # older entries stay valid for param-agnostic (native) lookups.
            meta = {"_created_ts": _gw_time.time(), "backend": BACKEND}
            if temperature is not None:
                meta["temp_bucket"] = _gw_bucket(temperature, _RECYCLER_TEMP_BUCKET)
            if top_p is not None:
                meta["top_p_bucket"] = _gw_bucket(top_p, _RECYCLER_TOPP_BUCKET)
            if max_tokens is not None:
                meta["max_tokens"] = max_tokens
            rsu = await self.compressor.compress(prompt, answer, model, metadata=meta)

            # Dedup-on-store (P1): VectorBank.store() is a bare insert with no cap
            # and no dedup, so repeated near-identical misses (e.g. two
            # near-simultaneous requests for the same new prompt, or a param
            # combo that keeps missing L1) accumulate same-answer duplicate
            # twins forever. Cheap guard: if a near-exact twin (score >=
            # EXACT_TWIN, same model/backend/param-buckets as the L2 filter
            # already uses) with the BYTE-IDENTICAL answer already exists,
            # skip this insert. Best-effort and fail-open -- any error here
            # (embed already succeeded above) falls through to the normal
            # store, so dedup can only reduce redundant points, never block
            # a real cache write or turn a store into a failure.
            try:
                must = [
                    {"key": "model", "match": {"value": model}},
                    {"key": "backend", "match": {"value": BACKEND}},
                ]
                if "temp_bucket" in meta:
                    must.append({"key": "temp_bucket", "match": {"value": meta["temp_bucket"]}})
                if "top_p_bucket" in meta:
                    must.append({"key": "top_p_bucket", "match": {"value": meta["top_p_bucket"]}})
                twins = await self.retriever.retrieve_candidates(
                    rsu.embedding, score_threshold=_RECYCLER_EXACT_TWIN,
                    filter_dict={"must": must}, limit=1,
                )
                if twins and twins[0].answer == answer:
                    _GW_METRICS["dedup_skipped_total"] += 1
                    return
            except Exception as e:
                logger.debug(f"Token Recycler dedup check failed (ignored, storing normally): {e}")

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


def _gw_l1_key(model: str, request: "ChatCompletionRequest") -> str:
    """Canonical, param-aware exact-match key for the L1 cache.

    sha256 over canonical JSON of the identity-affecting request surface:
      model     -- the RESOLVED served model (the exact value L2 keys on), so
                   the two layers agree on what "model" means.
      backend   -- bound in like L2's filter: an L1 entry written under one
                   backend can never serve a request routed to another.
      messages  -- role+content, ORDER PRESERVED (message order is semantic;
                   sort_keys below sorts dict keys, never list elements),
                   content byte-exact -- whitespace/fuzzy matching is L2's job;
                   an "exact" cache that normalised content would reintroduce a
                   false-hit class.
      params    -- every field on the request EXCEPT model/messages (folded
                   above) and stream (transport, not output: one entry serves
                   both stream and non-stream callers, matching the synth/tee
                   design). Reflected from the request rather than hand-listed,
                   so any sampling field later added to ChatCompletionRequest
                   (seed, stop, response_format, ...) is folded in automatically:
                   a forgotten field only fragments the cache (a perf loss); it
                   can never silently serve a wrong-param answer.
    """
    dump = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    params = {k: v for k, v in dump.items() if k not in ("model", "messages", "stream")}
    key_obj = {
        "model": model,
        "backend": BACKEND,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "params": params,
    }
    canon = _gw_json.dumps(key_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # surrogatepass: message content can legitimately arrive carrying a lone
    # UTF-16 surrogate (e.g. automation JSON-encoding text sliced mid-emoji);
    # Pydantic accepts it, but a plain utf-8 encode would raise UnicodeEncodeError.
    # Pass it through to bytes deterministically -- this only ever feeds sha256,
    # so non-UTF-8 bytes are fine and the mapping stays injective (no key
    # collisions). The call site additionally wraps this whole computation in a
    # fail-open guard for any other error (see _gw_openai_chat_completions).
    return _gw_hashlib.sha256(canon.encode("utf-8", errors="surrogatepass")).hexdigest()


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
                                      top_p: float, recycler,
                                      l1_key: Optional[str] = None,
                                      allow_l2_store: bool = True) -> AsyncIterator[str]:
    """Real streaming passthrough for a cache MISS under stream=true.

    Forwards Ollama's own OpenAI-compat SSE lines to the client as they arrive
    (no buffering -- each line is yielded the moment it's read off the
    upstream response), while accumulating the assistant's `delta.content`
    fragments in the background. Once the upstream stream completes
    successfully AND the terminal chunk reports a complete finish_reason, the
    accumulated full answer is stored into L1 (exact) and the Token Recycler
    L2 (semantic) -- this happens AFTER the client has already received every
    chunk, so it never adds latency to the client's real-time streaming
    experience. If accumulation or a store call fails, the client-visible
    stream is unaffected (the exception is caught and logged, matching
    _TokenRecyclerCache.store's own fail-open contract).

    Truncation guard (two conditions, both required): completed_ok gates on the
    [DONE] sentinel (rules out aborted/partial streams -- client disconnect,
    upstream error mid-stream, generator closed early). But [DONE] alone is NOT
    proof of completeness: Ollama emits [DONE] after a max_tokens-truncated
    stream too, with a `finish_reason:"length"` terminal chunk preceding it.
    So we also capture the last non-null finish_reason and require
    _gw_answer_is_complete() -- otherwise a length-truncated stream would be
    stored as a full answer and replayed by a future exact/semantic hit.
    """
    accumulated: List[str] = []
    completed_ok = False
    last_finish_reason = None
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
                choice0 = (chunk.get("choices") or [{}])[0]
                delta = (choice0.get("delta") or {})
                content = delta.get("content")
                if content:
                    accumulated.append(content)
                fr = choice0.get("finish_reason")
                if fr is not None:
                    last_finish_reason = fr
            except Exception as e:
                logger.debug(f"chat/completions stream-tee: could not parse chunk for "
                             f"accumulation (client stream unaffected): {e}")
    finally:
        # Store only a stream that both completed ([DONE]) and ended on a
        # complete finish_reason -- never a length-truncated one (see docstring).
        if completed_ok and accumulated and _gw_answer_is_complete(last_finish_reason):
            full_answer = "".join(accumulated)
            if l1_key is not None:
                try:
                    _L1_CACHE.put(l1_key, full_answer)
                except Exception as e:
                    logger.warning(f"chat/completions stream-tee: L1 store failed "
                                  f"(ignored, client stream already completed): {e}")
            # allow_l2_store is False only when the miss came from an
            # ambiguous-reject lookup: L1 still gets this request's own exact
            # answer (above), but we must not add a point to the L2 cloud we
            # just failed to disambiguate.
            if recycler is not None and allow_l2_store:
                try:
                    await recycler.store(prompt_text, model, full_answer,
                                         temperature=temperature, top_p=top_p, max_tokens=max_tokens)
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

    Model cache key: model = _resolve_chat_model(request.model) -- the request's
    model when it is in ALLOWED_CHAT_MODELS, otherwise the served default
    (OLLAMA_MODEL). Ecosystem callers send many labels that all resolve to the one
    served model, so keying on the RESOLVED value (not raw request.model) keeps the
    cache from fragmenting across labels that hit the same upstream model, while
    still honoring an explicitly allowlisted model selection. Both the L1 exact key
    and the L2 filter bind this resolved model. (This docstring previously claimed
    the handler "ignored request.model entirely and always forwarded OLLAMA_MODEL";
    that was stale -- _ollama_chat is called with the resolved `model` below.)

    Param-awareness: sampling params (temperature/top_p/max_tokens) are part of the
    L1 exact key and are bucket-filtered in the L2 lookup + stored in the L2 RSU,
    so a temp=0 deterministic answer is never served to a temp=1 request, and a
    complete answer is only L2-served to an equal-or-larger max_tokens budget.
    """
    recycler = _get_recycler()
    prompt_text = _gw_extract_openai_prompt(request.messages)
    model = _resolve_chat_model(getattr(request, "model", ""))
    # L1 exact-match key: full param-aware tuple, independent of the semantic
    # recycler (so L1 still serves exact repeats even if Qdrant/embeddings are
    # down). None disables the L1 path entirely.
    # Fail-open: key computation runs on EVERY request before any hit check, so
    # it must never turn a request that would otherwise succeed into a 500. On
    # any error, disable L1 for this request and fall through to L2 / live
    # inference -- matching the fail-open contract every other cache op honors.
    l1_key = None
    if _L1_ENABLED:
        try:
            l1_key = _gw_l1_key(model, request)
        except Exception as e:
            logger.warning(f"chat/completions: L1 key computation failed, disabling L1 "
                          f"for this request (served via L2/live inference): {e}")

    if not request.stream:
        # L1 first -- exact (model, messages, params) match, zero embed round-trip.
        if l1_key is not None:
            l1_hit = _L1_CACHE.get(l1_key)
            if l1_hit is not None:
                _GW_METRICS["passthrough_total"] += 1
                return JSONResponse(
                    _gw_wrap_openai_cached_answer(req_id, model, l1_hit),
                    headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "hit",
                             "X-Cache-Layer": "l1-exact"},
                )
        allow_l2_store = True
        if recycler is not None and prompt_text:
            # lookup() has its own internal try/except (degrades to a miss), but
            # this call site wraps it too so /v1/chat/completions keeps working
            # even if a future refactor lets an exception through lookup().
            try:
                res = await recycler.lookup(
                    prompt_text, model, temperature=request.temperature,
                    top_p=request.top_p, max_tokens=request.max_tokens,
                )
            except Exception as e:
                logger.warning(f"chat/completions: recycler.lookup() raised, treating as "
                              f"miss (client response unaffected): {e}")
                res = _GwRecyclerLookup(None)
            if res.answer is not None:
                _GW_METRICS["passthrough_total"] += 1
                # Deliberately do NOT backfill L1 from an L2 (fuzzy) hit: that
                # would "bless" a semantic match into an exact-keyed fact.
                return JSONResponse(
                    _gw_wrap_openai_cached_answer(req_id, model, res.answer),
                    headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "hit",
                             "X-Cache-Layer": "l2-semantic",
                             "X-Cache-Score": (f"{res.score:.4f}" if res.score is not None else "na")},
                )
            # On an ambiguous-reject miss, still populate L1 below (this request's
            # own exact answer) but skip the L2 store -- don't deepen the cloud.
            allow_l2_store = res.allow_l2_store
        result = await _ollama_chat(
            request.messages, model, request.max_tokens,
            request.temperature, request.top_p, request.stream,
        )
        # Completeness gate: never cache a truncated answer into either layer.
        # A finish_reason of "length" (max_tokens hit) etc. is served to THIS
        # client but not stored, so a later exact/semantic hit can't replay the
        # cut-off text as if it were whole.
        choice0 = (result.get("choices") or [{}])[0]
        answer_text = (choice0.get("message") or {}).get("content", "")
        if answer_text and _gw_answer_is_complete(choice0.get("finish_reason")):
            if l1_key is not None:
                _L1_CACHE.put(l1_key, answer_text)
            if recycler is not None and prompt_text and allow_l2_store:
                # store() is expected to fail open internally, but the
                # already-successful Ollama answer must reach the client even if
                # a cache-store exception somehow escapes it anyway.
                try:
                    await recycler.store(prompt_text, model, answer_text,
                                         temperature=request.temperature,
                                         top_p=request.top_p, max_tokens=request.max_tokens)
                except Exception as e:
                    logger.warning(f"chat/completions: recycler.store() raised (ignored, "
                                  f"client response unaffected): {e}")
        _GW_METRICS["passthrough_total"] += 1
        return JSONResponse(result, headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "miss"})

    # stream=true
    # L1 exact-match hit -> synthesize SSE from the stored answer (same chunk
    # framing as the L2-hit path), zero embed round-trip.
    if l1_key is not None:
        l1_hit = _L1_CACHE.get(l1_key)
        if l1_hit is not None:
            _GW_METRICS["passthrough_total"] += 1
            return StreamingResponse(
                _gw_stream_openai_cached_answer(req_id, model, l1_hit),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "X-Served-By": "ryzanstein-gateway",
                         "X-Cache": "hit", "X-Cache-Layer": "l1-exact"},
            )
    allow_l2_store = True
    if recycler is not None and prompt_text:
        # F1: wrap lookup() in try/except like the non-stream path -- a stream
        # request must degrade to live inference, not 500, if lookup ever raises.
        try:
            res = await recycler.lookup(
                prompt_text, model, temperature=request.temperature,
                top_p=request.top_p, max_tokens=request.max_tokens,
            )
        except Exception as e:
            logger.warning(f"chat/completions(stream): recycler.lookup() raised, treating as "
                          f"miss (client stream unaffected): {e}")
            res = _GwRecyclerLookup(None)
        if res.answer is not None:
            _GW_METRICS["passthrough_total"] += 1
            return StreamingResponse(
                _gw_stream_openai_cached_answer(req_id, model, res.answer),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "X-Served-By": "ryzanstein-gateway",
                         "X-Cache": "hit", "X-Cache-Layer": "l2-semantic",
                         "X-Cache-Score": (f"{res.score:.4f}" if res.score is not None else "na")},
            )
        allow_l2_store = res.allow_l2_store

    _GW_METRICS["passthrough_total"] += 1
    return StreamingResponse(
        _gw_stream_and_tee_to_cache(
            req_id, model, prompt_text, request.messages,
            request.max_tokens, request.temperature, request.top_p, recycler,
            l1_key, allow_l2_store,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "X-Served-By": "ryzanstein-gateway", "X-Cache": "miss"},
    )


@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE"])
async def _ollama_api_gateway(path: str, request: _GwRequest):
    body_bytes = await request.body()

    if path.split("/", 1)[0] not in _GW_API_ALLOWED_PATHS:
        return _GwJSON(
            {"error": f"/api/{path} is not permitted through the gateway"},
            status_code=403,
        )

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
            # Native Ollama options carry the sampling params (num_predict is the
            # max_tokens equivalent); thread them so this path is param-aware too.
            _opts = body.get("options") if isinstance(body.get("options"), dict) else {}
            _temp = _opts.get("temperature")
            _top_p = _opts.get("top_p")
            _mt = _opts.get("num_predict")
            if recycler is not None and prompt_text:
                # Parity with the two /v1 call sites: never let a lookup() error
                # 500 an /api request (lookup already degrades to a miss internally;
                # this is defense-in-depth for any future refactor).
                try:
                    res = await recycler.lookup(prompt_text, model,
                                                temperature=_temp, top_p=_top_p, max_tokens=_mt)
                except Exception as e:
                    logger.warning(f"/api/{path}: recycler.lookup() raised, treating as miss "
                                  f"(client response unaffected): {e}")
                    res = _GwRecyclerLookup(None)
                if res.answer is not None:
                    _GW_METRICS["passthrough_total"] += 1
                    return _GwJSON(
                        _gw_wrap_cached_answer(path, model, res.answer),
                        headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "hit",
                                 "X-Cache-Layer": "l2-semantic",
                                 "X-Cache-Score": (f"{res.score:.4f}" if res.score is not None else "na")},
                    )
                _allow_l2 = res.allow_l2_store
                async with _gw_httpx.AsyncClient(timeout=_GW_UPSTREAM_TIMEOUT) as client:
                    upstream = await client.post(
                        f"{_resolve_upstream(model, _GW_OLLAMA_URL)}/api/{path}",
                        content=body_bytes,
                    )
                    upstream.raise_for_status()
                    resp_json = upstream.json()
                answer_text = _gw_extract_answer(path, resp_json)
                # Completeness gate: Ollama's native /api/{generate,chat} reports
                # "done_reason" ("stop" | "length" | ...). Don't store a
                # length-truncated answer into the shared L2 -- it could later be
                # served (semantically) to a /v1/chat/completions request too. Also
                # skip the store on an ambiguous-reject miss (don't deepen the cloud).
                if (answer_text and _allow_l2
                        and _gw_answer_is_complete(resp_json.get("done_reason"))):
                    await recycler.store(prompt_text, model, answer_text,
                                         temperature=_temp, top_p=_top_p, max_tokens=_mt)
                _GW_METRICS["passthrough_total"] += 1
                return _GwJSON(
                    resp_json,
                    headers={"X-Served-By": "ryzanstein-gateway", "X-Cache": "miss"},
                )

    # Generic transparent passthrough for everything else (tags, embed, ps, pull, streaming, ...)
    _generic_model = ""
    if body_bytes:
        try:
            _gbody = _gw_json.loads(body_bytes)
            if isinstance(_gbody, dict):
                _generic_model = str(_gbody.get("model", "") or "")
        except ValueError:
            pass
    target = f"{_resolve_upstream(_generic_model, _GW_OLLAMA_URL)}/api/{path}"
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
    # L1 is independent of the semantic recycler -- report it even when the L2
    # (Qdrant) side is disabled or failed to import.
    l1_stats = _L1_CACHE.stats() if _L1_ENABLED else {"enabled": False}
    if recycler is None:
        return {"enabled": False, "reason": _RECYCLER_IMPORT_ERROR or "disabled via env",
                "l1": l1_stats}
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
        "last_hit_score": recycler.last_hit_score,
        "sigmalang_gate_enabled": _SIGMALANG_ENABLED,
        "sigmalang_threshold": _SIGMALANG_THRESHOLD,
        "sigmalang_rejected": recycler.sigmalang_rejected,
        "sigmalang_last_score": recycler.sigmalang_last_score,
        "stores_total": _GW_METRICS["stores_total"],
        "store_failures_total": _GW_METRICS["store_failures_total"],
        "passthrough_total": _GW_METRICS["passthrough_total"],
        # step-4 gate telemetry (A1) -- reject-reason breakdown for calibration.
        "topk": _RECYCLER_TOPK,
        "margin": _RECYCLER_MARGIN,
        "exact_twin": _RECYCLER_EXACT_TWIN,
        "l2_expired_filtered": _GW_METRICS["l2_expired_filtered"],
        "l2_expired_deleted": _GW_METRICS["l2_expired_deleted"],
        "l2_rejected_lexical": _GW_METRICS["l2_rejected_lexical"],
        "l2_rejected_maxtok": _GW_METRICS["l2_rejected_maxtok"],
        "l2_rejected_ambiguous": _GW_METRICS["l2_rejected_ambiguous"],
        "dedup_skipped_total": _GW_METRICS["dedup_skipped_total"],
        "l1": l1_stats,
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
    # L1 exact-match cache (independent of the Qdrant L2 above -- reported even
    # when the semantic recycler is disabled).
    if _L1_ENABLED:
        _l1s = _L1_CACHE.stats()
        lines += [
            "# HELP ryzanstein_recycler_l1_hits_total L1 exact-match cache hits.",
            "# TYPE ryzanstein_recycler_l1_hits_total counter",
            f"ryzanstein_recycler_l1_hits_total {_l1s['hits']}",
            "# HELP ryzanstein_recycler_l1_misses_total L1 exact-match cache misses.",
            "# TYPE ryzanstein_recycler_l1_misses_total counter",
            f"ryzanstein_recycler_l1_misses_total {_l1s['misses']}",
            "# HELP ryzanstein_recycler_l1_entries Current entries held in the L1 cache.",
            "# TYPE ryzanstein_recycler_l1_entries gauge",
            f"ryzanstein_recycler_l1_entries {_l1s['entries']}",
            "# HELP ryzanstein_recycler_l1_evictions_total L1 LRU evictions.",
            "# TYPE ryzanstein_recycler_l1_evictions_total counter",
            f"ryzanstein_recycler_l1_evictions_total {_l1s['evictions']}",
        ]
    # step-4 L2 gate telemetry (A1): reject-reason + expiry breakdown for calibration.
    lines += [
        "# HELP ryzanstein_recycler_l2_expired_filtered_total L2 candidates skipped as expired.",
        "# TYPE ryzanstein_recycler_l2_expired_filtered_total counter",
        f"ryzanstein_recycler_l2_expired_filtered_total {_GW_METRICS['l2_expired_filtered']}",
        "# HELP ryzanstein_recycler_l2_expired_deleted_total Expired RSUs deleted from Qdrant.",
        "# TYPE ryzanstein_recycler_l2_expired_deleted_total counter",
        f"ryzanstein_recycler_l2_expired_deleted_total {_GW_METRICS['l2_expired_deleted']}",
        "# HELP ryzanstein_recycler_l2_rejected_lexical_total L2 candidates rejected by the digit/date guard.",
        "# TYPE ryzanstein_recycler_l2_rejected_lexical_total counter",
        f"ryzanstein_recycler_l2_rejected_lexical_total {_GW_METRICS['l2_rejected_lexical']}",
        "# HELP ryzanstein_recycler_l2_rejected_maxtok_total L2 candidates rejected as max_tokens-incompatible.",
        "# TYPE ryzanstein_recycler_l2_rejected_maxtok_total counter",
        f"ryzanstein_recycler_l2_rejected_maxtok_total {_GW_METRICS['l2_rejected_maxtok']}",
        "# HELP ryzanstein_recycler_l2_rejected_ambiguous_total L2 lookups rejected as an ambiguous cloud.",
        "# TYPE ryzanstein_recycler_l2_rejected_ambiguous_total counter",
        f"ryzanstein_recycler_l2_rejected_ambiguous_total {_GW_METRICS['l2_rejected_ambiguous']}",
        "# HELP ryzanstein_recycler_dedup_skipped_total store() calls skipped as a same-answer near-exact twin.",
        "# TYPE ryzanstein_recycler_dedup_skipped_total counter",
        f"ryzanstein_recycler_dedup_skipped_total {_GW_METRICS['dedup_skipped_total']}",
    ]
    # sigma-telemetry real metrics (latency histograms w/ p50/p95/p99). Fail-open.
    if _TEL is not None:
        try:
            _tel_text = _TEL.render().strip()
            if _tel_text:
                lines.append(_tel_text)
        except Exception:
            pass
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
