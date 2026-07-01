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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .glyphs import router as glyphs_router

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
    """Forward chat completion to Ollama's OpenAI-compat endpoint."""
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
async def chat_completions(request: ChatCompletionRequest):
    req_id = uuid.uuid4().hex[:12]

    if BACKEND == "ollama":
        result = await _ollama_chat(
            request.messages, OLLAMA_MODEL, request.max_tokens,
            request.temperature, request.top_p, request.stream,
        )
        return JSONResponse(result)

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
async def create_embeddings(request: EmbeddingRequest):
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
        return JSONResponse(await _ollama_embed(inputs, OLLAMA_MODEL))

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
async def mcp_call_tool(request: _McpToolCallRequest):
    """
    Execute an MCP tool call.  Dispatches to the matching FastAPI handler.
    """
    if request.name == "generate":
        prompt     = request.input.get("prompt", "")
        max_tokens = int(request.input.get("max_tokens", 256))
        ids        = _tokenize(prompt)
        gen        = _generate_tokens(ids, max_tokens)
        return {"text": _decode_tokens(gen), "tokens_generated": len(gen)}

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


# === Ollama-compatible gateway passthrough (Tier 0.1, 2026-07-01) =============
# Transparently reverse-proxies Ollama's native /api/* to the real ollama daemon
# so Ryzanstein :8000 is the single LLM entry point every consumer routes through
# (superset of ollama's API plus this server's /v1). Streaming preserved; adds an
# X-Served-By header so we can confirm Ryzanstein is on the hot path.
import os as _gw_os
import httpx as _gw_httpx
from fastapi import Request as _GwRequest
from fastapi.responses import StreamingResponse as _GwStreaming
from starlette.background import BackgroundTask as _GwBg

_GW_OLLAMA_URL = _gw_os.getenv(
    "GATEWAY_OLLAMA_URL", _gw_os.getenv("OLLAMA_URL", "http://localhost:11434")
)


@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE"])
async def _ollama_api_gateway(path: str, request: _GwRequest):
    target = f"{_GW_OLLAMA_URL}/api/{path}"
    body = await request.body()
    fwd = {k: v for k, v in request.headers.items()
           if k.lower() not in ("host", "content-length", "connection")}
    client = _gw_httpx.AsyncClient(timeout=None)
    upstream = await client.send(
        client.build_request(request.method, target, content=body, headers=fwd,
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
