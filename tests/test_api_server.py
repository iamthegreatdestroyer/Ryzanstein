"""
Sprint 2–3 API Server Tests
============================

Covers:
  /health                         — liveness probe
  GET  /v1/models                 — model list
  GET  /v1/models/{id}            — model detail + 404
  POST /v1/chat/completions       — non-streaming + streaming
  POST /v1/embeddings             — single string + batch, shape check
  GET  /mcp/tools                 — manifest presence + schema
  POST /mcp/tools/call            — generate + embed dispatchers

All tests use TestClient (synchronous HTTPX); no real weights needed.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.api.server import app, MODEL_NAME, EMBED_DIM

client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body
    assert body["model"] == MODEL_NAME


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

def test_list_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == MODEL_NAME


def test_get_model_by_id():
    r = client.get(f"/v1/models/{MODEL_NAME}")
    assert r.status_code == 200
    assert r.json()["id"] == MODEL_NAME


def test_get_unknown_model_404():
    r = client.get("/v1/models/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

def test_chat_completion_non_streaming():
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "stream": False,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "choices" in body
    assert len(body["choices"]) == 1
    assert "content" in body["choices"][0]["message"]
    assert "usage" in body
    assert body["usage"]["total_tokens"] > 0


def test_chat_completion_with_system_message():
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "What is 2+2?"},
        ],
        "max_tokens": 8,
        "stream": False,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_sse():
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "stream": True,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    data_lines = [ln[6:] for ln in lines if ln.startswith("data: ")]
    assert len(data_lines) >= 2  # at least one content chunk + [DONE]
    assert data_lines[-1] == "[DONE]"


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------

def test_embeddings_single_string():
    r = client.post("/v1/embeddings", json={"input": "Hello, world!"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    emb = body["data"][0]["embedding"]
    assert len(emb) == EMBED_DIM, f"Expected {EMBED_DIM}-dim, got {len(emb)}"
    assert all(isinstance(v, float) for v in emb)


def test_embeddings_batch():
    texts = ["alpha", "beta", "gamma delta"]
    r = client.post("/v1/embeddings", json={"input": texts})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == len(texts)
    for i, item in enumerate(data):
        assert item["index"] == i
        assert len(item["embedding"]) == EMBED_DIM


def test_embeddings_normalised():
    """L2-normalised vectors should have unit norm (±1e-3 tolerance)."""
    import math
    r = client.post("/v1/embeddings", json={"input": "test vector normalisation"})
    emb = r.json()["data"][0]["embedding"]
    norm = math.sqrt(sum(v * v for v in emb))
    assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm:.4f}"


def test_embeddings_empty_input_400():
    r = client.post("/v1/embeddings", json={"input": []})
    assert r.status_code == 400


def test_embeddings_usage_populated():
    r = client.post("/v1/embeddings", json={"input": "count tokens"})
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"]


# ---------------------------------------------------------------------------
# /mcp/tools
# ---------------------------------------------------------------------------

def test_mcp_tools_list():
    r = client.get("/mcp/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "1.0"
    assert body["server_name"] == "ryzanstein-llm"
    assert "tools" in body
    names = {t["name"] for t in body["tools"]}
    assert "generate" in names
    assert "embed"    in names


def test_mcp_tools_have_required_schema_fields():
    tools = client.get("/mcp/tools").json()["tools"]
    for tool in tools:
        assert "name"          in tool
        assert "description"   in tool
        assert "input_schema"  in tool
        assert "output_schema" in tool


def test_mcp_call_generate():
    payload = {"name": "generate", "input": {"prompt": "hello", "max_tokens": 4}}
    r = client.post("/mcp/tools/call", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert "tokens_generated" in body
    assert body["tokens_generated"] >= 1


def test_mcp_call_embed():
    payload = {"name": "embed", "input": {"text": "sigma ecosystem"}}
    r = client.post("/mcp/tools/call", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "embedding" in body
    assert body["dim"] == EMBED_DIM


def test_mcp_call_unknown_tool_404():
    r = client.post("/mcp/tools/call", json={"name": "nonexistent", "input": {}})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /v1/glyphs (Sprint 1 router still mounted)
# ---------------------------------------------------------------------------

def test_glyphs_router_still_mounted():
    r = client.post("/v1/glyphs", json={"tokens": [0, 1, 2, 3, 4, 5, 6, 7]})
    # Either 200 (sigmalang present) or 503 (not installed) — not 404
    assert r.status_code in (200, 503)
