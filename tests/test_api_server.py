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
  POST /v1/chat/completions (ollama backend) — Token Recycler cache wiring:
      non-streaming hit/miss, streaming hit/miss with stream-and-tee, model
      cache-key normalization, store() failure counters

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


def test_embeddings_ollama_backend_uses_embed_model_not_chat_model(monkeypatch):
    """OLLAMA_MODEL (qwythos-9b / phi4-mini / ...) is a chat model and is not
    embedding-capable -- Ollama's /v1/embeddings 501s if asked to embed with
    one. This was the actual reason every real /v1/embeddings caller
    (sigma-compress, sigma-index, sigma-diff) always fell back to a
    local/non-semantic path: not that Ryzanstein was unreachable, but that
    every real call it forwarded errored upstream. Confirms the ollama
    backend path forwards with OLLAMA_EMBED_MODEL, never OLLAMA_MODEL."""
    import src.api.server as server

    monkeypatch.setattr(server, "BACKEND", "ollama")

    captured = {}

    async def fake_ollama_embed(texts, model):
        captured["model"] = model
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.0]}],
        }

    monkeypatch.setattr(server, "_ollama_embed", fake_ollama_embed)

    r = client.post("/v1/embeddings", json={"input": "route me correctly"})
    assert r.status_code == 200
    assert captured["model"] == server.OLLAMA_EMBED_MODEL
    assert captured["model"] != server.OLLAMA_MODEL


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


# ---------------------------------------------------------------------------
# /v1/chat/completions (ollama backend) — Token Recycler wiring
# ---------------------------------------------------------------------------
# Previously this route unconditionally forwarded to Ollama on every call,
# never checking or populating the Token Recycler cache the /api/{path}
# gateway already used, and hardcoded stream=False upstream regardless of
# what the request actually asked for. These tests cover the fix using a
# fake in-memory recycler double (no real Qdrant/Ollama) so they run fast and
# deterministically, matching this file's existing monkeypatch convention
# (see test_embeddings_ollama_backend_uses_embed_model_not_chat_model above).

class _FakeRecycler:
    """Minimal double for _TokenRecyclerCache: an in-memory {(model, prompt): answer}
    dict instead of real Qdrant/embedding calls. lookup() only "hits" on an
    EXACT (model, prompt) match, which is enough to test the wiring/shape
    without needing real embedding similarity."""

    def __init__(self):
        self._store: dict = {}
        self.lookups: list = []
        self.stores: list = []

    async def lookup(self, prompt: str, model: str):
        self.lookups.append((prompt, model))
        return self._store.get((model, prompt))

    async def store(self, prompt: str, model: str, answer: str) -> None:
        self.stores.append((prompt, model, answer))
        self._store[(model, prompt)] = answer


class _FailingStoreRecycler(_FakeRecycler):
    """Recycler double whose store() always raises, to exercise store()'s own
    failure path indirectly through the /v1/chat/completions handler (the
    handler must not let a cache-store exception break the client response)."""

    async def store(self, prompt: str, model: str, answer: str) -> None:
        raise RuntimeError("simulated cache backend outage")


def _patch_ollama_backend(monkeypatch, server, recycler=None, reply="Hello from Ollama"):
    """Shared setup: BACKEND=ollama, a fake _get_recycler(), and a fake
    _ollama_chat() that returns a fixed OpenAI-shaped response without any
    real network call."""
    monkeypatch.setattr(server, "BACKEND", "ollama")
    monkeypatch.setattr(server, "_get_recycler", lambda: recycler)

    async def fake_ollama_chat(messages, model, max_tokens, temperature, top_p, stream):
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    monkeypatch.setattr(server, "_ollama_chat", fake_ollama_chat)
    return server


def test_chat_completions_ollama_cache_miss_then_hit(monkeypatch):
    """First call with a given prompt is a cache miss (real _ollama_chat call,
    then store()); an identical second call is a cache hit served from the
    fake recycler, with X-Cache reflecting each outcome."""
    import src.api.server as server

    recycler = _FakeRecycler()
    _patch_ollama_backend(monkeypatch, server, recycler=recycler)

    payload = {
        "model": "ryzanstein-bitnet-7b",
        "messages": [{"role": "user", "content": "unique cache-wiring test prompt"}],
        "max_tokens": 8,
        "stream": False,
    }

    r1 = client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200
    assert r1.headers["x-cache"] == "miss"
    assert r1.json()["choices"][0]["message"]["content"] == "Hello from Ollama"
    assert len(recycler.stores) == 1, "a successful miss must call recycler.store() exactly once"

    r2 = client.post("/v1/chat/completions", json=payload)
    assert r2.status_code == 200
    assert r2.headers["x-cache"] == "hit"
    assert r2.json()["choices"][0]["message"]["content"] == "Hello from Ollama"
    # Cache hits skip real inference -- usage is reported as zero, not fabricated.
    assert r2.json()["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_chat_completions_ollama_cache_key_uses_served_model_not_request_model(monkeypatch):
    """Verified real ecosystem callers send differing model labels for what
    resolves to the same served OLLAMA_MODEL on this box (myceloforge:
    "ryzanstein-bitnet-7b", NEURECTOMY spectrum-workspace: "ryot-bitnet-7b",
    sigma-index inference_client.go: "ryzanstein-bitnet-3b") -- this endpoint
    already ignored request.model entirely and always forwarded OLLAMA_MODEL.
    The cache key must therefore also key on OLLAMA_MODEL, or these callers
    would never cross-hit each other despite hitting the identical served
    model. Confirms a hit even when the second call's request.model differs
    from the first."""
    import src.api.server as server

    recycler = _FakeRecycler()
    _patch_ollama_backend(monkeypatch, server, recycler=recycler)

    prompt = {"role": "user", "content": "same prompt, different caller model label"}

    r1 = client.post("/v1/chat/completions", json={
        "model": "ryzanstein-bitnet-7b", "messages": [prompt], "max_tokens": 8, "stream": False,
    })
    assert r1.headers["x-cache"] == "miss"

    r2 = client.post("/v1/chat/completions", json={
        "model": "ryot-bitnet-3b-totally-different-label", "messages": [prompt],
        "max_tokens": 8, "stream": False,
    })
    assert r2.headers["x-cache"] == "hit", (
        "expected a cross-hit keyed on OLLAMA_MODEL regardless of request.model"
    )
    # Both lookups must have been keyed on the served model, never the
    # caller-supplied label.
    assert all(model == server.OLLAMA_MODEL for _, model in recycler.lookups)


def test_chat_completions_ollama_respects_actual_stream_flag(monkeypatch):
    """Previously _ollama_chat's payload hardcoded "stream": False and the
    caller's request.stream was never consulted for the ollama backend at
    all -- every request, streaming or not, took the non-streaming path.
    stream=false must still return a normal JSON chat.completion (not SSE)."""
    import src.api.server as server

    recycler = _FakeRecycler()
    _patch_ollama_backend(monkeypatch, server, recycler=recycler)

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "non-streaming request"}],
        "max_tokens": 8, "stream": False,
    })
    assert r.status_code == 200
    assert "text/event-stream" not in r.headers["content-type"]
    assert r.json()["object"] == "chat.completion"


def test_chat_completions_ollama_streaming_cache_miss_tees_into_store(monkeypatch):
    """stream=true on a cache MISS must still forward as a real stream AND
    accumulate the full answer into recycler.store() once the stream
    completes, without buffering the client's real-time chunks (the SSE
    body must contain more than one data: line, and store() must have been
    called with the reassembled full text)."""
    import src.api.server as server

    recycler = _FakeRecycler()
    monkeypatch.setattr(server, "BACKEND", "ollama")
    monkeypatch.setattr(server, "_get_recycler", lambda: recycler)

    async def fake_ollama_chat_stream(messages, model, max_tokens, temperature, top_p):
        for word in ("Hello", " streaming", " world"):
            chunk = {
                "id": "chatcmpl-fakestream", "object": "chat.completion.chunk", "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            }
            yield f"data: {__import__('json').dumps(chunk)}"
        yield "data: [DONE]"

    monkeypatch.setattr(server, "_ollama_chat_stream", fake_ollama_chat_stream)

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "streaming cache-miss test prompt"}],
        "max_tokens": 8, "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert r.headers["x-cache"] == "miss"

    data_lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert len(data_lines) >= 3, "expected the real per-token chunks to pass through, not be buffered"
    assert data_lines[-1] == "data: [DONE]"

    assert len(recycler.stores) == 1, "the completed stream must tee exactly one store() call"
    stored_prompt, stored_model, stored_answer = recycler.stores[0]
    assert stored_answer == "Hello streaming world", (
        "accumulated delta.content fragments must reassemble in order, exactly matching "
        "what the client received"
    )


def test_chat_completions_ollama_streaming_cache_hit_synthesizes_sse(monkeypatch):
    """stream=true on a cache HIT must synthesize a proper SSE/NDJSON response
    from the cached answer (not require re-hitting Ollama at all)."""
    import src.api.server as server

    recycler = _FakeRecycler()
    monkeypatch.setattr(server, "BACKEND", "ollama")
    monkeypatch.setattr(server, "_get_recycler", lambda: recycler)

    async def _unused_stream(*a, **kw):
        raise AssertionError("a cache HIT must not call _ollama_chat_stream at all")
        yield  # pragma: no cover - unreachable, makes this a generator function

    monkeypatch.setattr(server, "_ollama_chat_stream", _unused_stream)

    # Pre-seed the fake cache directly so this is a hit on the first real call.
    recycler._store[(server.OLLAMA_MODEL, "user: pre-seeded cached prompt")] = "cached answer text"

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "pre-seeded cached prompt"}],
        "max_tokens": 8, "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert r.headers["x-cache"] == "hit"

    data_lines = [ln[len("data: "):] for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"
    import json as _json
    reassembled = "".join(
        _json.loads(ln).get("choices", [{}])[0].get("delta", {}).get("content", "")
        for ln in data_lines[:-1]
    )
    assert reassembled == "cached answer text"


def test_chat_completions_ollama_streaming_miss_does_not_store_on_client_disconnect(monkeypatch):
    """A stream that is never fully consumed (client disconnect, generator
    closed early -- simulated here by only reading part of the body) must
    NOT store a truncated partial answer into the cache: that would poison
    future exact-prompt hits with a cut-off response."""
    import src.api.server as server

    recycler = _FakeRecycler()
    monkeypatch.setattr(server, "BACKEND", "ollama")
    monkeypatch.setattr(server, "_get_recycler", lambda: recycler)

    async def fake_ollama_chat_stream_no_done(messages, model, max_tokens, temperature, top_p):
        # Deliberately never yields the [DONE] sentinel, simulating an
        # aborted/interrupted upstream stream.
        chunk = {
            "id": "chatcmpl-fakestream2", "object": "chat.completion.chunk", "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
        }
        yield f"data: {__import__('json').dumps(chunk)}"

    monkeypatch.setattr(server, "_ollama_chat_stream", fake_ollama_chat_stream_no_done)

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "aborted stream test prompt"}],
        "max_tokens": 8, "stream": True,
    })
    assert r.status_code == 200
    assert len(recycler.stores) == 0, "an incomplete stream (no [DONE]) must never be cached"


def test_chat_completions_ollama_store_failure_does_not_break_response(monkeypatch):
    """A cache-store failure (compress/embed/Qdrant exception) must be fully
    fail-open: the client-facing chat completion still succeeds. Exercises
    the /v1/chat/completions call site's own try/except around
    recycler.store(), independent of _TokenRecyclerCache.store()'s internal
    exception handling (covered separately at the unit level below)."""
    import src.api.server as server

    recycler = _FailingStoreRecycler()
    _patch_ollama_backend(monkeypatch, server, recycler=recycler)

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "store-failure fail-open test prompt"}],
        "max_tokens": 8, "stream": False,
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello from Ollama"


# ---------------------------------------------------------------------------
# _TokenRecyclerCache.store() — failure observability (unit-level)
# ---------------------------------------------------------------------------
# Previously store() swallowed every exception with only a log line and no
# counter, so a persistently-failing cache write was invisible short of
# grepping logs. These exercise the real _TokenRecyclerCache class (not the
# fake double above) against the module-level _GW_METRICS counters.

def test_recycler_store_failure_increments_counter_and_logs(monkeypatch, caplog):
    import src.api.server as server

    cache = server._TokenRecyclerCache.__new__(server._TokenRecyclerCache)

    class _BoomCompressor:
        async def compress(self, *a, **kw):
            raise ConnectionError("simulated embedding backend outage")

    cache.compressor = _BoomCompressor()

    before = server._GW_METRICS["store_failures_total"]
    with caplog.at_level("WARNING"):
        import asyncio
        asyncio.run(cache.store("p", "m", "a"))

    assert server._GW_METRICS["store_failures_total"] == before + 1
    assert any("Token Recycler store failed" in rec.message for rec in caplog.records)


def test_recycler_sigma_index_dualwrite_failure_logs_at_warning(monkeypatch, caplog):
    """The sigma-index shadow dual-write previously failed at DEBUG level
    (invisible by default). Confirms it now logs at WARNING, and confirms a
    dual-write failure never prevents the primary Qdrant store from counting
    as a success (stores_total still increments)."""
    import src.api.server as server

    monkeypatch.setattr(server, "_SIGMA_INDEX_DUALWRITE", True)

    cache = server._TokenRecyclerCache.__new__(server._TokenRecyclerCache)

    class _FakeRSU:
        id = "fake-id"
        embedding = [0.0]
        prompt = "p"

    class _OkCompressor:
        async def compress(self, *a, **kw):
            return _FakeRSU()

    class _OkBank:
        async def store(self, rsu):
            return rsu.id

    cache.compressor = _OkCompressor()
    cache.bank = _OkBank()

    before_stores = server._GW_METRICS["stores_total"]

    class _FakeAsyncClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            raise ConnectionError("simulated sigma-index outage")

    monkeypatch.setattr(server._gw_httpx, "AsyncClient", _FakeAsyncClient)

    with caplog.at_level("WARNING"):
        import asyncio
        asyncio.run(cache.store("p", "m", "a"))

    assert server._GW_METRICS["stores_total"] == before_stores + 1, (
        "primary Qdrant store succeeded, so this must still count as a store even "
        "though the best-effort sigma-index dual-write failed"
    )
    assert any("sigma-index shadow dual-write failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# /v1/recycler/stats and /metrics — degrade gracefully on a Qdrant outage
# ---------------------------------------------------------------------------
# Previously both endpoints called recycler.bank.count() with no exception
# handling, so a Qdrant outage made the diagnostic endpoints themselves 500
# instead of reporting a degraded reading -- discovered manually while
# verifying the new counters against a throwaway instance with a
# deliberately-broken QDRANT_URL.

def test_recycler_stats_degrades_on_qdrant_outage(monkeypatch):
    import src.api.server as server

    recycler = _FakeRecycler()
    recycler.hits, recycler.misses = 0, 0
    recycler.sigmalang_rejected, recycler.sigmalang_last_score = 0, None

    class _BoomBank:
        async def count(self):
            raise ConnectionError("simulated Qdrant outage")

    recycler.bank = _BoomBank()
    monkeypatch.setattr(server, "_get_recycler", lambda: recycler)

    r = client.get("/v1/recycler/stats")
    assert r.status_code == 200, "a Qdrant outage must not 500 the stats endpoint itself"
    assert r.json()["rsu_count"] == -1


def test_metrics_degrades_on_qdrant_outage(monkeypatch):
    import src.api.server as server

    recycler = _FakeRecycler()
    recycler.hits, recycler.misses = 0, 0

    class _BoomBank:
        async def count(self):
            raise ConnectionError("simulated Qdrant outage")

    recycler.bank = _BoomBank()
    monkeypatch.setattr(server, "_get_recycler", lambda: recycler)

    r = client.get("/metrics")
    assert r.status_code == 200, "a Qdrant outage must not 500 the /metrics scrape itself"
    assert "ryzanstein_recycler_rsu_count -1" in r.text
