# Ryot (Ryzanstein LLM) — Autonomous Session Brief

## Project Identity
- **Repo:** `iamthegreatdestroyer/Ryot`
- **Local path:** `S:\Ryot`
- **Language:** Python + C++ (T-MAC kernels)
- **Castle Layer:** Layer 4 — Storage & Inference (Core LLM Engine)
- **Status:** ✅ LIVE — v2.0.0, BitNet 1.58b + T-MAC + AVX-512 @ 55 tok/s
- **Mission:** CPU-First LLM inference engine for AMD Ryzanstein processors

## This Session's Goal
Ryot is already live. This session: **integration hardening + ecosystem wiring**.

### Sprint 1 — Verify Inference Still Works (Hour 1)
```
@APEX start the API server: python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
Test: curl http://localhost:8000/v1/chat/completions -d '{"model":"bitnet-7b","messages":[{"role":"user","content":"hi"}]}'
Verify: response received, tokens/sec logged.
If server fails to start: read logs, fix startup error.
```

### Sprint 2 — /v1/embeddings Endpoint (Hours 1–2)
```
@APEX verify POST /v1/embeddings exists and returns embedding vectors.
This endpoint is called by: sigma-compress (semantic dedup), sigma-index (HNSW), sigma-diff (scoring).
If missing: implement it using the loaded model's last hidden state as embedding.
Input: {"input": "code snippet", "model": "bitnet-7b"}
Output: {"data": [{"embedding": [...1024 floats], "index": 0}]}
Test: curl the endpoint and verify vector dimension.
```

### Sprint 3 — MCP Protocol Support (Hour 3)
```
@APEX verify the MCP tool_use endpoint: GET /mcp/tools → list available tools
If missing: wire MCP adapter that exposes inference as a "generate" tool.
Test with mcp-mesh: the mesh should be able to register Ryot as an agent.
```

### Sprint 4 — Docker Image + Documentation (Hour 4)
```
@FORGE run: docker build --target runtime -t ryzanstein-llm:latest .
Verify: docker run -p 8000:8000 ryzanstein-llm:latest → server starts.

@SCRIBE write ECOSYSTEM_INTEGRATION.md:
  - /v1/embeddings API reference (for sigma-compress, sigma-index, sigma-diff)
  - /v1/chat/completions reference (for agent use)
  - MCP integration guide (for mcp-mesh registration)
  - Required env vars: MODEL_PATH, RYZANSTEIN_PORT (default 8000)
```

## Done Criteria
- [ ] API server starts and responds to inference requests
- [ ] `/v1/embeddings` returns 1024-dim float vectors
- [ ] `docker build` succeeds
- [ ] `ECOSYSTEM_INTEGRATION.md` written with API reference
- [ ] MCP tools endpoint accessible

## Completion Signal
Commit "chore: ecosystem wiring for v2.0.0" and push.
