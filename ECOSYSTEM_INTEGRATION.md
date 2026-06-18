# Ryzanstein v3.0.0 — Ecosystem Integration Guide

## Castle Layer Architecture Position

Ryzanstein is **Layer 4 (Storage & Inference)** in the Sigma Ecosystem Castle Layer Architecture.

```
Layer 7  sigma-harvest, audioshift, AutoAG-CommGateway, ...
Layer 6  DePIN-Orcha
Layer 5  sigma-diff, NSTG, NLCI, QADR, PTL, SED, NEURECTOMY
Layer 4  [Ryzanstein / RYZEN-LLM]  ←─ this repo
         sigma-index, sigma-compress, sigma-telemetry, AlgoSmash, sigmalang
Layer 3  NexusZero-Protocol, vault-git, sigmavault, PhantomMesh-VPN
Layer 2  mcp-mesh, HyperBox
Layer 1  sigmavault-nas-os
```

---

## What Ryzanstein Provides

| Component | Description |
|-----------|-------------|
| `cpu-infer` (Rust) | GGUF v2/v3 parser, multi-layer Llama transformer, GQA+RoPE+SwiGLU |
| AVX-512 VNNI kernels | `bitnet_matvec`, `dot_i8`, 2-bit ternary packing |
| SemanticKVCache | Chunk-aligned (256-token) prefix caching, SHA-256 keyed, LRU |
| FastAPI server | OpenAI-compatible `/v1/chat/completions`, Bearer auth, rate limiting |
| ZK audit trail | Schnorr sigma protocol proofs over inference Merkle roots |
| MCP endpoint | JSON-RPC 2.0 tool bridge at `/mcp` |

---

## API Endpoints

### Authentication

All `/v1/*` and `/mcp` endpoints require:
```
Authorization: Bearer <api-key>
```
Set valid keys via `API_KEYS=key1,key2` env var or `API_KEYS_FILE=/path/to/keys`.
Disable for dev: `AUTH_DISABLED=1`.

### Rate Limiting

Sliding-window per API key.  
Defaults: 60 requests / 60 seconds.  
Configure: `RATE_LIMIT_REQUESTS=N`, `RATE_LIMIT_WINDOW_S=N`.

### Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server + engine health |
| GET | `/health/live` | Kubernetes liveness probe |
| GET | `/health/ready` | Kubernetes readiness probe |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| POST | `/v1/embeddings` | Text embeddings |
| POST | `/v1/audit` | ZK proof of inference audit trail |
| POST | `/mcp` | MCP JSON-RPC 2.0 tool bridge |

---

## ZK Audit Integration (`/v1/audit`)

The audit endpoint produces a Schnorr sigma protocol zero-knowledge proof
binding a Merkle root over inference entries to the request ID, without
revealing prompt content.

**Protocol details:**
- Group: RFC 3526 Group 5 (1536-bit safe prime, `g=2`)
- Fiat-Shamir transform: `c = SHA256(g^r mod p || message)`
- Proof: `{h, t, c, s}` where `s = r + c·x`, verifiable as `g^s ≡ t·h^c (mod p)`
- Merkle tree: SHA-256 leaf hashing, balanced binary tree, duplicate-last padding
- Hash chain: each audit entry chains `SHA256(id|ts|action|desc|actor|prev_hash)`

**The Rust `zkaudit` crate** implements the same protocol in `dependencies/zkaudit/`.
Python and Rust proofs are cross-verifiable (same prime, same hash function, same encoding).

```bash
# Example audit call
curl -s -X POST http://localhost:8000/v1/audit \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"request_id":"abc123","model":"ryzanstein-7b","prompt_tokens":42,"output_tokens":80}'
```

**Response:**
```json
{
  "request_id": "abc123",
  "merkle_root": "<64-char hex>",
  "entry_count": 1,
  "chain_integrity": true,
  "zk_proof": {"commitment": "...", "challenge": "...", "response": "...", "t_point": "..."},
  "zk_verified": true
}
```

---

## MCP Integration (`/mcp`)

The `/mcp` endpoint implements [Model Context Protocol](https://modelcontextprotocol.io) JSON-RPC 2.0.

**Built-in tools:**

| Tool | Description |
|------|-------------|
| `health` | Returns server and engine health status |
| `list_models` | Lists loaded inference models |

**Example: tools/call**
```bash
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"method":"tools/call","id":1,"params":{"name":"health","arguments":{}}}'
```

**Extending with custom tools:**
```python
from RYZEN_LLM.src.api.server import _mcp

def my_search_tool(args: dict) -> dict:
    return {"results": ["..."]}

_mcp.register_tool(
    name="search",
    description="Search the local corpus",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    handler=my_search_tool,
)
```

---

## Connecting to Other Sigma Layers

### Layer 3 — sigmavault / PhantomMesh-VPN

Ryzanstein's API key auth integrates with the sigmavault key store:
- Store API keys in sigmavault (Kyber-1024 encrypted vault)
- Export plaintext keys into `API_KEYS_FILE` at container start
- Keys never touch source or git history

### Layer 2 — mcp-mesh

The `/mcp` endpoint is discoverable by mcp-mesh for automatic agent registration.
Add to mcp-mesh's service registry:
```yaml
services:
  ryzanstein:
    url: http://ryzanstein-api:8000/mcp
    auth: bearer
    tools: [health, list_models]
```

### Layer 5 — sigma-telemetry

Ryzanstein emits OpenTelemetry spans via Jaeger.  
Set `TRACING_JAEGER_HOST=jaeger` to enable.  
The `/v1/audit` ZK proof receipt can be forwarded to sigma-telemetry for on-chain anchoring.

### Layer 7 — sigma-harvest

sigma-harvest can call Ryzanstein for local LLM inference instead of cloud APIs:
```python
import httpx
async with httpx.AsyncClient() as client:
    resp = await client.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "ryzanstein-7b", "messages": [{"role": "user", "content": prompt}]},
    )
```

---

## Docker Deployment

```bash
# Build the image
docker-compose build ryzanstein-api

# Start with auth
RYZANSTEIN_API_KEYS="key1,key2" docker-compose up -d ryzanstein-api

# Dev mode (no auth)
AUTH_DISABLED=1 docker-compose --profile dev up ryzanstein-api-dev
```

The full observability stack (Prometheus, Grafana, Jaeger, Qdrant) starts with:
```bash
docker-compose up -d
```

---

## Performance Targets (v3.0.0)

| Model | Hardware | Target tok/s | Method |
|-------|----------|-------------|--------|
| BitNet 700M | i7-13700H | ≥119 | AVX-512 VNNI ternary |
| Llama 7B Q4_K_M | i7-13700H | ≥18.75 | VNNI + 2-bit packing |
| Llama 70B Q4_K_M | i7-13700H | ≥2.44 | rayon parallel batching |

Source: arxiv:2410.16144v1 — confirmed by 2-of-3 adversarial verification.

---

## Sprint Completion Summary

| Sprint | What Was Built | Status |
|--------|---------------|--------|
| S1 | Fix 4 crash bugs (lock names, asyncio.Queue thread safety) | ✅ |
| S2 | Real GGUF parser + Llama transformer (GQA/RoPE/SwiGLU/KV cache) | ✅ |
| S3 | SemanticKVCache (chunk-aligned prefix caching, LRU) | ✅ |
| S4 | AVX-512 VNNI + BitNet 2-bit ternary kernels | ✅ |
| S5 | Bearer auth + sliding-window rate limiter + Docker | ✅ |
| S6 | ZK audit trail + MCP endpoint + this document | ✅ |
| Sprint 1 (v3.1) | Glyph-native KV cache (HybridKVCache + GlyphKVCache), `/v1/glyphs` API | ✅ |
| Sprint 2 (v3.1) | `/v1/embeddings` (1024-dim, L2-norm, last hidden state), `src/api/server.py` | ✅ |
| Sprint 3 (v3.1) | `/mcp/tools` manifest + `/mcp/tools/call` dispatcher, Linux Dockerfile | ✅ |

---

## Sprint 2–3 (v3.1) — Detailed Reference

### `/v1/embeddings` — Embedding Vectors

Generates 1024-dim float vectors from the model's mean-pooled last hidden state.
Vectors are L2-normalised so that `dot(a, b) == cosine_similarity(a, b)`.

**Request:**
```json
{"input": "code or text", "model": "ryzanstein-bitnet-7b"}
```

Batch form:
```json
{"input": ["doc one", "doc two"]}
```

**Response:**
```json
{
  "object": "list",
  "model": "ryzanstein-bitnet-7b",
  "data": [{"object": "embedding", "index": 0, "embedding": [...1024 floats]}],
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

**Sigma consumers and their call patterns:**

| Repo | What it does | Endpoint |
|------|-------------|----------|
| `sigma-compress` | Semantic dedup before compression | `POST /v1/embeddings` |
| `sigma-index` | Feed HNSW index (ChromaDB) | `POST /v1/embeddings` |
| `sigma-diff` | Cosine similarity scoring | `POST /v1/embeddings` (pair) |

### `/mcp/tools` — Tool Manifest

Returns the MCP-compatible tool manifest for mcp-mesh auto-registration.

```
GET /mcp/tools      → {schema_version, server_name, server_version, tools: [...]}
POST /mcp/tools/call → {name, input} → tool-specific response
```

**Available tools:**

| Tool | Input | Output |
|------|-------|--------|
| `generate` | `{prompt, max_tokens, temperature}` | `{text, tokens_generated}` |
| `embed` | `{text}` | `{embedding, dim}` |
| `encode_glyphs` | `{tokens: [int]}` | `{glyph_hex, compression_ratio, glyph_count}` |

### Docker (Linux, Sprint 3)

The Dockerfile was rewritten as a clean Linux multi-stage build.

```bash
docker build --target runtime -t ryzanstein-llm:latest .
docker run -p 8000:8000 ryzanstein-llm:latest
```

Runtime image: Python 3.11-slim + CPU-only torch ≈ 1.6 GB.

---

## Innovation Addendum — v3.3.0

### Innovation #2 — Mamba-Glyph Fusion (`src/models/glyph_mamba.py`)

**Architecture insight:** Standard Mamba computes selective parameters (B, C, Δ)
from d_model-dimensional token embeddings. Glyph-Mamba computes B, C, Δ from
d_glyph-dimensional glyph coordinate embeddings (d_glyph=32 << d_model=128+),
making the selectivity mechanism ask "which glyph subspace is relevant?" instead
of "which token matters?" — sub-linear in model dimension.

| Component | Description |
|-----------|-------------|
| `GlyphCoordinateEmbedding` | token_ids → primitive_ids → learned d_glyph coords (sigmalang or fallback) |
| `selective_scan_sequential` | Pure-PyTorch ZOH-discretized SSM scan, O(L) in sequence length |
| `GlyphSSMLayer` | Core: B_proj, C_proj, dt_proj take d_glyph inputs (not d_model) |
| `GlyphMambaBlock` | Full Mamba block: in_proj → conv1d → SSM → SiLU gate → out_proj + residual |
| `GlyphMambaModel` | End-to-end stackable model; drop-in backbone for DistributedServingEngine |

**Benchmark results (d_model=128, n_layers=4, d_glyph=32, CPU):**

| Metric | Value |
|--------|-------|
| GlyphMamba parameters | 596,224 (35% fewer than MHA baseline 921,344) |
| Glyph selectivity ratio | 8.6% of all params are glyph-selectivity projections |
| Empirical scaling exponent | **0.92** (linear O(L) — confirmed ✅) |
| Causal SSM correctness | Verified: output[t] independent of input[t+1] |
| Sigmalang integration | Tier-0/1/2 primitive mapping + semantic dedup (tokens 256≡384) |

**Tests:** 38 unit tests in `tests/test_glyph_mamba.py` (38/38 pass).

**Performance note:** The sequential Python scan (`selective_scan_sequential`)
has Python loop overhead per step, making absolute latency slower than PyTorch's
fused C++ SDPA on CPU. Swap for `mamba_ssm.selective_scan_cuda` on CUDA for
10-100× speedup. The O(L) linear scaling is architecturally correct and confirmed.

**Swap attention head usage:**
```python
from src.models.glyph_mamba import GlyphMambaBlock

# Replace attention in any Transformer block:
block = GlyphMambaBlock(d_model=256, d_glyph=32)
output = block(hidden_states, token_ids=token_ids)  # [B, L, d_model]
```

### Sprint Completion Table (updated)

| Sprint | Tag | Scope |
|--------|-----|-------|
| Sprint 1 | v3.0.0 | E2E tests, benchmarks, serving integration |
| Sprint 2 | v3.1.0 | FastAPI server, /v1/chat, /v1/embeddings, MCP |
| Sprint 3 | v3.1.0 | Dockerfile rewrite (Linux multi-stage) |
| Innovation #3 | v3.2.0 | Bidirectional Token Recycling (GlyphPriorPool) |
| Innovation #4 | v3.2.0 | Living Benchmarks as Probes (GlyphBenchmarkIndex) |
| Innovation #2 | v3.3.0 | Mamba-Glyph Fusion (GlyphSSMLayer, GlyphMambaModel) |
No CUDA, no Windows Server Core — builds anywhere.
