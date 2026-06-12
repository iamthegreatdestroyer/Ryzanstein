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
