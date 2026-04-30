# ADR-043: Compression & Embedding Strategy

| Field       | Value                               |
| ----------- | ----------------------------------- |
| **Status**  | Proposed                            |
| **Date**    | 2025-07-17                          |
| **Authors** | Ryzanstein Core Team                |
| **Relates** | ADR-042 (Agent Memory Architecture) |

## Context

Several Rust crates in the Ryzanstein monorepo (`sigma-compress`,
`ann-hybrid`, `agentmem` bridge) contain **stub** implementations for
`get_embeddings()` and `health_check()`. These stubs return zero-vectors
and unconditional `Ok(true)` respectively, making semantic search and service
monitoring non-functional.

The platform needs real embeddings for vector similarity and honest health
checks to support the observability layer (`sigma-telemetry`).

## Decision

### 1. `get_embeddings()` — Real HTTP Calls

Replace stubs with blocking HTTP requests to the configured LLM backend.

```
Endpoint : POST /v1/embeddings
Model    : text-embedding-ada-002 (configurable)

Request body:
{
  "input": ["<text>"],
  "model": "<model_id>"
}

Response body:
{
  "data": [
    { "embedding": [<f32; 1536>] }
  ]
}
```

**Fallback:** If the HTTP call fails (timeout, network error, non-200),
generate a deterministic hash-based pseudo-embedding so callers never panic.

**Dependency:**

```toml
reqwest = { version = "0.11", features = ["blocking", "json"] }
```

### 2. `health_check()` — Honest Probes

```
Endpoint : GET /health
Timeout  : 2 seconds

Logic:
  200        → Ok(true)
  non-200    → Ok(false)
  timeout    → Ok(false)
  network err → Ok(false)
```

Remove the unconditional `Ok(true)` stub.

### 3. Embedding Dimensions

| Model                  | Dimensions | Size per vector |
| ---------------------- | ---------- | --------------- |
| text-embedding-ada-002 | 1536       | ~6 KB           |
| text-embedding-3-small | 1536       | ~6 KB           |
| local hash fallback    | 1536       | ~6 KB           |

All embeddings are normalized to the same dimensionality (1536) regardless of
source, ensuring consistent cosine-similarity calculations.

## Consequences

### Positive

- Real semantic similarity for memory search and compression dedup.
- `health_check()` reflects actual backend state—enables proper alerting.
- Hash fallback ensures graceful degradation when LLM backend is offline.
- `reqwest` blocking client is acceptable for desktop single-threaded context.

### Negative

- Each embedding call adds ~100–300 ms network latency.
- 1536-dimension embeddings consume ~6 KB each (10K entries ≈ 60 MB).
- `reqwest` adds compile-time dependency (~30s incremental).

### Risks

| Risk                            | Mitigation                           |
| ------------------------------- | ------------------------------------ |
| LLM backend unavailable         | Hash-based fallback, never panic     |
| High latency                    | Batch embedding calls where possible |
| Embedding drift on model change | Pin model version in config          |

## Alternatives Considered

| Alternative                   | Why Rejected                                     |
| ----------------------------- | ------------------------------------------------ |
| Local embedding model (ONNX)  | 50 MB+ binary; unacceptable for desktop bundle   |
| Sentence-transformers via FFI | Python↔Rust FFI complexity too high              |
| No embeddings (keyword only)  | Loses semantic search capability entirely        |
| Async reqwest                 | Unnecessary complexity for desktop single-thread |
