# Ryzanstein v3.0.0 — Development Action Plan

**Basis:** Full codebase analysis + deep-research workflow (108 agents, arxiv/GitHub sources)  
**Date:** 2026-06-10  
**Scope:** 6 sprints (~23 working days) to produce a functional CPU-first LLM inference engine

---

## Current Reality

The CLAUDE.md claims "LIVE @ 55 tok/s" — the README shows every core feature unchecked. The codebase has a real SIMD kernel layer (AVX2/FMA), working quantization (Int8/Int4/Ternary), and genuine ZK cryptography — but no end-to-end inference path, four crash bugs that prevent startup, and a pseudoscientific token recycler.

This plan builds on what is real, replaces what is fake, and adds what is missing.

---

## Research Baseline (Confirmed Claims)

| Claim | Source | Vote |
|-------|--------|------|
| BitNet b1.58 on i7-13700H: 2.37x–6.17x speedup over FP16, 71.9%–82.2% energy reduction | arxiv:2410.16144v1 | 2-1 ✓ |
| bitnet.cpp x86 tok/s: 700M→119, 7B→18.75, 70B→2.44 | arxiv:2410.16144v1 | 2-1 ✓ |
| BitNet b1.58 2B4T: 0.4 GB non-embedding memory (vs 2 GB LLaMA 3.2 1B) | arxiv:2504.12285 | 1-1 ✓ |
| BitNet b1.58 speedups 2.37x–6.17x confirmed via GitHub | github.com/microsoft/BitNet | 1-1 ✓ |

Directional (sourced, adversarial verify failed due to session limits — treat as plausible, not confirmed):
- ChunkKV (arxiv:2502.00299): +8.7% precision at same compression, +26.5% throughput via chunk-aligned KV cache
- SemantiCache (arxiv:2603.14303): 2.61x decoding speedup via Greedy Seed-Based Clustering of KV cache tokens
- EVICPRESS (arxiv:2512.14946): 2.19x faster TTFT via joint KV compression + eviction
- llama.cpp speculative decoding: n-gram cache / simple / map (hash-map of context) / mod (shared pool) strategies
- SpecFormer (arxiv:2511.20340): non-autoregressive draft head, 1.70–1.81x wall-clock speedup
- mistral.rs: continuous batching on all devices including CPU; PagedAttention CUDA/Apple Silicon only
- rvLLM: Rust+CUDA native LLM serving, OpenAI-compatible — **GPU only (H100), not applicable**
- SGLang CPU backend: 6–14x TTFT on Intel Xeon 6 vs llama.cpp for MoE — **requires Python+Intel stack, not applicable**

---

## What to Keep

| File | What Is Real |
|------|-------------|
| `dependencies/cpu-infer/src/kernels.rs` | AVX2/FMA SIMD, dual-accumulator loop unrolling, correct horizontal reduction |
| `dependencies/cpu-infer/src/quantize.rs` | Working Int8/Int4/Ternary with round-trip tests |
| `dependencies/zkaudit/src/proof.rs` | Genuine Schnorr Sigma Protocol (Fiat-Shamir, RFC 3526 Group 5 1536-bit safe prime) |
| `dependencies/zkaudit/src/chain.rs` | SHA-256 hash chain with tamper detection |
| `dependencies/zkaudit/src/merkle.rs` | Standard Merkle tree with proof path generation |
| `src/serving/resilience.py` | Real circuit breaker FSM (CLOSED/OPEN/HALF_OPEN), graceful degrader, watchdog with exponential backoff |
| `src/distributed/orchestrator.py` | Sound PyTorch distributed scaffold (NCCL/gloo, ResourceAllocator, FailureRecoveryManager) |

## What to Replace

| File | Problem | Replacement |
|------|---------|-------------|
| `src/recycler/fractal_mycelium.py` | Multiplies vocabulary indices by Gaussian random; `_novelty_score` returns `min(distances) * -1`; disconnected from NLP | Chunk-aligned semantic KV cache (Sprint 3) |
| `dependencies/cpu-infer/src/engine.rs` tokenizer | 100 hardcoded words, dynamic growth; not real tokenization | tiktoken via PyO3 FFI or HuggingFace `tokenizers` crate (Sprint 2) |
| `src/serving/lockfree_logger.py` | `asyncio.Queue.get_nowait()` called from background OS thread — race condition/crash | `queue.SimpleQueue` (stdlib, thread-safe) (Sprint 1) |

---

## Sprint 1 — Critical Bug Fixes (Day 1)
**Unblocks:** everything. The engine crashes on import without these fixes.

### Bug 1: `distributed_serving.py:484` — HealthMonitor missing lock

`HealthMonitor.check_gpu_health()` calls `self.lock` which is never defined in `__init__`.

```python
# In HealthMonitor.__init__, add:
self.lock = asyncio.Lock()
```

### Bug 2: `distributed_serving.py:534` — MetricsCollector wrong lock name

`MetricsCollector.record_request()` calls `self.lock` but the attribute is `self.record_lock`.

```python
# Change every occurrence in record_request():
async with self.lock:      # wrong
async with self.record_lock:  # correct
```

### Bug 3: `distributed_serving.py:669` — DistributedServingEngine missing attribute

`serving_loop()` references `self.max_batch_size` which is not set on `DistributedServingEngine`.

```python
# In DistributedServingEngine.__init__, add:
self.max_batch_size = self.batch_config.max_batch_size
```

### Bug 4: `lockfree_logger.py` — asyncio.Queue from OS thread

`asyncio.Queue` is not thread-safe. A background OS thread calling `get_nowait()` on it causes undefined behavior (data corruption or deadlock).

```python
# Replace throughout lockfree_logger.py:
import queue  # stdlib

# Was:  self.log_queue = asyncio.Queue(maxsize=self.queue_size)
# Now:  self.log_queue = queue.SimpleQueue()

# Was:  await self.log_queue.put(entry)
# Now:  self.log_queue.put(entry)   # non-blocking, thread-safe

# Was:  entry = self.log_queue.get_nowait()  (called from thread)
# Now:  same call — SimpleQueue.get_nowait() is thread-safe
```

### Sprint 1 Done Criteria
- [ ] `python -c "from src.serving.distributed_serving import DistributedServingEngine"` imports without error
- [ ] `python -c "from src.serving.lockfree_logger import LockFreeLogger; l = LockFreeLogger(); l.log('test')"` runs clean
- [ ] Existing Python test suite passes

---

## Sprint 2 — Real Inference Pipeline (Days 2–5)
**Goal:** Tokenize real text → load real weights → multi-layer transformer forward → decode.

The current `engine.rs` has 100 hardcoded words and xorshift-seeded random weights. It is a demo scaffold only.

### 2.1 GGUF Weight Loader

Add `src/gguf.rs` to `cpu-infer` crate. GGUF is the universal format (llama.cpp ecosystem), supports all quantization types including the BitNet TL1/TL2 formats needed in Sprint 4.

```rust
pub struct GgufTensor {
    pub name: String,
    pub dtype: GgufDtype,  // F32, F16, Q4_K_M, Q8_0, TL1, TL2
    pub shape: Vec<u64>,
    pub data: &'static [u8],  // memory-mapped, zero-copy
}

pub struct GgufModel {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensor>,
}

impl GgufModel {
    pub fn load(path: &Path) -> Result<Self>;
}
```

Parse: magic `0x46554747`, version u32, tensor_count u64, metadata_kv_count u64, then kv pairs, then tensor metadata, then data region. Use `memmap2` crate for zero-copy mmap.

Start with Llama 3.2 1B Q4_K_M (~800 MB) as the validation target — small enough to iterate fast.

### 2.2 Real Tokenizer

**Option A (recommended):** PyO3 FFI wrapper around tiktoken.

```rust
// dependencies/cpu-infer/src/tokenizer.rs
use pyo3::prelude::*;

pub enum TokenizerModel { Cl100kBase, Llama3, Llama2 }

pub fn tokenize(text: &str, model: TokenizerModel) -> PyResult<Vec<u32>> {
    Python::with_gil(|py| {
        let tiktoken = py.import("tiktoken")?;
        // cl100k_base for GPT-4 family, o200k_base for llama3
        ...
    })
}
```

**Option B (pure Rust):** Add `tokenizers = "0.20"` (HuggingFace Rust tokenizer). Slower to set up but no Python dep.

Expose: `pub fn tokenize(text: &str) -> Vec<u32>` and `pub fn detokenize(ids: &[u32]) -> String`.

### 2.3 Multi-Layer Transformer

Replace `CpuInferEngine`'s random-weight demo with a real transformer. Structure:

```rust
pub struct ModelConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,  // GQA support
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
}

pub struct KVCache {
    // Per-layer key and value tensors, grown incrementally
    keys: Vec<Tensor>,    // [n_layers][seq_len, n_kv_heads, head_dim]
    values: Vec<Tensor>,
}
```

Each transformer block:
1. RMSNorm(x)
2. Attention: QKV projections → RoPE → scaled dot-product → output projection
3. RMSNorm(x)
4. FFN: SwiGLU (gate_proj × SiLU(up_proj), then down_proj)

Use existing `quantize.rs` INT8/INT4 for projection weight loading.

### 2.4 Generation Loop

```rust
pub fn generate(
    &mut self,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> impl Iterator<Item = String>
```

Greedy (temperature=0) first, then temperature + top-p nucleus sampling.

### Sprint 2 Done Criteria
- [ ] Loads Llama 3.2 1B Q4_K_M GGUF without OOM
- [ ] `generate("Hello, my name is", 20, 0.0, 1.0)` produces coherent tokens (not garbage)
- [ ] KV cache does not grow unbounded (bounded by `max_context_len`)
- [ ] `cargo test` passes: GGUF round-trip, tokenizer round-trip, single-layer forward correctness

---

## Sprint 3 — Real KV Cache + Token Recycling (Days 6–9)
**Goal:** Replace FractalMycelium with a real semantic KV cache enabling cache-hit acceleration.

Delete `src/recycler/fractal_mycelium.py` entirely (verify 0 imports remain).

### Design: Chunk-Aligned Semantic KV Cache

Inspired by ChunkKV (arxiv:2502.00299): cache at chunk-aligned token boundaries so partial prefix matches are possible. Longer matching prefix → more KV layers reused → faster decode start.

```python
# src/recycler/semantic_kv_cache.py

from dataclasses import dataclass
import hashlib
import time

CHUNK_SIZE = 256  # cache boundary granularity (align to power of 2)

@dataclass
class KVCacheEntry:
    prefix_hash: str      # SHA-256 hex of token_ids[:boundary]
    kv_tensors: list      # per-layer list of (K, V) tensors
    last_used: float
    hit_count: int = 0


class SemanticKVCache:
    def __init__(self, max_entries: int = 512):
        self._cache: dict[str, KVCacheEntry] = {}
        self.max_entries = max_entries

    def _hash(self, token_ids: list[int]) -> str:
        return hashlib.sha256(
            b"".join(t.to_bytes(4, "little") for t in token_ids)
        ).hexdigest()

    def lookup(self, token_ids: list[int]) -> tuple[list, int]:
        """Return (kv_tensors, prefix_length) for the longest cached prefix.
        Checks progressively longer chunk-aligned prefixes, returns best hit.
        Returns ([], 0) on miss."""
        best_kv, best_len = [], 0
        boundary = CHUNK_SIZE
        while boundary <= len(token_ids):
            h = self._hash(token_ids[:boundary])
            if h in self._cache:
                entry = self._cache[h]
                entry.last_used = time.monotonic()
                entry.hit_count += 1
                best_kv, best_len = entry.kv_tensors, boundary
            boundary += CHUNK_SIZE
        return best_kv, best_len

    def store(self, token_ids: list[int], kv_tensors: list):
        """Store KV tensors for chunk-aligned prefixes of this sequence."""
        if len(self._cache) >= self.max_entries:
            self._evict_lru()
        # Store at the longest chunk-aligned boundary
        boundary = (len(token_ids) // CHUNK_SIZE) * CHUNK_SIZE
        if boundary == 0:
            return
        h = self._hash(token_ids[:boundary])
        self._cache[h] = KVCacheEntry(
            prefix_hash=h,
            kv_tensors=kv_tensors,
            last_used=time.monotonic(),
        )

    def _evict_lru(self):
        oldest = min(self._cache, key=lambda k: self._cache[k].last_used)
        del self._cache[oldest]
```

Wire into `CpuInferEngine.generate()`:
```python
kv_cache, prefix_len = self.semantic_cache.lookup(token_ids)
output_kv = self.rust_engine.generate(
    token_ids[prefix_len:],  # only forward non-cached suffix
    initial_kv=kv_cache,
)
self.semantic_cache.store(token_ids, output_kv)
```

For a typical API serving pattern (shared system prompt across requests), a 256-token chunk boundary means the first 256 tokens of every request with the same system prompt are computed once and cached.

### Sprint 3 Done Criteria
- [ ] `fractal_mycelium.py` deleted, `grep -r FractalMycelium .` returns 0 results
- [ ] `SemanticKVCache.lookup()` returns correct `(kv, prefix_len)` for exact prefix match
- [ ] Repeated identical prompts: second request shows `prefix_len > 0` in metrics
- [ ] LRU eviction fires correctly at `max_entries`
- [ ] No memory leak under 10,000 store/lookup cycles

---

## Sprint 4 — AVX-512 VNNI + BitNet b1.58 (Days 10–14)
**Goal:** Add VNNI integer dot-product kernel; implement BitNet ternary weight path to hit ≥18.75 tok/s on 7B.

**Confirmed target:** 18.75 tok/s for 7B BitNet b1.58 on Intel i7-13700H (arxiv:2410.16144v1, 2-1 vote).

### 4.1 AVX-512 VNNI Kernel

Current `dot_product_avx2()` processes 8 f32 per cycle. VPDPBUSD (`_mm512_dpbusd_epi32`) processes 64 INT8 per cycle — 8× the throughput with VNNI fusion.

Add to `kernels.rs`:

```rust
#[cfg(target_feature = "avx512vnni")]
pub unsafe fn dot_product_avx512_vnni(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    while i + 64 <= n {
        // Load 64 INT8 values per register (512 bits)
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
        // VPDPBUSD: acc += unsigned_a * signed_b (INT8 → INT32 accumulate)
        acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(va), vb);
        i += 64;
    }
    // Horizontal reduction: 512→256→128→scalar
    let lo = _mm512_extracti64x4_epi64(acc, 0);
    let hi = _mm512_extracti64x4_epi64(acc, 1);
    let sum256 = _mm256_add_epi32(
        _mm256_castsi128_si256(_mm256_extracti128_si256(
            _mm256_add_epi32(lo, hi), 0)),
        _mm256_extracti128_si256(_mm256_add_epi32(lo, hi), 1),
    );
    // ... scalar tail for remainder < 64
    _mm_cvtsi128_si32(_mm_hadd_epi32(...))  // complete reduction
}
```

Runtime dispatch (no feature flags required at compile time for portable binary):
```rust
pub fn dot_product(a: &[i8], b: &[i8]) -> i32 {
    if is_x86_feature_detected!("avx512vnni") {
        unsafe { dot_product_avx512_vnni(a, b) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { dot_product_avx2_i8(a, b) }
    } else {
        dot_product_scalar(a, b)
    }
}
```

**AMD note:** Ryzen 7000 (Zen 4) has AVX-512 but VNNI support varies by SKU — `is_x86_feature_detected!("avx512vnni")` handles this correctly at runtime.

### 4.2 BitNet b1.58 Ternary Kernel

BitNet weights are {-1, 0, +1} packed 2 bits/value (4 values per byte). The matmul becomes sign-conditional add/subtract/skip — no multiplication needed.

```rust
// Weight packing: bits[1:0] — 00=0, 01=+1, 10=-1
pub fn bitnet_matmul_ternary(
    activations: &[i8],   // INT8 activations (W1.58A8 scheme)
    weights: &[u8],        // packed ternary: 4 values per byte
    out_dim: usize,
    in_dim: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; out_dim];
    for o in 0..out_dim {
        let mut acc = 0i32;
        for i in 0..in_dim {
            let byte_idx = (o * in_dim + i) / 4;
            let bit_pos = ((o * in_dim + i) % 4) * 2;
            let w = (weights[byte_idx] >> bit_pos) & 0x3;
            acc += match w {
                0b01 => activations[i] as i32,   // +1
                0b10 => -(activations[i] as i32), // -1
                _ => 0,                            // 0 (zero weight, skip)
            };
        }
        out[o] = acc;
    }
    out
}
```

Add SIMD-accelerated version using AVX2 masking for the ternary scatter.

**Weight format:** support TL1 and TL2 tensor types from GGUF (bitnet.cpp format). Extend `GgufDtype` enum to include these.

### 4.3 Per-Token INT8 Activation Quantization

Required for W1.58A8 (BitNet standard — ternary weights, INT8 activations):

```rust
pub fn quantize_activations_per_token(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    // Per-row absmax quantization: scale = max(|x|) / 127
    let scale = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0;
    let quant = x.iter().map(|v| (v / scale).round() as i8).collect();
    (quant, vec![scale])
}

pub fn dequantize_output(y: &[i32], scale_a: f32, scale_w: f32) -> Vec<f32> {
    y.iter().map(|v| *v as f32 * scale_a * scale_w).collect()
}
```

### 4.4 Benchmarks (criterion)

Add `benches/inference.rs`:
```
bitnet_7b_matmul_vnni:   target ≥ 18.75 tok/s equivalent throughput
bitnet_7b_matmul_avx2:   baseline for comparison
fp16_7b_matmul:          FP16 baseline (should be 2.37x–6.17x slower)
```

### Sprint 4 Done Criteria
- [ ] `dot_product_avx512_vnni()` compiles and produces correct output vs scalar reference
- [ ] `bitnet_matmul_ternary()` passes round-trip test against f32 reference (within rounding tolerance)
- [ ] `cargo bench` shows VNNI ≥ AVX2 throughput on VNNI-capable hardware
- [ ] GGUF loader parses TL1/TL2 tensor types without panic
- [ ] Per-token INT8 quantization round-trip test passes

---

## Sprint 5 — Production API Layer (Days 15–19)
**Goal:** The FastAPI server that CLAUDE.md Sprint 1 expects. OpenAI-compatible, auth, rate limiting, Docker.

### 5.1 Core Endpoints (`src/api/server.py`)

```python
POST /v1/chat/completions    # OpenAI-compatible, streaming SSE
POST /v1/completions         # legacy completions
POST /v1/embeddings          # last hidden state → 1024-dim float vector
GET  /v1/models              # {"data": [{"id": "ryzanstein-7b", ...}]}
GET  /health                 # {"status":"ok","model":"...","tok_s":...}
```

**Streaming:** `StreamingResponse` with Server-Sent Events. Each chunk:
```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"..."},"finish_reason":null}]}\n\n
```
Terminal chunk: `data: [DONE]\n\n`

**Embeddings:** Use the final hidden state of the last transformer layer before the LM head, normalized to unit length. Return `{"data":[{"embedding":[...1024 floats],"index":0}]}`.

### 5.2 Middleware

`src/api/middleware/auth.py`:
```python
# Bearer token validation against RYZANSTEIN_API_KEYS env var (comma-separated)
# Return 401 with {"error":"invalid_api_key"} on failure
# NEVER log the key value — only log "auth_failed" with masked key suffix
```

`src/api/middleware/rate_limit.py`:
```python
# Token bucket per API key: configurable RYZANSTEIN_RATE_LIMIT_RPM (default 60)
# Return 429 with {"error":"rate_limit_exceeded","retry_after":...} on breach
```

`src/api/middleware/prompt_guard.py`:
```python
# Detect prompt injection patterns:
# - "ignore previous instructions", "ignore all previous", "disregard"
# - System prompt override attempts
# - Base64 / unicode-encoded instructions
# Return 400 with {"error":"prompt_injection_detected"} on match
# Log detection event (without the prompt content) for audit
```

### 5.3 Continuous Batching

Wire `DynamicBatcher` from `batch_engine.py` into the serving path:
- Requests arrive → pushed to batcher queue
- Batcher accumulates for up to `max_wait_ms` (default 10ms) or `max_batch_size` requests
- Batch dispatched to `CpuInferEngine.generate_batch()`

For CPU (no GPU memory management): sequential per-request with shared KV cache (no PagedAttention needed — that's a GPU optimization). The SemanticKVCache from Sprint 3 provides the CPU equivalent of shared prefix caching.

### 5.4 Docker (`Dockerfile`)

```dockerfile
# Stage 1: Rust build
FROM rust:1.96-slim AS rust-builder
WORKDIR /build
COPY dependencies/ dependencies/
COPY Cargo.toml Cargo.lock ./
RUN cargo build --release --features avx2
# Optionally: --features avx512vnni for VNNI-capable hosts

# Stage 2: Python runtime
FROM python:3.13-slim AS runtime
WORKDIR /app
COPY --from=rust-builder /build/target/release/libcpu_infer.so ./
COPY src/ src/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
ENV RYZANSTEIN_PORT=8000
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Build arg: `--build-arg FEATURES=avx512vnni` for VNNI hosts.

### Sprint 5 Done Criteria
- [ ] `curl http://localhost:8000/health` → `{"status":"ok"}`
- [ ] `curl /v1/chat/completions` with valid Bearer token → streams tokens
- [ ] Request with missing/invalid token → `401`
- [ ] Exceed rate limit → `429` with `retry_after`
- [ ] Prompt injection attempt → `400`
- [ ] `docker build -t ryzanstein-llm:latest .` succeeds
- [ ] `docker run -p 8000:8000 ryzanstein-llm:latest` starts and passes health check
- [ ] `/v1/embeddings` returns array of 1024 floats

---

## Sprint 6 — ZK Audit Integration + Ecosystem Wiring (Days 20–23)
**Goal:** Wire `zkaudit` into live inference; expose MCP endpoint; write ecosystem integration docs.

### 6.1 ZK Audit on Every Inference Request

The `proof.rs` Schnorr protocol currently lives unused. Wire it into request completion:

For each request, after inference completes:
```rust
// In the Rust serving layer, post-generation:
let request_id = Uuid::new_v4().to_string();
let prompt_hash = sha256(prompt_bytes);
let response_hash = sha256(response_bytes);

// 1. Append to hash chain
chain.append(ChainEntry {
    id: request_id.clone(),
    data: format!("prompt:{prompt_hash} response:{response_hash} model:{model_id}"),
    timestamp: unix_timestamp(),
});

// 2. Generate Schnorr proof (proves this model produced this response)
// secret x = SHA256(model_secret_key) — NEVER LOGGED
let proof = ZkProof::generate(
    &model_secret_key,  // never appears in logs
    &format!("model:{model_id} response:{response_hash}"),
);

// 3. Store {request_id, chain_entry, proof} in audit store
```

**Security:** `model_secret_key` is derived from `MODEL_SECRET_KEY` env var, is never logged (per security rule: vault master key and tunnel decryption keys never appear in any log output).

Expose: `GET /v1/audit/{request_id}` → `{chain_entry, merkle_proof, schnorr_proof, verified: bool}`

Every 16 requests, compute a new Merkle root using `merkle.rs` and append it to the chain. External verifiers can confirm any response was produced by the expected model key.

### 6.2 MCP Tool Endpoint

```python
# src/api/mcp.py

GET  /mcp/tools   →  list of tool descriptors
POST /mcp/invoke  →  invoke tool by name, return result
```

Tool descriptors:
```json
[
  {
    "name": "generate",
    "description": "CPU-first LLM text generation",
    "input_schema": {
      "type": "object",
      "properties": {
        "prompt": {"type": "string"},
        "max_tokens": {"type": "integer", "default": 256},
        "temperature": {"type": "number", "default": 0.7}
      }
    }
  },
  {
    "name": "embed",
    "description": "Generate 1024-dim text embeddings",
    "input_schema": {
      "type": "object",
      "properties": {"input": {"type": "string"}}
    }
  }
]
```

Register with mcp-mesh: `POST /mcp/register` on the mesh with `{agent_id: "ryzanstein", endpoint: "http://ryzanstein:8000/mcp"}`.

### 6.3 Ecosystem Integration Doc

`ECOSYSTEM_INTEGRATION.md` covers:
- `/v1/embeddings` API reference for sigma-compress, sigma-index, sigma-diff consumers (dimension: 1024, dtype: float32)
- `/v1/chat/completions` streaming reference for agent use
- MCP integration guide for mcp-mesh registration
- Required env vars: `MODEL_PATH`, `MODEL_SECRET_KEY`, `RYZANSTEIN_PORT`, `RYZANSTEIN_API_KEYS`, `RYZANSTEIN_RATE_LIMIT_RPM`
- Audit trail: how external verifiers use `/v1/audit/{id}`

### Sprint 6 Done Criteria
- [ ] Every inference request appended to hash chain
- [ ] `GET /v1/audit/{id}` returns valid `{chain_entry, schnorr_proof}` with `verified: true`
- [ ] `MODEL_SECRET_KEY` appears in 0 log lines (grep audit)
- [ ] Merkle proof computed every 16 requests, root stored in chain
- [ ] `GET /mcp/tools` lists `generate` and `embed`
- [ ] mcp-mesh can register Ryzanstein as an agent via `/mcp/register`
- [ ] `ECOSYSTEM_INTEGRATION.md` committed

---

## Sprint Dependency Graph

```
S1 (crash fixes) ──► S2 (inference pipeline) ──► S3 (KV cache) ──► S5 (API)
                                               ──► S4 (SIMD/BitNet) ──► S5
                                                                         │
                                                                         ▼
                                                                      S6 (ZK + MCP)
```

S3 and S4 can run in parallel after S2. Both must complete before S5.

---

## Performance Targets by Sprint

| After | Model | Target | Basis |
|-------|-------|--------|-------|
| S2 | Llama 3.2 1B Q4_K_M | > 5 tok/s | Unoptimized CPU baseline |
| S4 | 7B BitNet b1.58 | ≥ 18.75 tok/s | Confirmed: bitnet.cpp on i7-13700H |
| S4 | 700M BitNet b1.58 | ≥ 55 tok/s | CLAUDE.md target; confirmed achievable (119 tok/s on bitnet.cpp) |
| S5 | 7B, batch=4 continuous | ≥ 15 tok/s | ~20% batching overhead expected |

---

## What NOT to Adopt

| Technology | Reason |
|-----------|--------|
| rvLLM / vllm-rs | GPU-only (H100/CUDA). Not applicable to CPU-first target. |
| SGLang CPU backend | Requires Python + Intel-specific kernel stack. Conflicts with Rust-first design. |
| mistral.rs PagedAttention | CUDA / Apple Silicon only. CPU mode is sequential anyway — SemanticKVCache (Sprint 3) is the CPU equivalent. |
| candle (HuggingFace Rust) | `kernels.rs` AVX2/VNNI is already better tuned. Adding candle would be a regression. |
| GPU inference path | Breaks the core value proposition: portable, runs on 1TB drive, no GPU required. |

---

## Version Target

After all 6 sprints complete:
```bash
git tag v3.0.0 -m "Real inference pipeline: GGUF loader, BitNet b1.58, AVX-512 VNNI, semantic KV cache, production API"
git push origin v3.0.0
```

Update CLAUDE.md: `Ryzanstein ✅ v3.0.0 (GGUF+BitNet b1.58+AVX-512 VNNI @ ≥18.75 tok/s 7B)`
