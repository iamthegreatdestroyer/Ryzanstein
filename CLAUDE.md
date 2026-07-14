# Ryot (Ryzanstein LLM) — v4.0 Upgrade Brief

## Project Identity
- **Repo:** iamthegreatdestroyer/Ryot
- **Language:** Python + C++ (T-MAC kernels)
- **Castle Layer:** Layer 4 — Storage & Inference
- **Current:** v3.1.0 — **correction 2026-07-14: the "BitNet 1.58b + T-MAC +
  AVX-512 @ 55 tok/s (shipped)" claim below was fabricated.** The repo's own
  `benchmark_results.txt` (Dec 2025, @VELOCITY) measured the real BitNet
  attempt at **0.4157 tok/s** on the (also fabricated) "AMD Ryzanstein 7
  7730U" — 19-28x short of the 8-12 tok/s target — with the T-MAC GEMM
  kernel crashing outright (100% correctness mismatches, 291-430% relative
  error) and AVX-512 never actually engaging ("using scalar fallback" x50).
  The live gateway (`src/api/server.py`, `RYZANSTEIN_BACKEND` env var) has
  exactly two real backends: `stub` and `ollama` — there is no `bitnet`
  branch anywhere in the server. Corrected per [[project_task228_scoping_2026-07-14]],
  matching this ecosystem's recurring "claims vs. ground truth" pattern.
- **Target:** v4.0.0 — Add vllm-rs Rust backend + MCP integration
- **Mission:** CPU-First LLM inference, no GPU required, no cloud dependency

## What Already Works
- OpenAI-compatible API server (/v1/chat/completions, /v1/embeddings), live
  via the `stub` and `ollama` backends
- BitNet 1.58b ternary inference code exists but is **not usable for real
  inference**: measured 0.4157 tok/s (near scalar-baseline parity, not the
  claimed 55 tok/s) with a broken T-MAC GEMM kernel — see the correction
  above and `benchmark_results.txt` for the full breakdown

## v4.0 Sprint Plan

### Sprint 1: Verify Current State
- [x] Start API server: `python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000`
- [x] Test /v1/chat/completions endpoint
- [x] Test /v1/embeddings endpoint
- [x] Document what works and what is broken

### Sprint 2: Add vllm-rs Backend Option
- [ ] Add configuration for vllm-rs as alternative backend
- [ ] vllm-rs provides: OpenAI-compatible API, MCP tool calling, 175 tok/s on GPU
- [ ] Keep BitNet/T-MAC as the CPU-optimized path
- [ ] Add backend selector in config: bitnet | vllm-rs | ollama-proxy
- **Correction 2026-07-14:** vllm-rs backend was never added; server.py
  supports only stub|ollama (line 49); no vllm code in src/.

### Sprint 3: MCP Server Integration
- [x] Expose Ryzanstein as an MCP tool server
- [x] Tools: generate, embed, model_info, benchmark
- [x] Compatible with Claude Code MCP protocol
- [x] Register in agents-mcp-server registry

### Sprint 4: Ecosystem Wiring
- [x] sigma-compress uses Ryzanstein for semantic dedup embeddings
- [x] sigma-index uses Ryzanstein for HNSW vector indexing
- [x] sigma-diff uses Ryzanstein for semantic similarity scoring
- [x] sigma-harvest can use Ryzanstein for content analysis
- [x] YT-Shorts uses Ollama (Ryzanstein backend) can use Ryzanstein as alternative to Ollama

## Security Rules
- No OpenAI products. This IS the OpenAI replacement.
- Model weights stored locally only
- API keys for rate limiting, not cloud auth

## Build Commands
```bash
cd /opt/sigmavault/repos/Layer-4-Storage-Ryot
pip install -e ".[dev]"
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
pytest tests/
```

## Completion Signal
```bash
git tag v4.0.0
```
