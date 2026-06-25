# Ryot (Ryzanstein LLM) — v4.0 Upgrade Brief

## Project Identity
- **Repo:** iamthegreatdestroyer/Ryot
- **Language:** Python + C++ (T-MAC kernels)
- **Castle Layer:** Layer 4 — Storage & Inference
- **Current:** v3.1.0 (BitNet 1.58b + T-MAC + AVX-512 @ 55 tok/s)
- **Target:** v4.0.0 — Add vllm-rs Rust backend + MCP integration
- **Mission:** CPU-First LLM inference, no GPU required, no cloud dependency

## What Already Works
- BitNet 1.58b ternary model inference
- T-MAC C++ kernels for AVX-512 optimization
- OpenAI-compatible API server (/v1/chat/completions, /v1/embeddings)
- 55 tok/s on AMD Ryzen, ~2 tok/s on AMD A9-9425

## v4.0 Sprint Plan

### Sprint 1: Verify Current State
- [x] Start API server: `python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000`
- [x] Test /v1/chat/completions endpoint
- [x] Test /v1/embeddings endpoint
- [x] Document what works and what is broken

### Sprint 2: Add vllm-rs Backend Option
- [x] Add configuration for vllm-rs as alternative backend
- [x] vllm-rs provides: OpenAI-compatible API, MCP tool calling, 175 tok/s on GPU
- [x] Keep BitNet/T-MAC as the CPU-optimized path
- [x] Add backend selector in config: bitnet | vllm-rs | ollama-proxy

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
