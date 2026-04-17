# ADR-004: Agent Memory Subsystem (`agentmem`)

**Status:** Accepted  
**Date:** 2025-07-24  
**Deciders:** @iamthegreatdestroyer  
**Supersedes:** None

---

## Context

The Ryzanstein platform requires a cross-agent episodic memory protocol that allows
multiple agents (APEX, CIPHER, ARCHITECT, TENSOR, etc.) to share experiences, learn
from past interactions, and consolidate knowledge autonomously. No existing library
provides the combination of four-layer memory architecture, cross-agent consolidation,
and sub-linear retrieval with Ryzanstein-native compression and encryption.

The `agentmem` package (located at `dependencies/agentmem/`) was designed to fill
this gap as a standalone Python library compatible with any LLM/agent framework via
MCP server interfaces.

---

## Decision

Adopt `agentmem` as the canonical agent memory subsystem with the following architecture:

### Four-Layer Memory Architecture

| Layer                 | Module          | Persistence                  | Purpose                                                   |
| --------------------- | --------------- | ---------------------------- | --------------------------------------------------------- |
| **Working Memory**    | `working.py`    | Ephemeral (task-scoped)      | Active task context, scratchpad                           |
| **Episodic Memory**   | `episodic.py`   | Persistent (timestamped)     | Experiences with HNSW vector index for O(log n) retrieval |
| **Semantic Memory**   | `semantic.py`   | Persistent (knowledge graph) | Facts and relations via NetworkX directed graph           |
| **Procedural Memory** | `procedural.py` | Persistent (workflow store)  | Learned multi-step workflows and strategies               |

### Consolidation Pipeline

The novel `ConsolidationPipeline` (`consolidation.py`) is the critical differentiator:

1. **Cross-agent pattern extraction** — identifies recurring strategies across agent boundaries
2. **Contradiction resolution** — detects and resolves conflicting facts in semantic memory
3. **Generalization** — promotes high-fitness episodic memories to semantic knowledge
4. **Decay** — time-weighted forgetting of low-value memories

### Ryzanstein Enhancements

When optional `ryzanstein` extras are installed (`sigma-lang`, `sigma-vault`):

- **ΣLANG Compression** — 5–10× reduction in memory storage via `sigma-compress`
- **ΣVAULT Encryption** — per-agent memory isolation with zero-knowledge audit trail
- **MCP-Mesh Routing** — O(1) agent-to-memory routing via HNSW-backed mesh topology

These enhancements are accessed through `RyzansteinMemoryClient` and `RyzansteinConfig`
defined in `ryzanstein.py`.

### Integration Path: Go Desktop ↔ Python agentmem

```
Desktop (Go/Wails)
    │
    ├── internal/agents/service.go   ← agent orchestration
    │       │
    │       ▼
    ├── internal/agents/memory.go    ← Go MCP client (NEW in Sprint 4.2)
    │       │
    │       ▼ (MCP JSON-RPC over stdio)
    │
    └── agentmem MCP Server (Python)
            │
            ├── mcp_server.py        ← MCP tool endpoints
            ├── store.py             ← MemoryStore facade
            ├── episodic.py          ← Episode CRUD + HNSW search
            ├── semantic.py          ← Fact/Relation graph ops
            ├── procedural.py        ← Workflow storage
            ├── working.py           ← Ephemeral context
            └── consolidation.py     ← Cross-agent consolidation
```

The Go side communicates with `agentmem` via MCP JSON-RPC over stdio, spawning
the Python MCP server as a child process. This avoids CGo complexity while
maintaining type safety through the MCP protocol schema.

---

## Package Structure

```
dependencies/agentmem/
├── pyproject.toml           # hatchling build, Python >=3.11
├── README.md
├── src/agentmem/
│   ├── __init__.py          # 15 public exports, v0.1.0
│   ├── consolidation.py     # ConsolidationPipeline
│   ├── episodic.py          # EpisodicMemory, Episode
│   ├── mcp_server.py        # MCP tool server (stdio transport)
│   ├── procedural.py        # ProceduralMemory, Workflow
│   ├── ryzanstein.py        # RyzansteinMemoryClient, RyzansteinConfig
│   ├── semantic.py          # SemanticMemory, Fact, Relation
│   ├── store.py             # MemoryStore (unified facade)
│   ├── types.py             # AgentId, MemoryQuery, MemoryResult
│   └── working.py           # WorkingMemory
└── tests/
    └── test_agentmem.py
```

### Public API (15 symbols)

```python
__all__ = [
    "MemoryStore",               # Unified facade
    "EpisodicMemory", "Episode", # Timestamped experiences
    "SemanticMemory", "Fact", "Relation",  # Knowledge graph
    "ProceduralMemory", "Workflow",        # Learned strategies
    "WorkingMemory",             # Ephemeral context
    "ConsolidationPipeline",     # Cross-agent consolidation
    "AgentId", "MemoryQuery", "MemoryResult",  # Core types
    "RyzansteinMemoryClient", "RyzansteinConfig",  # Ryzanstein integration
]
```

### Dependencies

| Package    | Version  | Purpose                                 |
| ---------- | -------- | --------------------------------------- |
| `numpy`    | >=1.24.0 | Vector operations for embeddings        |
| `pydantic` | >=2.0.0  | Type-safe data models                   |
| `hnswlib`  | >=0.7.0  | HNSW approximate nearest neighbor index |
| `networkx` | >=3.0    | Semantic memory knowledge graph         |

Optional (`ryzanstein` extra): `sigma-lang`, `sigma-vault`  
Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`

---

## Consequences

### Positive

- **Standalone compatibility** — works with any LLM/agent framework via MCP without Ryzanstein lock-in
- **Sub-linear retrieval** — HNSW provides O(log n) semantic search across episodes
- **Cross-agent learning** — consolidation pipeline enables emergent collective intelligence
- **Security isolation** — per-agent ΣVAULT encryption prevents cross-agent data leakage
- **Clean integration** — Go↔Python via MCP stdio avoids CGo, FFI, or HTTP overhead

### Negative

- **Python dependency** — desktop app must bundle or locate a Python 3.11+ runtime
- **Process boundary** — MCP stdio adds ~1-5ms latency per memory operation vs in-process
- **Schema coupling** — MCP tool definitions must stay synchronized between Go client and Python server

### Mitigations

- Python runtime bundled via PyInstaller or detected on PATH with graceful degradation
- Memory operations are batched where possible to amortize process-crossing cost
- MCP schema generated from shared `types.py` Pydantic models to prevent drift

---

## References

- `dependencies/agentmem/README.md` — Package documentation
- `dependencies/agentmem/pyproject.toml` — Build configuration
- `dependencies/mcp-mesh/` — MCP mesh networking layer
- `docs/STREAMING_API_CONTRACT.md` — Streaming protocol (related integration)
