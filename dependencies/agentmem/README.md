# agentmem — Cross-Agent Episodic Memory Protocol

**Tier:** 3 — Hybrid (standalone + Ryzanstein-enhanced)  
**Languages:** Python + Go  
**Status:** Scaffolded  
**Version:** 0.1.0

## Overview

`agentmem` implements a four-layer memory architecture for multi-agent systems:

- **Working Memory** — current task context (ephemeral)
- **Episodic Memory** — timestamped experiences with outcomes
- **Semantic Memory** — extracted facts and relationships (knowledge graph)
- **Procedural Memory** — learned workflows and strategies

The novel primitive is the **memory consolidation pipeline**: episodic memories from
individual agents are periodically processed by a consolidation agent that extracts
cross-agent patterns, resolves contradictions, and generalizes episodes into semantic
knowledge.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Agent A   │  Agent B   │  Agent C   │  ...     │
│  (working) │  (working) │  (working) │          │
└─────┬──────┴─────┬──────┴─────┬──────┘          │
      │            │            │                  │
      ▼            ▼            ▼                  │
┌─────────────────────────────────────────────────┐
│           EPISODIC MEMORY STORE                 │
│  Timestamped experiences, outcome labels        │
│  Vector-indexed via HNSW for semantic recall    │
├─────────────────────────────────────────────────┤
│        CONSOLIDATION PIPELINE                   │
│  Pattern extraction → Contradiction resolution  │
│  → Generalization → Semantic knowledge          │
├─────────────────────────────────────────────────┤
│          SEMANTIC MEMORY (Knowledge Graph)      │
│  Entities, relationships, extracted facts       │
├─────────────────────────────────────────────────┤
│          PROCEDURAL MEMORY                      │
│  Learned workflows, strategy templates          │
└─────────────────────────────────────────────────┘
```

## Standalone Value

Works with any LLM and any agent framework (LangGraph, CrewAI, OpenAI Agents SDK)
via MCP server interfaces.

## Ryzanstein Enhancements

1. **ΣLANG compression** — 5–10× memory storage reduction with semantic queryability
2. **ΣVAULT encryption** — per-agent encrypted memory for multi-tenant security
3. **mcp-mesh routing** — O(1) memory lookup via HNSW on ΣLANG-encoded memories

## Quick Start

```python
from agentmem import MemoryStore, EpisodicMemory, ConsolidationPipeline

store = MemoryStore(backend="local")
memory = EpisodicMemory(store=store, agent_id="apex-01")

# Record an experience
memory.record(
    task="implement rate limiter",
    outcome="success",
    strategy="sliding window with Redis",
    context={"language": "python", "framework": "fastapi"},
)

# Query relevant memories
results = memory.recall(query="rate limiting strategies", top_k=5)

# Cross-agent consolidation
pipeline = ConsolidationPipeline(store=store)
pipeline.consolidate(agent_ids=["apex-01", "cipher-02", "fortress-08"])
```

## Go MCP Server

```bash
cd server && go run ./cmd/agentmem-server --port 50061
```

## License

Apache-2.0 (standalone) / Ryzanstein Commercial License (enhanced features)
