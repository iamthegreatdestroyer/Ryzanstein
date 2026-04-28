# ADR-042: Agent Memory Architecture

| Field       | Value                          |
| ----------- | ------------------------------ |
| **Status**  | Proposed                       |
| **Date**    | 2025-07-17                     |
| **Authors** | Ryzanstein Core Team           |
| **Relates** | ADR-043 (Compression Strategy) |

## Context

The `agentmem` library provides an in-memory `MemoryStore` backed by Python
dicts with optional numpy-based vector search. While this is fine for a single
session, all agent memories are lost on process restart. A desktop-first AI
platform **must** persist conversation history, learned procedures, and semantic
facts across restarts without requiring an external database process.

### Current Architecture

```
MemoryStore (in-memory)
├── _memories      : Dict[MemoryId, StoredMemory]
├── _by_agent      : DefaultDict[AgentId, List[MemoryId]]
└── _by_layer      : DefaultDict[MemoryLayer, List[MemoryId]]
```

The existing `MemoryStore` is sophisticated—pluggable backends (local, redis,
neo4j, vault), numpy cosine-similarity search, and layered indexing (episodic,
semantic, procedural, working). Modifying it for persistence would risk
regressions across the entire memory subsystem.

## Decision

Introduce a **separate, lightweight `AgentMemoryStore`** in a new module
(`agentmem.persistence`) that provides:

1. **File-backed JSON Lines (JSONL) persistence** — one file per agent codename.
2. **Append-only writes** — no in-place mutation, minimizing corruption risk.
3. **Bounded memory** — cap at 10,000 entries with oldest-first rotation.
4. **Session-aware retrieval** — filter by `session_id`.

### Storage Format

```
Location : %APPDATA%\Ryzanstein\memory\<agent_codename>.jsonl
           (falls back to ~/.Ryzanstein/memory/ on non-Windows)

Entry schema (one JSON object per line):
{
  "timestamp":    <float, Unix epoch seconds>,
  "role":         <"user" | "assistant" | "system">,
  "content":      <string>,
  "session_id":   <string, default "default">
}
```

### Retrieval Strategies

| Strategy        | Method             | Complexity |
| --------------- | ------------------ | ---------- |
| Last-N          | `get_last_n(n)`    | O(1) slice |
| Session filter  | `get_session(sid)` | O(n) scan  |
| Semantic search | Deferred to Week 5 | —          |

### Integration Point

The new `AgentMemoryStore` does **not** replace `MemoryStore`. It is a
complementary class for simple JSONL persistence. Higher-level orchestrators
can use both: `MemoryStore` for runtime vector search, and
`AgentMemoryStore` for durable cross-restart history.

## Consequences

### Positive

- Agent memory survives application restarts.
- Append-only writes prevent data corruption on crash.
- No external dependencies (no SQLite, no Redis).
- `MemoryStore` unchanged—zero regression risk.

### Negative

- Large JSONL files (10K entries) may slow cold start (~50 ms for 10K lines).
- No built-in encryption of on-disk data (future ADR).
- Duplicate concept surface: two "store" classes exist in agentmem.

### Risks

| Risk               | Mitigation                                      |
| ------------------ | ----------------------------------------------- |
| Disk full          | Bounded to 10K entries; rotation deletes oldest |
| Corrupt JSONL      | Skip unparseable lines on load                  |
| Concurrent writers | Desktop is single-process; no locking needed    |

## Alternatives Considered

| Alternative          | Why Rejected                                          |
| -------------------- | ----------------------------------------------------- |
| SQLite               | Adds a native dependency; overkill for append log     |
| Redis                | Requires external process; unacceptable for desktop   |
| Modify MemoryStore   | High regression risk; MemoryStore is complex          |
| Pure in-memory only  | Memories lost on every restart; unacceptable          |
| Pickle serialization | Not human-readable; security risk with untrusted data |
