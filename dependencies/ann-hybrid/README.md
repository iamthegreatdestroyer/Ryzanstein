# ann-hybrid — Unified Sub-Linear Search Engine

**Tier:** 3 — Hybrid (standalone + Ryzanstein-enhanced)  
**Language:** Rust (with WASM target)  
**Status:** Scaffolded  
**Version:** 0.1.0

## Overview

`ann-hybrid` fuses three sub-linear data structures into a **single unified search index**:

| Structure | Role | Complexity |
|-----------|------|------------|
| **HNSW** (Hierarchical Navigable Small World) | Semantic nearest-neighbor search | O(log n) |
| **Cuckoo Filter** | Exact identifier set membership | O(1) |
| **Count-Min Sketch** | Frequency estimation for ranking | O(1) |

Search queries flow through all three structures in parallel, with results merged
and ranked by a composite scoring function.

## Performance Targets

| Metric | Target |
|--------|--------|
| Search latency (1M items) | <5ms |
| Index build (1M items) | <30s |
| Memory per item | <256 bytes |
| WASM bundle size | <500KB |

## Architecture

```
                    ┌─────────────┐
                    │  QUERY API  │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │    HNSW    │  │  Cuckoo    │  │  Count-Min │
    │  Semantic  │  │  Filter    │  │  Sketch    │
    │  Nearest   │  │  Exact     │  │  Frequency │
    │  Neighbor  │  │  Membership│  │  Ranking   │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          └───────────────┼───────────────┘
                    ┌─────▼─────┐
                    │  MERGER   │
                    │  Score &  │
                    │  Rank     │
                    └───────────┘
```

## Quick Start

### Library Usage (Rust)

```rust
use ann_hybrid::{HybridIndex, IndexConfig, SearchQuery};

let config = IndexConfig::default();
let mut index = HybridIndex::new(config);

// Add items (id, embedding vector, metadata)
index.insert("fn_main", &embedding, Some("entry point"));

// Search
let results = index.search(&SearchQuery {
    vector: Some(&query_embedding),
    keyword: Some("main"),
    top_k: 10,
});
```

### CLI

```bash
ann-hybrid index --input corpus.jsonl --output index.bin
ann-hybrid search --index index.bin --query "error handling" --top-k 5
```

### WASM (browser/VS Code extension)

```js
import init, { HybridIndex } from 'ann-hybrid';
await init();
const index = new HybridIndex();
```

## License

Apache-2.0 (standalone) / Ryzanstein Commercial License (enhanced features)
