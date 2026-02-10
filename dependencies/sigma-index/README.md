# σ-index — Succinct Semantic Code Index

**Tier:** 1 (Ecosystem-locked, Free/OSS)  
**Languages:** Rust  
**License:** AGPL-3.0  

## Overview

σ-index combines an FM-index built on wavelet trees with ΣLANG-compressed code
embeddings to create a code search index occupying space comparable to
gzip-compressed source while supporting O(m) pattern matching and O(log n) semantic search.

## Key Features

- **FM-index**: Wavelet tree BWT for O(m) exact pattern matching
- **HNSW**: Hierarchical Navigable Small World graph for semantic search
- **Incremental**: tree-sitter based live updates on file changes
- **Compression**: Index size ~12–15MB for 500K-line codebases
- **Hybrid search**: Fused exact + semantic result ranking

## Architecture

```
┌───────────────────────────┐
│     Query Interface       │
│  (CLI / Library / WASM)   │
├────────────┬──────────────┤
│  FM-Index  │  HNSW Layer  │
│  (Exact)   │  (Semantic)  │
├────────────┴──────────────┤
│   Incremental Parser      │
│   (tree-sitter)           │
├───────────────────────────┤
│  Ryzanstein Integration   │
│  (ΣLANG Embeddings)       │
└───────────────────────────┘
```

## Usage

### Library
```rust
use sigma_index::{SigmaIndex, IndexConfig, SearchQuery};

let config = IndexConfig::default();
let mut index = SigmaIndex::new(config)?;
index.index_directory(Path::new("./src")).await?;
let results = index.search(&SearchQuery::hybrid("error handling")).await?;
```

### CLI
```bash
sigma-index build --path ./src
sigma-index search --query "retry logic" --mode hybrid --top-k 10
sigma-index stats
sigma-index watch --path ./src
```

## Ryzanstein Integration

σ-index connects to Ryzanstein's ΣLANG encoder for semantic embeddings:
- **Endpoint**: `http://localhost:8000/v1/embeddings`
- **Fallback**: Hash-based local embeddings when Ryzanstein unavailable
- **Model**: `sigma-lang-v1` (1024-dim vectors)

## Complexity Guarantees

| Operation          | Complexity  | Notes                     |
|--------------------|-------------|---------------------------|
| Exact search       | O(m)        | m = pattern length        |
| Semantic search    | O(log n)    | n = indexed code chunks   |
| Index build        | O(n log n)  | n = total source bytes    |
| Incremental update | O(Δ log n)  | Δ = changed bytes         |
| Space              | ~15 MB      | Per 500K lines            |

## Development

```bash
cargo build
cargo test
cargo bench
```

## Testing

```bash
cargo test                    # Unit + integration tests
cargo test --features simd    # With SIMD optimizations
```
