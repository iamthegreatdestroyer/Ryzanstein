# σ-diff — Structural AST-Aware Diff Engine

**Tier:** 1 (Ecosystem-locked)  
**Languages:** Rust (core) + Python bindings  
**License:** AGPL-3.0

## Overview

σ-diff computes semantically-meaningful diffs between code files using tree-sitter
AST parsing. Instead of line-level diffs, it produces structural changes
(function added, parameter renamed, logic restructured) with risk scoring.

## Features

- **AST-aware**: Uses tree-sitter for structural parsing
- **Multi-language**: Rust, Python, Go, JavaScript/TypeScript
- **Risk scoring**: Weighted change risk assessment
- **Semantic diffing**: Optional Ryzanstein-powered semantic comparison
- **Structural change categories**: Function, Struct, Import, Logic, Comment, etc.

## Usage

```rust
use sigma_diff::{DiffEngine, DiffConfig};

let engine = DiffEngine::new(DiffConfig::default());
let result = engine.diff_content(old_code, new_code, "old.rs", "new.rs")?;

for change in &result.changes {
    println!("{:?} {:?}: {}", change.kind, change.category, change.description);
}
println!("Risk score: {:.2}", result.summary.risk_score);
```

## Ryzanstein Integration

σ-diff leverages Ryzanstein's embedding API for semantic diff:
- Computes embedding similarity between code versions
- Falls back to text-based Jaccard similarity when offline
