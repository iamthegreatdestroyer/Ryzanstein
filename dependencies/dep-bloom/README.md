# dep-bloom

Probabilistic dependency resolution using Bloom filters for the Ryzanstein LLM ecosystem.

## Overview

dep-bloom provides O(1) dependency existence checking via Bloom filters, dependency graph cycle detection, topological sorting, and version conflict analysis. Used by Ryzanstein for fast dependency resolution across its multi-language ecosystem.

## Features

- **O(1) Dependency Lookup** via Bloom filter with configurable false positive rate (default 1%)
- **Cycle Detection** via DFS on directed dependency graphs
- **Topological Sort** via Kahn's algorithm for build ordering
- **Version Conflict Detection** for exact version constraints
- **Filter Union** for merging dependency sets across workspaces

## Quick Start

```rust
use dep_bloom::{new_resolver, Dependency, VersionConstraint, DependencyGraph};

let mut resolver = new_resolver();
resolver.add_dependency(Dependency {
    name: "tokio".into(),
    version: "1.34.0".into(),
    constraint: VersionConstraint::Compatible,
});
assert!(resolver.has_dependency("tokio")); // O(1)

let mut graph = DependencyGraph::new();
graph.add_edge("app", "lib");
graph.add_edge("lib", "core");
let order = graph.topological_sort().unwrap(); // ["core", "lib", "app"]
```

## License

AGPL-3.0
