# causedb

Causal inference database for the Ryzanstein LLM ecosystem.

## Overview

causedb tracks cause-and-effect relationships between system events, model outputs, pipeline stages, and user actions. It enables root cause analysis, counterfactual reasoning, and causal chain tracing.

## Quick Start

```rust
use causedb::{CauseDb, EventType, LinkType};
use causedb::config::CauseDbConfig;

let db = CauseDb::new(CauseDbConfig::default());

let prompt = db.record_event(EventType::UserAction, "ui", "user submitted prompt")?;
let infer = db.record_event(EventType::ModelInference, "engine", "inference started")?;
let tokens = db.record_event(EventType::TokenGeneration, "engine", "tokens generated")?;

db.link_events(prompt, infer, LinkType::Direct, 0.95)?;
db.link_events(infer, tokens, LinkType::Direct, 0.90)?;

let chain = db.trace_chain(prompt)?;
println!("Chain length: {}, Confidence: {:.2}", chain.chain.len(), chain.total_confidence);
```

## Architecture

```
Events → CauseDb (in-memory + persistence)
    ↓
Causal Links (weighted directed edges)
    ↓
Causal Graph (petgraph DAG)
    ↓
Queries (trace_chain, find_effects, find_causes)
```

## License

AGPL-3.0
