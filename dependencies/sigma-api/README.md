# sigma-api

Unified API gateway for the Ryzanstein LLM ecosystem.

## Features

- **OpenAI-Compatible** — `/v1/chat/completions` and `/v1/models` endpoints
- **JWT Authentication** — token generation and validation
- **Rate Limiting** — per-client token-bucket with DashMap
- **Ryzanstein Proxy** — transparent upstream forwarding

## Quick Start

```rust
use sigma_api::{build_router, SigmaApiConfig};

let router = build_router(SigmaApiConfig::default());
// Serve with axum::Server
```

## License

AGPL-3.0
