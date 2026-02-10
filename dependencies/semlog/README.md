# semlog — Semantic Log Compression with Queryable Compressed Format

**Tier:** 3 — Hybrid (standalone + Ryzanstein-enhanced)  
**Languages:** Rust + Go  
**Status:** Scaffolded  
**Version:** 0.1.0

## Overview

`semlog` compresses logs by **understanding their meaning**, not just their text patterns.
It classifies tokens semantically (timestamps → delta encoding, IPs → dictionary encoding,
error codes → enum encoding), then groups log lines into **behavioral patterns**
("retry sequence", "cascading failure", "normal startup").

The compressed format is **directly queryable**: search for "all retry sequences longer
than 3 attempts" without decompression.

## Compression Ratios

| Mode | Compression vs Raw | Compression vs gzip |
|------|-------------------|---------------------|
| Standalone (Drain + semantic typing) | 10–20× | 3–5× better |
| With Ryzanstein (ΣLANG patterns) | 30–50× | 10–15× better |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  LOG INGESTION (OpenTelemetry / stdin / file)   │
├─────────────────────────────────────────────────┤
│  DRAIN PARSER — Template extraction             │
│  "GET /api/{path} {status} {latency}ms"         │
├─────────────────────────────────────────────────┤
│  SEMANTIC TOKEN CLASSIFIER                      │
│  timestamps → delta, IPs → dict, codes → enum   │
├─────────────────────────────────────────────────┤
│  BEHAVIORAL PATTERN GROUPER                     │
│  Sequences → patterns (retry, cascade, normal)  │
├─────────────────────────────────────────────────┤
│  COMPRESSED STORAGE (queryable .semlog format)  │
│  Templates + variables + patterns + index       │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Compress logs
semlog compress --input server.log --output server.semlog

# Query compressed logs
semlog query --file server.semlog "retry sequences > 3 attempts"

# Stream mode (pipe from stdout)
tail -f /var/log/app.log | semlog stream --output live.semlog
```

## Go Query Server

```bash
cd query-server && go run ./cmd/semlog-server --port 8090
```

## License

Apache-2.0 (standalone) / Ryzanstein Commercial License (enhanced features)
