# neurectomy-shell — Encrypted Confidential Development Environment

**Tier:** 3 — Hybrid (standalone + Ryzanstein-enhanced)  
**Languages:** Go (server/orchestrator) + Rust (Tauri desktop app)  
**Status:** Scaffolded  
**Version:** 0.1.0

## Overview

`neurectomy-shell` provides a fully encrypted confidential development environment that
ensures source code, build artifacts, and development context are never exposed—even
to the host operating system.

### Core Capabilities

| Feature | Technology | Standard |
|---------|------------|----------|
| Confidential VM | AMD SEV-SNP / Intel TDX | Hardware TEE |
| Encrypted Storage | ΣVAULT integration | AES-256-GCM |
| Desktop App | Tauri (Rust + Svelte) | Cross-platform |
| Remote Dev | VS Code Server tunnels | SSH/HTTPS |
| Attestation | TPM 2.0 remote attestation | NIST SP 800-155 |

## Architecture

```
┌──────────────────────────────────────────────────┐
│                TAURI DESKTOP APP                  │
│         (Rust core + Svelte frontend)             │
├──────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ Workspace  │  │ Attestation│  │ Encrypted  │ │
│  │ Manager    │  │ Monitor    │  │ File Sync  │ │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘ │
└─────────┼───────────────┼───────────────┼────────┘
          │               │               │
     ┌────▼───────────────▼───────────────▼────┐
     │          GO ORCHESTRATOR SERVER          │
     │   VM lifecycle, key management, audit    │
     ├─────────────────────────────────────────┤
     │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
     │  │ SEV-SNP │  │ ΣVAULT  │  │ Audit   │ │
     │  │ Manager │  │ Bridge  │  │ Logger  │ │
     │  └─────────┘  └─────────┘  └─────────┘ │
     └─────────────────────────────────────────┘
                        │
            ┌───────────▼───────────┐
            │   CONFIDENTIAL VM     │
            │  (AMD SEV-SNP / TDX)  │
            │  Encrypted memory     │
            │  Isolated execution   │
            └───────────────────────┘
```

## Quick Start

### Server (Go)

```bash
cd cmd/server
go run . --config config.yaml
```

### Desktop App (Tauri)

```bash
cd desktop
cargo tauri dev
```

### CLI

```bash
neurectomy-shell create --name my-project --confidential
neurectomy-shell connect --workspace my-project
neurectomy-shell attest --workspace my-project
```

## Security Model

1. **Memory Encryption**: All VM memory encrypted by hardware (SEV-SNP)
2. **Storage Encryption**: ΣVAULT envelope encryption for all artifacts
3. **Attestation**: Remote attestation proves VM integrity before secrets are released
4. **Key Management**: Keys never leave the TEE boundary
5. **Audit Trail**: Immutable log of all operations

## Ryzanstein Integration

- **ΣVAULT**: End-to-end encrypted storage for source code and artifacts
- **sigma-telemetry**: Confidential metrics (encrypted at rest)
- **sigma-api**: API traffic compression within confidential boundary
- **zkaudit**: Zero-knowledge proofs of code properties without revealing source

## License

Apache-2.0 (standalone) / Ryzanstein Commercial License (TEE features)
