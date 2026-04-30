# STRIDE Threat Model — Ryzanstein IPC Router

**Component:** `desktop/internal/ipc/` (server.go, router.go, bridge.go)  
**Protocol:** Plain TCP JSON newline-delimited, `localhost:9001`  
**Sprint:** 10 — Security hardening

---

## Architecture Summary

The IPC server accepts TCP connections on `localhost:9001`. Clients (VS Code
extension, MCP bridge) send newline-terminated JSON commands; the server
dispatches to handlers for inference, model listing, and agent invocation.
There is currently **no authentication, no TLS, no rate limiting**.

---

## Threat Analysis

### S — Spoofing

| ID | Threat | Current state | Mitigation |
|----|--------|--------------|------------|
| S-1 | Any local process can connect and impersonate VS Code | No client identity check | Add a pre-shared token exchanged over a secure channel (e.g. env var injected at launch) |
| S-2 | `clientID` is the remote address string — trivially guessable | `clientID = conn.RemoteAddr().String()` | Replace with a UUID assigned at `acceptConnections` time |

### T — Tampering

| ID | Threat | Current state | Mitigation |
|----|--------|--------------|------------|
| T-1 | JSON payload can be mutated in transit | Plain TCP, no MAC | Localhost-only binding limits exposure; add HMAC if loopback-only assumption is ever relaxed |
| T-2 | `invoke_tool` with arbitrary `agent_codename`/`tool_name` can trigger unintended agent actions | No allowlist | Validate `AgentCodename` against the loaded agent registry before dispatch |

### R — Repudiation

| ID | Threat | Current state | Mitigation |
|----|--------|--------------|------------|
| R-1 | No audit trail for IPC commands | `log.Printf` only; logs not persisted | Wire each dispatched command into the `zkaudit.AuditChain` (Sprint 10) |
| R-2 | Client disconnect loses correlation between request and action | No session IDs | Assign per-request correlation IDs; include in audit log |

### I — Information Disclosure

| ID | Threat | Current state | Mitigation |
|----|--------|--------------|------------|
| I-1 | Inference prompt echoed in debug log | `log.Printf("[IPC] Received from %s: %s\n", clientID, line)` | Redact payload body in production builds; log only command type + request ID |
| I-2 | Error messages may leak internal paths | `fmt.Errorf("inference failed: %v", err)` propagated to client | Wrap errors; map to opaque codes before sending |
| I-3 | `handleInfer` response includes full usage stats | Intentional but worth noting | Document as designed |

### D — Denial of Service

| ID | Threat | Current state | Mitigation |
|----|--------|--------------|------------|
| D-1 | Unbounded concurrent connections | `go s.handleClient(...)` per accept | Add `maxClients` semaphore; reject connections beyond limit |
| D-2 | 1 MB scanner buffer per client | `scanner.Buffer(make([]byte, 1024*1024), ...)` | Already bounded; acceptable |
| D-3 | Slow-loris: client sends partial JSON forever | Scanner blocks on `Scan()` | Add per-connection read deadline: `conn.SetReadDeadline(time.Now().Add(30 * time.Second))` |
| D-4 | Inference handler blocks goroutine for full model latency | No timeout on `s.apiClient.Infer(...)` | Pass context with deadline; cancel after configurable timeout |

### E — Elevation of Privilege

| ID | Threat | Current state | Mitigation |
|----|--------|--------------|------------|
| E-1 | `invoke_tool` executes arbitrary agent tools as the desktop process user | No capability check | Add an allowlist of permitted tool names per agent; reject unknown tools |
| E-2 | Future commands (e.g. file I/O, shell exec) could be added without security review | No command registration policy | Require threat-model update for any new command added to `dispatchCommand` |

---

## Priority Mitigations for Sprint 10

1. **I-1 (HIGH)** — Redact prompt body from IPC logs; replace with `[REDACTED len=N]`
2. **D-3 (HIGH)** — Add `conn.SetReadDeadline` in `handleClient`
3. **S-1 (MEDIUM)** — Pre-shared token auth (env var `RYZANSTEIN_IPC_TOKEN`)
4. **E-1 (MEDIUM)** — Agent tool allowlist in `InvokeToolPayload` handler
5. **R-1 (LOW)** — zkaudit wiring (separate Sprint 10 track)
6. **D-1 (LOW)** — Max-clients semaphore

---

## Out of Scope

- Network-level attacks: server binds `localhost` only; external network unreachable
- TLS: loopback TCP without TLS is acceptable for localhost IPC; add if UDS migration is deferred
