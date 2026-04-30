# NEXT STEPS MASTER ACTION PLAN

## Ryzanstein Desktop AI Platform — Post-Sprint 6 Forward Roadmap

**Branch:** `sprint6/api-integration` @ `595465d`
**Status:** Weeks 1–4 + Sprints 5 & 6 committed; working tree dirty.

---

## 1. Executive Review (Completed Work)

| Phase          | Commit    | Deliverable                                      |
| -------------- | --------- | ------------------------------------------------ |
| Sprint 1.1     | `002a5a1` | Streaming API contract docs                      |
| Sprint 1.2     | `80a63cd` | Dead code removal, div-by-zero fixes, LogService |
| Sprint 1.3     | `0668dc2` | Circuit breaker Wails binding                    |
| Sprint 1.4     | `95df508` | Build fix, GetRecentLogs                         |
| Week 2         | `8a66ada` | Retry/timeout matrix                             |
| Week 3         | `b99aa3b` | IPC router                                       |
| Sprint 3.1     | `7226a6c` | IPC JSON router refinement                       |
| Sprint 3.2     | `9588d61` | VS Code extension fixes                          |
| Week 4         | `1763b20` | agent-memory ADR, Go memory client               |
| Sprint 4.1     | `a83e53c` | InvokeAgentChat + RyzansteinClient               |
| Sprint 4.2–4.4 | `595465d` | ADR-042, ADR-043, agentmem persistence           |
| Sprint 5       | `fdbf148` | Streaming decompression client                   |
| Sprint 6       | `e673298` | Integration Orchestrator                         |

---

## 2. Current State Assessment

- **HEAD:** `595465d` on `sprint6/api-integration`, in sync with `origin`.
- **⚠️ Ancestry concern:** Sprints 5 (`fdbf148`) and 6 (`e673298`) appear in reflog but may not be ancestors of HEAD. Verify with `git log --oneline --all --graph --decorate` before merging to main.
- **Dirty working tree:**
  - Modified: `desktop/frontend/wailsjs/go/main/App.{d.ts,js}` (regenerated bindings)
  - Modified: `docs/adr/ADR-042-agent-memory-architecture.md`, `ADR-043-compression-strategy.md`
  - Modified: `vscode-extension/src/client/RyzansteinClient.ts`, `src/commands/CommandHandler.ts`
  - Modified submodules: `sigma-compress`, `sigma-diff`, `sigma-index`
  - Untracked submodule dirs: `agentmem`, `ann-hybrid`, `causedb`, `cpu-infer`, `dep-bloom`, `mcp-mesh`, `semlog`

---

## 3. Known Gaps & Technical Debt

### P0 — Blocking

1. **Verify HEAD ancestry** of Sprint 5/6 commits; cherry-pick or merge if orphaned.
2. **Reconcile uncommitted wailsjs bindings** — regenerate via `wails generate` and commit deterministically.
3. **Reconcile uncommitted ADR-042/043 edits** — review diffs, finalize, commit.
4. **Submodule hygiene** — pin `sigma-*` submodule SHAs, register untracked submodule dirs in `.gitmodules`.

### P1 — Quality

5. VS Code extension uncommitted changes (`RyzansteinClient.ts`, `CommandHandler.ts`) need review + test.
6. No E2E test suite covering Desktop ↔ Ryzanstein streaming pipeline end-to-end.
7. No CI gating on `wails build` or `go test ./...` for `desktop/`.

### P2 — Hardening

8. Telemetry pipeline (`sigma-telemetry`) not wired into Desktop runtime.
9. Audit log integration (`zkaudit`, `vault-git`) deferred.
10. No code-signing / notarization story for Windows / macOS releases.

---

## 4. Phase 2 Sprint Roadmap (Weeks 6+)

### Sprint 7 — Working Tree Reconciliation (1–2 days)

- Verify git ancestry; rebase/cherry-pick orphaned Sprint 5/6 commits if needed.
- Regenerate Wails bindings cleanly; commit.
- Finalize ADR-042/043; commit.
- Fix submodule registration; commit `.gitmodules`.
- Open PR `sprint6/api-integration` → `main`.

### Sprint 8 — End-to-End QA

- Wire Playwright/WebDriver against built Wails binary.
- Cover: agent invoke → memory persist → streaming decompress → UI render.
- Add `go test ./desktop/...` to CI.

### Sprint 9 — Performance & Observability

- Integrate `sigma-telemetry` exporter into Desktop runtime.
- Profile streaming decompression hot path (`cpu-infer`, `sigma-compress`).
- Establish P50/P95/P99 latency baselines per IPC route.

### Sprint 10 — Security Hardening

- Wire `zkaudit` Merkle audit chain for agent invocations.
- Wire `vault-git` for secret material.
- Threat model + STRIDE review on IPC surface.

### Sprint 11 — Packaging & Release

- Code-signing pipeline (Windows Authenticode, macOS notarization).
- Auto-update channel.
- Installer for Windows (MSIX), macOS (DMG), Linux (AppImage).

### Sprint 12 — Advanced Agent Memory

- Hybrid ANN (`ann-hybrid`) integration with `agentmem`.
- Causal graph queries (`causedb`) exposed to agent context.
- Memory eviction & summarization policies.

---

## 5. Priority Matrix

| Priority | Sprint                        | Risk if Deferred                                 |
| -------- | ----------------------------- | ------------------------------------------------ |
| **P0**   | Sprint 7 (reconciliation)     | Cannot merge to main; blocks all downstream work |
| **P0**   | Sprint 8 (E2E QA)             | Regressions ship undetected                      |
| **P1**   | Sprint 9 (perf/observability) | No production visibility                         |
| **P1**   | Sprint 11 (packaging)         | Cannot distribute                                |
| **P2**   | Sprint 10 (security)          | Acceptable for alpha, blocker for v1.0           |
| **P2**   | Sprint 12 (advanced memory)   | Feature, not blocker                             |

---

## 6. Branch Strategy

- **Now:** `sprint6/api-integration` → land Sprint 7 reconciliation commits.
- **Then:** Open PR → `main`. Squash-merge after CI green + 1 review.
- **Going forward:** trunk-based with short-lived `sprintN/*` branches; each merges to `main` weekly.
- **Tags:** `v0.1.0-alpha` after Sprint 8; `v0.5.0-beta` after Sprint 11; `v1.0.0` after Sprint 12.

---

## 7. Testing Strategy

| Layer         | Tool                | Owner                                                                |
| ------------- | ------------------- | -------------------------------------------------------------------- |
| Unit (Go)     | `go test ./...`     | desktop/, mcp-mesh/, neurectomy-shell/, vault-git/                   |
| Unit (Rust)   | `cargo test`        | sigma-\*, ann-hybrid, causedb, cpu-infer, dep-bloom, semlog, zkaudit |
| Unit (TS)     | `vitest` / `jest`   | flowstate/, intent-spec/, vscode-extension/                          |
| Unit (Python) | `pytest`            | agentmem/, archaeo/                                                  |
| Integration   | custom harness      | IPC router, streaming pipeline                                       |
| E2E           | Playwright on Wails | full Desktop UX                                                      |
| CI            | GitHub Actions      | matrix: windows-latest, macos-latest, ubuntu-latest                  |

**Coverage target:** 80% lines on critical paths (IPC router, streaming codec, agent memory).

---

## 8. Release Milestones

| Milestone        | Gate                                      | Target          |
| ---------------- | ----------------------------------------- | --------------- |
| **v0.1.0-alpha** | Sprint 8 complete; E2E green              | After Sprint 8  |
| **v0.2.0**       | Sprint 9 complete; telemetry live         | After Sprint 9  |
| **v0.5.0-beta**  | Sprints 10–11 complete; signed installers | After Sprint 11 |
| **v1.0.0**       | Sprint 12 complete; advanced memory GA    | After Sprint 12 |

---

## Immediate Next Action (Sprint 7, Day 1)

```powershell
cd S:\Ryot
git log --oneline --all --graph --decorate | Select-Object -First 30
git status
# Reconcile working tree per Section 3 P0 items, then:
git checkout -b sprint7/reconciliation
```

**End of plan.**
