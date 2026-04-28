# ADR-044: Submodule Pollution Silencing Strategy

| Field       | Value                                         |
| ----------- | --------------------------------------------- |
| **Status**  | Accepted                                      |
| **Date**    | 2025-07-17                                    |
| **Authors** | Ryzanstein Core Team                          |
| **Relates** | ADR-045 (Submodule Working Tree Hygiene)      |
| **Commit**  | `56cf960` (sprint7/reconciliation)            |

## Context

The Ryzanstein monorepo embeds 18 sibling repositories under `dependencies/`
as Git submodules (org `iamthegreatdestroyer`). During Sprint 6 development,
several submodule working trees accumulated **untracked pollution** from
local tooling and IDE artifacts, primarily:

- `.history/` directories produced by the **Local History** VS Code
  extension (per-file timestamped snapshots).
- Editor scratch files, stale build artifacts, and ad-hoc notes.

Running `git status` from the superproject reported every polluted submodule
as `dirty`, producing noise like:

```
 m dependencies/agentmem      (untracked content)
 m dependencies/ann-hybrid    (untracked content)
 m dependencies/causedb       (untracked content)
 m dependencies/cpu-infer     (untracked content)
 m dependencies/dep-bloom     (untracked content)
 m dependencies/mcp-mesh      (untracked content)
 m dependencies/semlog        (untracked content)
 m dependencies/sigma-compress (untracked content)
```

This noise:

1. Masked **real** in-flight changes during reconciliation.
2. Made `git status` unusable as a sanity check before commits.
3. Risked accidental commits of IDE artifacts into submodule history if
   developers ran `git add -A` from inside a submodule.

The pollution was **not committable** in the superproject (untracked content
inside a submodule is invisible to the parent's index) but it disrupted
workflow and obscured signal.

## Decision

Adopt a **two-layer silencing strategy** that suppresses pollution at the
superproject level **without** modifying submodule working trees or histories.

### Layer 1: Superproject `.gitignore`

Add a global exclusion for the Local History extension's output directory:

```gitignore
# .gitignore (superproject root)
**/.history/
```

The `**/` glob ensures the rule fires at any nesting depth, including inside
every submodule working tree. Git applies the superproject `.gitignore` when
walking submodule trees during status reporting.

### Layer 2: `.gitmodules` `ignore = untracked`

For the 8 submodules with persistent untracked pollution, add an explicit
`ignore = untracked` directive in `.gitmodules`:

```ini
[submodule "dependencies/agentmem"]
    path = dependencies/agentmem
    url = https://github.com/iamthegreatdestroyer/agentmem.git
    ignore = untracked

[submodule "dependencies/ann-hybrid"]
    path = dependencies/ann-hybrid
    url = https://github.com/iamthegreatdestroyer/ann-hybrid.git
    ignore = untracked

[submodule "dependencies/causedb"]
    path = dependencies/causedb
    url = https://github.com/iamthegreatdestroyer/causedb.git
    ignore = untracked

[submodule "dependencies/cpu-infer"]
    path = dependencies/cpu-infer
    url = https://github.com/iamthegreatdestroyer/cpu-infer.git
    ignore = untracked

[submodule "dependencies/dep-bloom"]
    path = dependencies/dep-bloom
    url = https://github.com/iamthegreatdestroyer/dep-bloom.git
    ignore = untracked

[submodule "dependencies/mcp-mesh"]
    path = dependencies/mcp-mesh
    url = https://github.com/iamthegreatdestroyer/mcp-mesh.git
    ignore = untracked

[submodule "dependencies/semlog"]
    path = dependencies/semlog
    url = https://github.com/iamthegreatdestroyer/semlog.git
    ignore = untracked

[submodule "dependencies/sigma-compress"]
    path = dependencies/sigma-compress
    url = https://github.com/iamthegreatdestroyer/sigma-compress.git
    ignore = untracked
```

The setting `ignore = untracked` instructs `git status` and `git diff` in the
superproject to **skip** untracked-file checks when inspecting that submodule.
Tracked-file modifications and HEAD pointer changes remain visible.

### Scope

The 10 submodules **not** listed (`archaeo`, `flowstate`, `intent-spec`,
`neurectomy-shell`, `sigma-api`, `sigma-diff`, `sigma-index`, `sigma-telemetry`,
`vault-git`, `zkaudit`) had clean working trees at reconciliation time and
do not need the directive. They will be added on demand if pollution recurs.

## Consequences

### Positive

- `git status` from superproject is **clean** — only real changes shown.
- Pre-commit sanity checks become trustworthy again.
- No mutation of submodule working trees or histories — fully reversible.
- Developers can still inspect submodule pollution with
  `git -C dependencies/<name> status` when debugging.
- `.history/` rule applies everywhere via `**/` glob — defense in depth.

### Negative

- A genuinely new untracked file inside a silenced submodule will not
  surface in superproject `git status`. Developers must `cd` into the
  submodule and run `git status` directly to see it.
- Onboarding documentation must mention the silencing convention so new
  contributors don't believe submodules are pristine.

### Risks

| Risk                                                 | Mitigation                                           |
| ---------------------------------------------------- | ---------------------------------------------------- |
| Untracked but important file goes unnoticed          | ADR-045 mandates periodic `git status` per submodule |
| Setting drifts (someone removes `ignore = untracked`)| Lint rule in CI (future) to verify `.gitmodules`     |
| New polluted submodule added without directive       | ADR-045 hygiene policy + reviewer checklist          |

## Alternatives Considered

| Alternative                                  | Why Rejected                                                                  |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| Add `.gitignore` inside each submodule       | Mutates submodule history; requires upstream PR per repo; high friction       |
| Vendor sources instead of using submodules   | Loses upstream sync benefits; doubles the maintenance burden                  |
| Fork and clean each polluted submodule       | Duplicates 8 repos; fork divergence; unsustainable                            |
| Global `core.excludesFile` only              | Local-only; doesn't apply to teammates / CI; not version-controlled           |
| `ignore = dirty`                             | Hides tracked-file modifications too — loses real signal during edits         |
| `ignore = all`                               | Hides HEAD pointer changes — would break reconciliation entirely              |

`ignore = untracked` is the **minimum-mutation** choice: it silences exactly
the noise we want to silence and preserves all other signal.

## Implementation

Implemented in commit `56cf960` on branch `sprint7/reconciliation`:

- `s:\Ryot\.gitignore` — added `**/.history/`.
- `s:\Ryot\.gitmodules` — added 8 `ignore = untracked` entries.

Verification (cycle 144, post-commit):

```
$ git status --short
 m dependencies/sigma-compress
```

The remaining single line reflects a **tracked** modification in
`sigma-compress` (HEAD `84783f2`, 1 ahead of `origin/main`) — real signal,
not pollution. ADR-045 covers the policy for resolving such states.
