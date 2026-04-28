# ADR-045: Submodule Working Tree Hygiene Policy

| Field       | Value                                              |
| ----------- | -------------------------------------------------- |
| **Status**  | Accepted                                           |
| **Date**    | 2025-07-17                                         |
| **Authors** | Ryzanstein Core Team                               |
| **Relates** | ADR-044 (Submodule Pollution Silencing Strategy)   |
| **Sprint**  | Sprint 7 — Working Tree Reconciliation             |

## Context

ADR-044 silenced **untracked pollution noise** at the superproject level by
combining a global `.gitignore` rule with `ignore = untracked` directives in
`.gitmodules`. That decision restored signal-to-noise on `git status` but
introduced a new operational concern:

> **If untracked changes inside silenced submodules are invisible from the
> superproject, how do we ensure genuinely meaningful changes don't go
> unnoticed?**

We also lack a documented procedure for the recurring lifecycle events of a
submodule working tree:

1. **Reconciliation** — moving a dirty submodule to a clean, committed state.
2. **Periodic audit** — verifying silenced submodules haven't accumulated
   real (non-pollution) untracked work.
3. **New submodule onboarding** — deciding whether to add `ignore = untracked`
   from day one.
4. **Drift detection** — catching submodule HEAD pointers that diverge from
   intended targets.

This ADR establishes the **policy** layer that complements ADR-044's
**mechanism** layer.

## Decision

Adopt the following hygiene policy for all submodules under `dependencies/`.

### 1. Reconciliation Procedure

When a submodule working tree is dirty and must be brought to a clean state
recorded by the superproject:

```powershell
# From the superproject root:

# Step 1 — Inspect the submodule directly
git -C dependencies/<name> status
git -C dependencies/<name> log --oneline -5

# Step 2 — If real work exists, commit & push within the submodule
git -C dependencies/<name> add <files>
git -C dependencies/<name> commit -m "<message>"
git -C dependencies/<name> push origin <branch>

# Step 3 — Capture the new HEAD in the superproject
git add dependencies/<name>
git commit -m "submodule: bump <name> to <short-sha>"

# Step 4 — Verify
git status --short        # superproject should be clean
git submodule status      # all submodules should report clean SHAs
```

Pollution-only working trees (no real work) require **no action** — the
ADR-044 silencing layer absorbs them.

### 2. Periodic Audit Cadence

Run a **submodule audit** at the following triggers:

| Trigger                                | Action                                              |
| -------------------------------------- | --------------------------------------------------- |
| Start of every sprint                  | Full audit (all 18 submodules)                      |
| Before opening a release PR            | Full audit                                          |
| After any `git submodule update`       | Quick audit (silenced submodules only)              |
| Weekly (recommended, not enforced)     | Quick audit                                         |

**Audit script** (target for `s:\Ryot\scripts\audit-submodules.ps1`):

```powershell
$submodules = git config --file .gitmodules --get-regexp path |
              ForEach-Object { ($_ -split ' ')[1] }

foreach ($sm in $submodules) {
    Write-Host "=== $sm ===" -ForegroundColor Cyan
    git -C $sm status --short
    git -C $sm log --oneline origin/main..HEAD 2>$null
}
```

The audit surfaces:

- Untracked files that escaped the `**/.history/` filter.
- Local commits not yet pushed to origin.
- Detached-HEAD states that the superproject doesn't track.

### 3. New Submodule Onboarding

When adding a new submodule:

1. Run `git submodule add <url> dependencies/<name>` from superproject root.
2. **Default to NOT silencing** — only add `ignore = untracked` after
   pollution is observed and confirmed non-actionable.
3. Add an entry to the dependency map in `s:\Ryot\docs\DEPENDENCIES.md`
   (future) describing the submodule's role.
4. Verify CI builds pick up the new submodule.

### 4. Drift Detection

The superproject records the **exact SHA** each submodule should be at.
Drift occurs when:

- A developer commits inside a submodule but forgets to bump the superproject.
- A `git pull` updates the superproject's recorded SHA but the developer's
  local submodule isn't synced via `git submodule update`.

**Detection**: `git submodule status` shows a `+` prefix on drifted entries.
**Resolution**: either bump the superproject (Step 3 of §1) or run
`git submodule update --recursive` to align local trees.

### 5. Reviewer Checklist

PR reviewers must confirm:

- [ ] If the PR touches `.gitmodules`, ADR-044/045 invariants still hold.
- [ ] If the PR bumps a submodule SHA, the bump message references the
      submodule's commit and PR.
- [ ] If a new submodule is added, this ADR's onboarding steps were followed.

## Consequences

### Positive

- Silenced submodules remain auditable through documented procedure.
- Reconciliation has a single canonical recipe — reduces tribal knowledge.
- New contributors have a clear policy to follow.
- Drift is caught before it causes mysterious build failures.

### Negative

- Periodic audits add operational overhead (~5 min/sprint when scripted).
- Policy compliance depends on developer discipline until the audit script
  is automated in CI.

### Risks

| Risk                                              | Mitigation                                          |
| ------------------------------------------------- | --------------------------------------------------- |
| Audit cadence skipped during deadline pressure    | Sprint-start trigger is hard-coded into ritual      |
| Audit script bitrots                              | Include in CI smoke-test once authored              |
| Reviewer checklist ignored                        | Add as PR template section in `.github/`            |
| New submodule added with silencing too eagerly    | Onboarding §3 default-off + reviewer checklist      |

## Alternatives Considered

| Alternative                                 | Why Rejected                                                            |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| No policy — rely on ad-hoc inspection       | Recreates the Sprint 6 noise problem; loses institutional memory        |
| Daily mandatory audit                       | Overhead disproportionate to risk; audit fatigue                        |
| Disallow `ignore = untracked` entirely      | Reverses ADR-044; reintroduces noise                                    |
| Auto-commit pollution into submodules       | Pollutes upstream history; ethically wrong for forked repos             |
| Replace submodules with subtree merges      | Loses upstream sync ergonomics; massive migration cost                  |

## Implementation

This ADR is **policy** — implementation artifacts are tracked separately:

| Artifact                                        | Status      | Owner          |
| ----------------------------------------------- | ----------- | -------------- |
| `s:\Ryot\scripts\audit-submodules.ps1`          | TODO        | Sprint 8       |
| `.github\PULL_REQUEST_TEMPLATE.md` checklist    | TODO        | Sprint 8       |
| `s:\Ryot\docs\DEPENDENCIES.md` registry         | TODO        | Sprint 8       |
| Sprint-start audit calendar entry               | Manual      | Tech lead      |

Sprint 7 reconciliation (commit `56cf960`) validated the §1 procedure:
the resulting superproject working tree contained only one tracked
submodule modification (`dependencies/sigma-compress`), proving the policy
correctly distinguishes pollution from signal.
