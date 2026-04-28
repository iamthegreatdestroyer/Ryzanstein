# ADR-046: E2E Framework Selection — Playwright over CDP for Wails/WebView2

- **Status**: Accepted
- **Date**: 2026-01-14
- **Sprint**: 8 (E2E QA)
- **Deciders**: @ECLIPSE @APEX @BRIDGE
- **Supersedes**: —
- **Related**: ADR-044, ADR-045

## Context

Sprint 8 entry gate of the Master Action Plan requires end-to-end coverage
of the desktop application path:

> agent invoke → memory persist → streaming decompress → UI render

The desktop binary is a Wails v2 app:

- **Backend**: Go, bound methods exposed to JS via Wails runtime.
- **Frontend**: Svelte 4 + Vite 5 + TypeScript (`desktop/frontend/`).
- **Runtime host (Windows)**: WebView2 (Chromium-based Edge engine).

We need an automation harness that can:

1. Launch the built `desktop.exe` (production-equivalent code path).
2. Drive the embedded webview programmatically — click, type, assert DOM
   state, intercept network/IPC, capture screenshots/traces.
3. Run in CI on Windows runners without a physical display when feasible,
   or with `actions/setup-virtual-display` style fallbacks.
4. Express tests in TypeScript so they share the frontend's type system
   (enables importing generated Wails bindings for type-safe assertions).

## Options Considered

### Option A — Playwright (chosen)

- TypeScript-native, matches existing frontend toolchain.
- Connects to WebView2 via Chrome DevTools Protocol (CDP) using
  `chromium.connectOverCDP('http://localhost:<port>')` after launching
  the binary with
  `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=<port>`.
- First-class trace viewer, video recording, auto-wait selectors,
  network interception (useful for verifying streaming decompression
  chunks).
- Mature ecosystem: parallel sharding, retry logic, fixtures,
  `expect()` library.
- License: Apache-2.0.

### Option B — chromedp (Go-native)

- Same CDP transport; consistent with monorepo Go bias.
- Pros: single language across backend + tests; no extra Node toolchain
  in CI.
- Cons: no auto-wait; manual `Sleep`/poll patterns common; weaker
  assertion ergonomics; no built-in trace viewer; Svelte
  component-aware selectors absent (must hand-roll
  `document.querySelector` calls). DX significantly worse for UI flows.

### Option C — WebDriver / Selenium (msedgedriver against WebView2)

- Standardized, but driver setup on Windows is fragile (must match
  WebView2 runtime version).
- No CDP-level network interception without bridges.
- Slower test feedback; the Wails community has migrated away from this
  approach.

### Option D — Cypress

- Cannot attach to an external browser process; requires loading the app
  inside its own Chromium. Incompatible with launching a packaged
  WebView2 binary.

## Decision

Adopt **Playwright** (Option A) under `desktop/e2e/`.

Test architecture:

```
desktop/e2e/
├── package.json              # @playwright/test, typescript
├── playwright.config.ts      # webServer = launch desktop.exe, CDP port
├── fixtures/
│   └── wailsApp.ts           # custom fixture: spawn binary, connect CDP,
│                             # yield BrowserContext + Page
├── tests/
│   ├── smoke.spec.ts         # window opens, root component mounts
│   ├── agent-invoke.spec.ts  # InvokeAgentChat → response rendered
│   ├── memory-persist.spec.ts# write, restart, read back
│   └── streaming.spec.ts     # streaming decompress UI updates
└── tsconfig.json
```

Launch contract:

```ts
// fixtures/wailsApp.ts (sketch)
const port = await getFreePort();
const proc = spawn(binaryPath, [], {
  env: {
    ...process.env,
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}`,
  },
});
const browser = await chromium.connectOverCDP(`http://localhost:${port}`);
const context = browser.contexts()[0];
const page = context.pages()[0] ?? await context.waitForEvent('page');
```

## Consequences

### Positive

- Shares TypeScript with the Svelte frontend; can `import` generated
  Wails bindings (`frontend/wailsjs/go/main/*`) for type-checked
  assertions on backend method signatures.
- Trace viewer dramatically reduces flake-debugging time on CI.
- Network interception lets us assert that streaming decompression
  produces the expected chunk cadence without coupling to internal Go
  state.

### Negative

- Adds a Node devDependency tree under `desktop/e2e/` (~200 MB
  `node_modules`). Mitigation: hoist or pin via lockfile; keep separate
  from `desktop/frontend/` to avoid Vite resolver conflicts.
- WebView2 CDP support depends on runtime version; pin minimum WebView2
  Evergreen version in build prerequisites.
- Requires Windows-hosted CI runners for the integration job.

### Neutral

- Test execution time will be measured during Task 3 (CI integration);
  budget is 5 minutes for the Sprint 8 smoke matrix.

## Validation

Acceptance criteria for Sprint 8 Task 1 closure:

1. `desktop/e2e/` scaffold exists with `playwright.config.ts` and at
   least one passing smoke test that:
   - Launches `desktop.exe` (or `wails dev` in fallback mode).
   - Connects via CDP.
   - Asserts the root Svelte component mounts.
2. ADR-046 (this document) committed.
3. `package.json` script `npm run e2e` runs the suite locally.

## References

- Playwright CDP docs: <https://playwright.dev/docs/api/class-browsertype#browser-type-connect-over-cdp>
- WebView2 remote debugging: <https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/overview-features-apis#remote-debugging>
- Wails v2 environment variables: <https://wails.io/docs/reference/options>
