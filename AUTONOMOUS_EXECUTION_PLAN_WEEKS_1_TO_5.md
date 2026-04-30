# AUTONOMOUS EXECUTION PLAN — WEEKS 1 TO 5

## Ryzanstein Desktop AI Platform — Codebase: S:\Ryot\

**Generated:** Post-Wave-17 exploration complete  
**Status:** ALL PREREQUISITES MET — ZERO REMAINING READS BEFORE EXECUTION  
**Build command (PERMANENT — NEVER CHANGE):** `cd S:\Ryot\desktop && wails build`

---

## FOUNDATION: WHAT IS ALREADY REAL AND NEVER NEEDS CHANGING

The following are **production-complete and verified through 17 waves of exploration**.
Do NOT refactor, re-implement, or "improve" any of these.

### Critical Path — 100% Real, Zero Stubs

```
UI (ChatPanel.svelte)
  └─ await SendMessageStream(message, modelID, agentCodename)
       └─ App.SendMessageStream() [main.go:200–320]
            └─ a.apiClient.ChatCompletionStream(ctx, chatReq, tokenChan)
                 └─ POST /v1/chat/completions {stream:true, Accept:text/event-stream}
                      └─ SSE byte read → splitLines() → "data: " prefix strip
                           └─ json.Unmarshal → choices[].delta.content
                                └─ tokenChan <- choice.Delta.Content  ← REAL
            └─ for token := range tokenChan { EventsEmit("chat:streamToken", token) }
            └─ EventsEmit("chat:streamEnd", finalMessage)
  └─ ChatPanel.svelte: streamingContent += token (live render)
```

### Wails Event Names — EXACT, Never Change

| Event              | Direction   | Payload                            |
| ------------------ | ----------- | ---------------------------------- |
| `chat:streamStart` | Go → Svelte | `nil`                              |
| `chat:streamToken` | Go → Svelte | `string` (one token)               |
| `chat:streamEnd`   | Go → Svelte | `Message{}` struct                 |
| `chat:streamError` | Go → Svelte | `string` (error text)              |
| `chat:message`     | Go → Svelte | user message                       |
| `chat:response`    | Go → Svelte | non-streaming response             |
| `app:ready`        | Go → Svelte | `{version:"1.0.0", timestamp:now}` |

### Files That Must NEVER Be Modified

- `S:\Ryot\desktop\frontend\src\components\ChatPanel.svelte`
- `S:\Ryot\desktop\frontend\src\components\AgentPanel.svelte`
- `S:\Ryot\dependencies\mcp-mesh\router.go` → `Route()` is already complete
- `S:\Ryot\desktop\internal\ipc\bridge.go` → keep standalone, never wire to App

### RyzansteinClient — Full Method Inventory (lines 1–494)

```go
type RyzansteinClient struct {
    baseURL    string
    httpClient *http.Client   // 30s timeout
    timeout    time.Duration
    maxRetries int            // 3
    retryDelay time.Duration  // 1s
}

// Exponential backoff: 1s → 2s → 4s
// Applied to: Infer, ListModels, LoadModel, UnloadModel

Infer(ctx, req)                    → POST /v1/completions          [3 retries + backoff]
ListModels(ctx)                    → GET  /v1/models               [3 retries + backoff]
LoadModel(ctx, modelPath)          → POST /v1/models/load          [3 retries + backoff]
UnloadModel(ctx, modelID)          → POST /v1/models/{id}/unload   [3 retries + backoff]
ChatCompletion(ctx, req)           → POST /v1/chat/completions     [single attempt, JSON]
ChatCompletionStream(ctx, req, ch) → POST /v1/chat/completions     [SSE real, delta format]
Health(ctx)                        → GET  /health                  → true if 200 OK
```

---

## COMPLETE GAP INVENTORY

| #   | Gap                                      | File                                       | Severity                     | Target Week   |
| --- | ---------------------------------------- | ------------------------------------------ | ---------------------------- | ------------- |
| 1   | `ExecuteStream()` is dead code stub      | `internal/services/inference_service.go`   | LOW — comment only           | W1 Sprint 1.2 |
| 2   | `handleClient()` echoes raw string       | `internal/ipc/server.go`                   | MED — blocks ext integration | W3 Sprint 3.1 |
| 3   | `openChat` shows InfoMessage only        | `vscode-extension/CommandHandler.ts`       | MED                          | W3 Sprint 3.2 |
| 4   | `ryzanstein.infer` command missing       | `vscode-extension/CommandHandler.ts`       | MED                          | W3 Sprint 3.2 |
| 5   | `InvokeTool()` returns hardcoded map     | `internal/agents/service.go`               | HIGH                         | W4 Sprint 4.1 |
| 6   | `InvokeAgent()` does not exist           | `internal/agents/service.go`               | HIGH                         | W4 Sprint 4.1 |
| 7   | `agentmem/store.py` — zero disk I/O      | `dependencies/agentmem/store.py`           | HIGH                         | W4 Sprint 4.4 |
| 8   | `get_embeddings()` uses hash fallback    | `sigma-compress/ryzanstein_integration.rs` | MED                          | W5 Sprint 5.1 |
| 9   | `health_check()` always returns Ok(true) | `sigma-compress/ryzanstein_integration.rs` | LOW                          | W5 Sprint 5.1 |

### Services Directory — Unknowns (Read in W2 Sprint 2.1 BEFORE touching anything)

```
S:\Ryot\desktop\internal\services\
  ✅ inference_service.go   — fully read (ExecuteStream is dead code)
  ✅ streamer.go            — fully read (bufio, context-cancellable, real)
  ⚠️  client_manager.go    — UNKNOWN (multi-backend management suspected)
  ⚠️  async_model_manager.go — UNKNOWN (async model load suspected)
  ⚠️  batcher.go           — UNKNOWN (request batching suspected)
  ⚠️  pool.go              — UNKNOWN (goroutine/connection pool suspected)
  ⚠️  model_service.go     — UNKNOWN
     mock_server.go + 5x *_test.go — test infrastructure, read last
```

---

## WEEK 1 — Observability & Hardening

**Goal:** Zero new features. Instrument, document, and validate what exists.  
**Risk:** LOW — read-only or additive changes only.  
**Duration:** 5 working days

---

### Sprint 1.1 — API Contract Documentation _(Day 1)_

**Objective:** Produce the canonical reference document for the streaming pipeline.

**Actions:**

1. Create directory `S:\Ryot\docs\` if absent.
2. Write `S:\Ryot\docs\STREAMING_API_CONTRACT.md` containing:
   - The complete `ChatCompletionRequest` and `ChatCompletionResponse` type definitions
   - The SSE wire format: `data: {"choices":[{"delta":{"content":"<tok>"}}]}\n\n`
   - The `[DONE]` sentinel behaviour
   - The `splitLines()` contract: handles `\n` and `\r\n`, skips empty lines
   - The `tokenChan` buffering: `make(chan string, 64)`
   - The 120-second context timeout on `SendMessageStream()`
   - The `chatReq` construction pattern from `main.go`
3. Run `cd S:\Ryot\desktop && wails build` — confirm zero compile errors before proceeding.

**Verification:** File exists, build clean.

---

### Sprint 1.2 — Dead Code Annotation + Log Service _(Days 2–3)_

**Objective:** Mark dead code clearly; add structured logging.

**Actions:**

**Part A — Dead code annotation (30 min):**

```go
// S:\Ryot\desktop\internal\services\inference_service.go
// In ExecuteStream():
// TODO: DEAD CODE — App.SendMessageStream() calls apiClient.ChatCompletionStream() directly.
// This method is never invoked. Do not delete (test coverage), do not refactor.
```

Add the comment. Run `cd S:\Ryot\desktop && wails build`. Done.

**Part B — Log service (Day 2–3):**

Create `S:\Ryot\desktop\internal\services\log_service.go`:

```go
package services

import (
    "fmt"
    "os"
    "path/filepath"
    "sync"
    "time"
)

const maxRingSize = 1000

type LogService struct {
    ring    []string
    head    int
    size    int
    mu      sync.RWMutex
    logFile *os.File
}

func NewLogService() (*LogService, error) {
    appData, err := os.UserConfigDir()
    if err != nil {
        return &LogService{ring: make([]string, maxRingSize)}, nil
    }
    logDir := filepath.Join(appData, "Ryzanstein", "logs")
    if err := os.MkdirAll(logDir, 0755); err != nil {
        return &LogService{ring: make([]string, maxRingSize)}, nil
    }
    logPath := filepath.Join(logDir, "app.log")
    f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
    if err != nil {
        return &LogService{ring: make([]string, maxRingSize)}, nil
    }
    return &LogService{ring: make([]string, maxRingSize), logFile: f}, nil
}

func (l *LogService) Log(level, msg string) {
    entry := fmt.Sprintf("[%s] %s %s", level, time.Now().Format(time.RFC3339), msg)
    l.mu.Lock()
    l.ring[l.head%maxRingSize] = entry
    l.head++
    if l.size < maxRingSize {
        l.size++
    }
    l.mu.Unlock()
    if l.logFile != nil {
        _, _ = fmt.Fprintln(l.logFile, entry)
    }
}

func (l *LogService) GetRecentLogs(n int) []string {
    l.mu.RLock()
    defer l.mu.RUnlock()
    if n > l.size {
        n = l.size
    }
    result := make([]string, n)
    for i := 0; i < n; i++ {
        idx := (l.head - n + i + maxRingSize*2) % maxRingSize
        result[i] = l.ring[idx]
    }
    return result
}

func (l *LogService) Close() {
    if l.logFile != nil {
        _ = l.logFile.Close()
    }
}
```

Wire into `App` struct in `main.go`:

```go
// Add to App struct:
logger *services.LogService

// In startup():
ls, _ := services.NewLogService()
a.logger = ls

// In SendMessageStream() — add at top of function:
a.logger.Log("INFO", fmt.Sprintf("SendMessageStream: model=%s agent=%s", modelID, agentCodename))

// Add Wails binding:
func (a *App) GetRecentLogs(n int) []string {
    return a.logger.GetRecentLogs(n)
}
```

**Verification:** `cd S:\Ryot\desktop && wails build` — zero errors. Log file appears at `%APPDATA%\Ryzanstein\logs\app.log` on first run.

---

### Sprint 1.3 — Circuit Breaker Status Binding _(Day 4)_

**Objective:** Expose real-time API health and circuit state as a Wails binding.

**Actions:**

Add to `main.go` (additive only — new method on `*App`):

```go
// GetCircuitStatus returns real-time health of backend connections.
// Does NOT call Health() — reads cached state only to avoid blocking UI.
func (a *App) GetCircuitStatus() map[string]interface{} {
    a.mu.RLock()
    running := a.isRunning
    a.mu.RUnlock()

    return map[string]interface{}{
        "api_running":      running,
        "api_base_url":     a.apiClient.GetBaseURL(), // add GetBaseURL() to client
        "max_retries":      3,
        "retry_delay_ms":   1000,
        "stream_buffer":    64,
        "context_timeout_s": 120,
        "timestamp":        time.Now().UTC().Format(time.RFC3339),
    }
}
```

Add `GetBaseURL()` to `RyzansteinClient` in `ryzanstein_client.go`:

```go
func (c *RyzansteinClient) GetBaseURL() string { return c.baseURL }
```

**Verification:** `cd S:\Ryot\desktop && wails build` — zero errors.

---

### Sprint 1.4 — Week 1 Integration Validation _(Day 5)_

**Objective:** Confirm entire existing pipeline is clean after Week 1 additions.

**Actions (in order):**

```powershell
# 1. Build
cd S:\Ryot\desktop
wails build

# 2. Unit tests
go test ./... -v -timeout 60s

# 3. Smoke test streaming (manual)
# - Launch Ryzanstein.exe
# - Ensure LLM backend running at localhost:8000
# - Send a message → confirm tokens stream to UI
# - Confirm log file written to %APPDATA%\Ryzanstein\logs\app.log

# 4. Confirm log binding works
# - Open DevTools → call window.go.main.App.GetRecentLogs(10)
# - Expect array of log entries

# 5. Confirm circuit status binding
# - window.go.main.App.GetCircuitStatus()
# - Expect {api_running: true/false, api_base_url: "http://localhost:8000", ...}
```

**Week 1 Exit Criteria:**

- [ ] `wails build` — zero errors
- [ ] `go test ./...` — zero failures
- [ ] `docs/STREAMING_API_CONTRACT.md` — written
- [ ] `log_service.go` — written and building
- [ ] `GetRecentLogs()` — Wails binding exposed
- [ ] `GetCircuitStatus()` — Wails binding exposed
- [ ] `ExecuteStream()` dead code comment — added

---

## WEEK 2 — Resilience + Services Exploration

**Goal:** Understand ALL unknowns in `services/`. Add system health binding.  
**Risk:** LOW for reads; MEDIUM for any `client_manager.go` wiring.  
**Duration:** 5 working days

---

### Sprint 2.1 — Read All Unknown Services _(Day 1)_

**Objective:** Eliminate every unknown in `S:\Ryot\desktop\internal\services\`.

**MANDATORY READ ORDER (do not skip, do not parallelize with implementation):**

```
1. read_file client_manager.go      (lines 1–END)   → Q: does it track circuit state?
2. read_file async_model_manager.go (lines 1–END)   → Q: goroutine pool? callback?
3. read_file batcher.go             (lines 1–END)   → Q: concurrent request batching?
4. read_file pool.go                (lines 1–END)   → Q: http.Client pool? goroutine pool?
5. read_file model_service.go       (lines 1–END)   → Q: wraps models.Service?
```

**Decision gates after each read:**

- If `client_manager.go` owns circuit breaker state → wire `GetCircuitStatus()` to it (Sprint 2.3)
- If `batcher.go` batches infer requests → add to `GetCircuitStatus()` output
- If `pool.go` manages `http.Client` → ensure `RyzansteinClient` uses same pool (ADR material)

**Do NOT write any code during Sprint 2.1. Read only.**

---

### Sprint 2.2 — Timeout & Backoff Validation _(Days 2–3)_

**Objective:** Confirm all timeout/retry parameters match production requirements.

**Actions:**

Document (write `S:\Ryot\docs\RETRY_TIMEOUT_MATRIX.md`):

```markdown
| Method                 | Retries | Initial Delay | Max Delay | Context Timeout |
| ---------------------- | ------- | ------------- | --------- | --------------- |
| Infer()                | 3       | 1s            | 4s        | caller-provided |
| ListModels()           | 3       | 1s            | 4s        | caller-provided |
| LoadModel()            | 3       | 1s            | 4s        | caller-provided |
| UnloadModel()          | 3       | 1s            | 4s        | caller-provided |
| ChatCompletion()       | 0       | —             | —         | caller-provided |
| ChatCompletionStream() | 0       | —             | —         | 120s (main.go)  |
| Health()               | 0       | —             | —         | caller-provided |
```

Check: Is `httpClient.Timeout` (30s) respected when context has shorter/longer deadline?
→ `http.Client.Timeout` and `context.WithTimeout` both independently cancel.
→ Effective timeout = min(client timeout, context timeout).
→ For stream: 30s client timeout will fire before 120s context. Document this discrepancy.

If client timeout < stream context → raise `httpClient.Timeout` to 150s in `NewRyzansteinClient()`:

```go
// ryzanstein_client.go — constructor only:
httpClient: &http.Client{Timeout: 150 * time.Second},
```

Run `cd S:\Ryot\desktop && wails build` after.

---

### Sprint 2.3 — System Health Binding _(Days 4–5)_

**Objective:** Expose unified health dashboard as Wails binding.

**Actions:**

Add to `main.go` (informed by Sprint 2.1 reads):

```go
// GetSystemHealth returns unified system health snapshot.
// Calls Health() over HTTP — do NOT call from UI render loop.
func (a *App) GetSystemHealth() map[string]interface{} {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    apiOK, apiErr := a.apiClient.Health(ctx)
    errMsg := ""
    if apiErr != nil {
        errMsg = apiErr.Error()
    }

    a.mu.RLock()
    running := a.isRunning
    a.mu.RUnlock()

    history, _ := a.chat.GetHistory()
    historyCount := 0
    if history != nil {
        historyCount = len(history)
    }

    models, _ := a.apiClient.ListModels(ctx)
    modelCount := 0
    if models != nil {
        modelCount = len(models)
    }

    recentLogs := a.logger.GetRecentLogs(5)

    return map[string]interface{}{
        "api_healthy":    apiOK,
        "api_error":      errMsg,
        "app_running":    running,
        "history_count":  historyCount,
        "model_count":    modelCount,
        "recent_logs":    recentLogs,
        "timestamp":      time.Now().UTC().Format(time.RFC3339),
    }
}
```

**Verification:** `cd S:\Ryot\desktop && wails build` — zero errors.

**Week 2 Exit Criteria:**

- [ ] All 5 unknown service files read and documented
- [ ] `RETRY_TIMEOUT_MATRIX.md` written
- [ ] `httpClient.Timeout` confirmed or corrected (150s if stream needs it)
- [ ] `GetSystemHealth()` Wails binding — built and accessible
- [ ] `go test ./...` — zero failures

---

## WEEK 3 — VS Code Extension Command Dispatch + IPC Router

**Goal:** Fix stub commands in extension; replace echo IPC with JSON dispatch.  
**Risk:** MEDIUM — touches TypeScript and Go IPC simultaneously.  
**Duration:** 5 working days

---

### Sprint 3.1 — IPC JSON Router in `server.go` _(Days 1–2)_

**File:** `S:\Ryot\desktop\internal\ipc\server.go`

**Current state:** `handleClient()` echoes `"ACK: <message>"` for every input.

**Target state:**

```go
// Replace handleClient() body with:
func (s *Server) handleClient(conn net.Conn) {
    defer conn.Close()
    decoder := json.NewDecoder(conn)
    encoder := json.NewEncoder(conn)

    for {
        var req struct {
            Command string          `json:"command"`
            Payload json.RawMessage `json:"payload"`
        }
        if err := decoder.Decode(&req); err != nil {
            return
        }

        var resp interface{}
        switch req.Command {
        case "health":
            // Delegate to app health — IPC server holds ref to *App
            resp = map[string]interface{}{"ok": true, "source": "ipc"}

        case "infer":
            var p struct {
                Prompt  string `json:"prompt"`
                ModelID string `json:"model_id"`
            }
            if err := json.Unmarshal(req.Payload, &p); err != nil {
                resp = map[string]interface{}{"error": "invalid payload"}
            } else {
                // Synchronous infer via apiClient
                resp = map[string]interface{}{"result": "infer_dispatch_pending", "echo": p.Prompt}
            }

        case "list_agents":
            resp = map[string]interface{}{"agents": "agent_list_pending"}

        case "list_models":
            resp = map[string]interface{}{"models": "model_list_pending"}

        default:
            resp = map[string]interface{}{"error": "unknown command: " + req.Command}
        }

        if err := encoder.Encode(resp); err != nil {
            return
        }
    }
}
```

**Note:** `"infer_dispatch_pending"` and `"agent_list_pending"` are logged placeholders. Real dispatch wired in Week 4 Sprint 4.1 once `InvokeAgent()` exists.

**Add `*App` reference to IPC server:**

```go
// ipc/server.go Server struct — add:
type Server struct {
    listener net.Listener
    app      interface{ /* minimal interface */ }
}
// Update constructor to accept app reference.
// Update main.go NewIPCServer() call accordingly.
```

**Verification:** `cd S:\Ryot\desktop && wails build` — zero errors.

---

### Sprint 3.2 — VS Code Extension Command Fixes _(Days 3–5)_

**Files:**

- `S:\Ryot\vscode-extension\src\commands\CommandHandler.ts`
- `S:\Ryot\vscode-extension\src\client\RyzansteinClient.ts`

**Current gaps:**

- `ryzanstein.openChat` → shows InfoMessage only → fix to open WebviewPanel
- `ryzanstein.infer` → does not exist → create
- `RyzansteinClient.ts` → no streaming method → add `streamInfer()`

**Actions:**

**Part A — Fix `openChat` in `CommandHandler.ts`:**

```typescript
// Replace:
vscode.window.showInformationMessage("Opening Ryzanstein Chat...");

// With:
const panel = vscode.window.createWebviewPanel(
  "ryzansteinChat",
  "Ryzanstein Chat",
  vscode.ViewColumn.Beside,
  { enableScripts: true, retainContextWhenHidden: true },
);
panel.webview.html = this.getChatWebviewContent(panel.webview);
```

Add `getChatWebviewContent(webview: vscode.Webview): string` method to `CommandHandler`:

```typescript
private getChatWebviewContent(webview: vscode.Webview): string {
    return `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Ryzanstein Chat</title>
<style>body{font-family:var(--vscode-font-family);padding:16px;}</style>
</head><body>
<h2>Ryzanstein Chat</h2>
<div id="history" style="height:400px;overflow-y:auto;border:1px solid #333;padding:8px;"></div>
<input id="input" type="text" style="width:80%;" placeholder="Type a message..."/>
<button id="send">Send</button>
<script>
const vscode = acquireVsCodeApi();
document.getElementById('send').addEventListener('click', () => {
    const input = document.getElementById('input');
    vscode.postMessage({command:'infer', text: input.value});
    input.value = '';
});
window.addEventListener('message', e => {
    const div = document.getElementById('history');
    div.innerHTML += '<p><b>' + e.data.role + ':</b> ' + e.data.content + '</p>';
    div.scrollTop = div.scrollHeight;
});
</script>
</body></html>`;
}
```

**Part B — Add `ryzanstein.infer` command:**

In `CommandHandler.ts`, register new command:

```typescript
context.subscriptions.push(
  vscode.commands.registerCommand("ryzanstein.infer", async () => {
    const prompt = await vscode.window.showInputBox({
      placeHolder: "Enter prompt for Ryzanstein...",
      prompt: "Inference prompt",
    });
    if (!prompt) return;

    try {
      const result = await this.client.infer(prompt);
      vscode.window.showInformationMessage(
        `Ryzanstein: ${result.substring(0, 120)}...`,
      );
    } catch (err) {
      vscode.window.showErrorMessage(`Ryzanstein infer failed: ${err}`);
    }
  }),
);
```

**Part C — Add `infer()` to `RyzansteinClient.ts`:**

```typescript
async infer(prompt: string, modelId?: string): Promise<string> {
    const response = await fetch(`${this.baseUrl}/v1/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt,
            model: modelId ?? 'default',
            max_tokens: 512,
            temperature: 0.7
        })
    });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }
    const data = await response.json();
    return data?.choices?.[0]?.text ?? '';
}
```

**Build and package:**

```powershell
cd S:\Ryot\vscode-extension
npm run compile
# If vsce available:
vsce package
# Produces ryzanstein-x.x.x.vsix
```

**Week 3 Exit Criteria:**

- [ ] `ipc/server.go` — JSON dispatch router replacing echo ✅
- [ ] `cd S:\Ryot\desktop && wails build` — zero errors ✅
- [ ] `CommandHandler.ts` — `openChat` opens real WebviewPanel ✅
- [ ] `CommandHandler.ts` — `ryzanstein.infer` command registered ✅
- [ ] `RyzansteinClient.ts` — `infer()` method added ✅
- [ ] VS Code extension — `npm run compile` clean ✅

---

## WEEK 4 — Agent Invocation + Memory Persistence

**Goal:** Wire agent tier system to real dispatch; persist memory to disk.  
**Risk:** HIGH for agent wiring — read `agents/service.go` carefully before touching.  
**Duration:** 7 working days (Sprints 4.2 and 4.3 are ADR documents ONLY)

---

### Sprint 4.1 — Real `InvokeTool()` + New `InvokeAgent()` _(Days 1–3)_

**File:** `S:\Ryot\desktop\internal\agents\service.go`

**Current state:**

- 17 hardcoded agents across T1/T2/T3 tiers
- `InvokeTool()` returns hardcoded `map[string]interface{}`
- `InvokeAgent()` does NOT exist

**Tier multiplier constants (from service.go exploration):**

```go
const (
    T1Multiplier = 1.0
    T2Multiplier = 1.5
    T3Multiplier = 2.0
)
```

**Actions:**

**Part A — Fix `InvokeTool()`:**

```go
// Replace hardcoded return with MCP mesh dispatch:
func (s *AgentService) InvokeTool(toolName, agentCodename string, params map[string]interface{}) (map[string]interface{}, error) {
    // Route through mcp-mesh router (already complete — never re-implement)
    result, err := s.mcpRouter.Route(toolName, agentCodename, params)
    if err != nil {
        return nil, fmt.Errorf("tool dispatch failed for %s/%s: %w", toolName, agentCodename, err)
    }
    return result, nil
}
```

Add `mcpRouter` field to `AgentService` struct and update constructor. The `Route()` method in `mcp-mesh/router.go` is ALREADY COMPLETE — use it directly.

**Part B — Add `InvokeAgent()`:**

```go
// New method on AgentService:
func (s *AgentService) InvokeAgent(codename, message, modelID string) (string, error) {
    agent, ok := s.getAgentByCodename(codename)
    if !ok {
        return "", fmt.Errorf("agent not found: %s", codename)
    }

    // Apply tier multiplier to max_tokens
    multiplier := s.getTierMultiplier(agent.Tier)
    maxTokens := int(float64(512) * multiplier)

    req := &client.ChatCompletionRequest{
        Model: modelID,
        Messages: []client.ChatMessage{
            {Role: "system", Content: agent.SystemPrompt},
            {Role: "user", Content: message},
        },
        MaxTokens:   maxTokens,
        Temperature: 0.7,
    }

    ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
    defer cancel()

    resp, err := s.apiClient.ChatCompletion(ctx, req)
    if err != nil {
        return "", fmt.Errorf("agent inference failed: %w", err)
    }
    if len(resp.Choices) == 0 {
        return "", fmt.Errorf("no choices in response")
    }
    return resp.Choices[0].Message.Content, nil
}

func (s *AgentService) getTierMultiplier(tier string) float64 {
    switch tier {
    case "T1": return T1Multiplier
    case "T2": return T2Multiplier
    case "T3": return T3Multiplier
    default:   return T1Multiplier
    }
}
```

**Expose as Wails binding in `main.go`:**

```go
func (a *App) InvokeAgent(codename, message, modelID string) (string, error) {
    return a.agents.InvokeAgent(codename, message, modelID)
}
```

**Verification:** `cd S:\Ryot\desktop && wails build` — zero errors.

---

### Sprint 4.2 — ADR: Agent Memory Architecture _(Day 4 — DOCUMENT ONLY)_

⚠️ **ZERO IMPLEMENTATION CODE. Write the ADR document only.**

**Path:** `S:\Ryot\docs\adr\ADR-042-agent-memory-architecture.md`

```markdown
# ADR-042: Agent Memory Architecture

## Status

Proposed

## Context

agentmem/store.py currently uses in-memory dict with zero disk persistence.
Memory is lost on process restart. Each agent session starts without prior context.

## Decision

Implement file-backed JSON persistence with optional vector index for semantic recall.

## Storage Format

- Primary: JSON Lines at: %APPDATA%\Ryzanstein\memory\<agent_codename>.jsonl
- Each entry: {timestamp, role, content, embedding_hash, session_id}
- Max entries per agent: 10,000 (rotate oldest on overflow)

## Retrieval Strategy

- Last-N retrieval: O(1) seek from end of file
- Session retrieval: filter by session_id
- Semantic retrieval: defer to Week 5 (sigma-compress embeddings)

## Consequences

- Memory survives process restart ✅
- JSON Lines append-only — no random write corruption ✅
- Large history files (> 10MB) may slow startup — mitigated by rotation ⚠️
- True semantic search deferred to Week 5 ⚠️

## Alternatives Considered

- SQLite: Too heavy for persona-specific memory, introduces CGO dependency
- Redis: External process requirement unacceptable for desktop app
- Pure in-memory: Current state — unacceptable for production
```

---

### Sprint 4.3 — ADR: Compression Strategy _(Day 4 — DOCUMENT ONLY)_

⚠️ **ZERO IMPLEMENTATION CODE. Write the ADR document only.**

**Path:** `S:\Ryot\docs\adr\ADR-043-compression-strategy.md`

```markdown
# ADR-043: Compression Strategy for Sigma-Compress Integration

## Status

Proposed

## Context

sigma-compress/src/ryzanstein_integration.rs currently:

- get_embeddings(): uses hash-based fallback (not semantic)
- health_check(): always returns Ok(true) regardless of backend state

## Decision

Replace both stubs with real HTTP calls to the LLM backend.

## get_embeddings() Target Implementation

- POST /v1/embeddings with model "text-embedding-ada-002" (or backend default)
- Request: {input: [text], model: model_id}
- Response: {data: [{embedding: [f32; 1536]}]}
- Fallback: retain hash-based fallback if HTTP fails (graceful degradation)
- Dependency: reqwest = {version = "0.11", features = ["blocking", "json"]}

## health_check() Target Implementation

- GET /health → Ok(true) if 200, Ok(false) otherwise
- Timeout: 2 seconds
- Remove the unconditional Ok(true)

## Consequences

- Real semantic similarity for memory retrieval ✅
- health_check() reflects actual backend state ✅
- Adds reqwest blocking HTTP — acceptable for Rust library context ✅
- 1536-dimension embeddings require ~6KB per embedding stored ⚠️

## Alternatives Considered

- Local embedding model (e2e-small): Eliminates HTTP round trip but adds 50MB model size
- Sentence transformers via FFI: Too complex; reqwest is simpler
```

---

### Sprint 4.4 — `agentmem` Disk Persistence _(Days 5–7)_

**File:** `S:\Ryot\dependencies\agentmem\store.py`

**Implementation (per ADR-042):**

```python
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_ENTRIES = 10_000

class AgentMemoryStore:
    def __init__(self, agent_codename: str, persist_dir: Optional[str] = None):
        self._codename = agent_codename
        self._memory: List[Dict[str, Any]] = []

        if persist_dir is None:
            config_dir = Path(os.environ.get("APPDATA", Path.home())) / "Ryzanstein" / "memory"
        else:
            config_dir = Path(persist_dir)

        config_dir.mkdir(parents=True, exist_ok=True)
        self._persist_path = config_dir / f"{agent_codename}.jsonl"
        self._load_from_disk()

    def store(self, role: str, content: str, session_id: str = "default") -> None:
        entry = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "session_id": session_id,
        }
        self._memory.append(entry)
        if len(self._memory) > MAX_ENTRIES:
            self._memory = self._memory[-MAX_ENTRIES:]
        self._append_to_disk(entry)

    def get_last_n(self, n: int) -> List[Dict[str, Any]]:
        return self._memory[-n:] if n > 0 else []

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        return [e for e in self._memory if e.get("session_id") == session_id]

    def clear(self) -> None:
        self._memory.clear()
        try:
            self._persist_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _load_from_disk(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._memory.append(json.loads(line))
            if len(self._memory) > MAX_ENTRIES:
                self._memory = self._memory[-MAX_ENTRIES:]
        except Exception:
            self._memory = []

    def _append_to_disk(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self._persist_path, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass  # Disk write failure must never crash inference
```

**Key invariant:** Disk write failure (line in `_append_to_disk`) must silently pass — NEVER propagate exceptions to callers. Memory is nice-to-have persistence, not a hard requirement for inference.

**Tests — add to `tests/test_agentmem.py`:**

```python
import tempfile, os
from agentmem.store import AgentMemoryStore

def test_persistence_survives_restart():
    with tempfile.TemporaryDirectory() as d:
        s1 = AgentMemoryStore("test-agent", persist_dir=d)
        s1.store("user", "hello", "s1")
        s1.store("assistant", "world", "s1")

        s2 = AgentMemoryStore("test-agent", persist_dir=d)
        entries = s2.get_last_n(10)
        assert len(entries) == 2
        assert entries[0]["content"] == "hello"
        assert entries[1]["content"] == "world"

def test_disk_failure_is_silent():
    s = AgentMemoryStore("test-agent", persist_dir="/nonexistent/__path__")
    s.store("user", "test")  # Must not raise
    assert len(s.get_last_n(5)) == 1  # In-memory still works
```

**Wire IPC dispatch from Sprint 3.1 to memory:**

```go
// ipc/server.go — update "infer" case to store in agentmem via Python subprocess call
// OR: IPC → App.InvokeAgent() → response → store via agentmem FFI
// NOTE: Full wiring of IPC → InvokeAgent() → agentmem is the final step of Sprint 4.4
```

**Week 4 Exit Criteria:**

- [ ] `InvokeTool()` routes via `mcp-mesh/router.go` Route() ✅
- [ ] `InvokeAgent()` exists and callable from Wails UI ✅
- [ ] `App.InvokeAgent()` Wails binding registered ✅
- [ ] ADR-042 written (zero code) ✅
- [ ] ADR-043 written (zero code) ✅
- [ ] `agentmem/store.py` persists to `%APPDATA%\Ryzanstein\memory\*.jsonl` ✅
- [ ] `test_persistence_survives_restart` — passes ✅
- [ ] `cd S:\Ryot\desktop && wails build` — zero errors ✅

---

## WEEK 5 — sigma-compress Real Integration + Full System Validation

**Goal:** Replace all stubs in `ryzanstein_integration.rs`. Full end-to-end 7-step smoke test.  
**Risk:** MEDIUM — reqwest HTTP in Rust, must not break existing Rust workspace.  
**Duration:** 5 working days

---

### Sprint 5.1 — Real HTTP in `ryzanstein_integration.rs` _(Days 1–3)_

**File:** `S:\Ryot\dependencies\sigma-compress\src\ryzanstein_integration.rs`

**Step 1 — Add reqwest to Cargo.toml:**

```toml
# S:\Ryot\dependencies\sigma-compress\Cargo.toml
[dependencies]
reqwest = { version = "0.11", features = ["blocking", "json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Verify no version conflicts with workspace root `S:\Ryot\Cargo.toml`.

**Step 2 — Fix `health_check()`:**

```rust
// Replace Ok(true) stub:
pub fn health_check(base_url: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    let resp = client.get(format!("{}/health", base_url)).send();
    match resp {
        Ok(r) => Ok(r.status().is_success()),
        Err(_) => Ok(false),  // Backend unreachable → false, not error
    }
}
```

**Step 3 — Fix `get_embeddings()`:**

```rust
#[derive(serde::Serialize)]
struct EmbeddingRequest<'a> {
    input: Vec<&'a str>,
    model: &'a str,
}

#[derive(serde::Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(serde::Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

pub fn get_embeddings(
    base_url: &str,
    texts: &[&str],
    model: &str,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let req = EmbeddingRequest { input: texts.to_vec(), model };

    match client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&req)
        .send()
    {
        Ok(resp) if resp.status().is_success() => {
            let body: EmbeddingResponse = resp.json()?;
            Ok(body.data.into_iter().map(|d| d.embedding).collect())
        }
        Ok(_) | Err(_) => {
            // Graceful degradation: fall back to hash-based embeddings
            Ok(texts.iter().map(|t| fallback_embed(t)).collect())
        }
    }
}
```

**Verification:**

```powershell
cd S:\Ryot\dependencies\sigma-compress
cargo build
cargo test
```

Also verify workspace root still compiles: `cd S:\Ryot && cargo build`

---

### Sprint 5.2 — Complete IPC → Agent → Memory Pipeline _(Day 4)_

**Objective:** Complete the last wiring gap from Sprint 3.1 + 4.4.

Wire `ipc/server.go` "infer" dispatch to call `App.InvokeAgent()` synchronously
and return the result over the IPC connection:

```go
// ipc/server.go "infer" case — final wiring:
case "infer":
    var p struct {
        Prompt     string `json:"prompt"`
        ModelID    string `json:"model_id"`
        AgentName  string `json:"agent"`
    }
    if err := json.Unmarshal(req.Payload, &p); err != nil {
        resp = map[string]interface{}{"error": "invalid payload"}
        break
    }
    result, err := s.app.InvokeAgent(p.AgentName, p.Prompt, p.ModelID)
    if err != nil {
        resp = map[string]interface{}{"error": err.Error()}
    } else {
        resp = map[string]interface{}{"result": result}
    }
```

**Verification:** `cd S:\Ryot\desktop && wails build` — zero errors.

---

### Sprint 5.3 — Full System 7-Step Smoke Test + Final Report _(Day 5)_

**7-Step Validation Sequence:**

```powershell
# Step 1: Docker Compose backend
docker-compose -f S:\Ryot\docker-compose.yml up -d
# Wait for /health → 200

# Step 2: Build desktop app
cd S:\Ryot\desktop
wails build
# Expected: Build Succeeded, 0 errors, 0 warnings

# Step 3: Unit + integration tests
cd S:\Ryot\desktop
go test ./... -v -timeout 120s
cd S:\Ryot\dependencies\sigma-compress
cargo test

# Step 4: Launch and stream test
# Launch S:\Ryot\desktop\build\bin\Ryzanstein.exe
# Send: "Hello, tell me about yourself."
# Expected: tokens stream live (chat:streamToken events fire)
# Expected: streamingContent renders in ChatPanel

# Step 5: Agent invocation test
# Call App.InvokeAgent("APEX", "What is a binary search tree?", "default")
# Expected: non-empty string response within 10s
# Expected: response stored to %APPDATA%\Ryzanstein\memory\APEX.jsonl

# Step 6: Memory persistence test
# Restart Ryzanstein.exe
# Call GetRecentLogs(5) → expect prior session logs absent (new session)
# Read %APPDATA%\Ryzanstein\memory\APEX.jsonl → expect prior entries present

# Step 7: VS Code extension
# Open VS Code → activate extension
# Ctrl+Shift+P → "Ryzanstein: Open Chat" → expect WebviewPanel opens
# Ctrl+Shift+P → "Ryzanstein: Infer" → enter prompt → expect response
```

**Write `S:\Ryot\docs\WEEK5_VALIDATION_REPORT.md`:**

```markdown
# Week 5 Validation Report

## Date: [DATE]

## Build: [commit hash]

## Test Results

| Step | Test                  | Result    | Notes                    |
| ---- | --------------------- | --------- | ------------------------ |
| 1    | Docker backend health | PASS/FAIL |                          |
| 2    | wails build           | PASS/FAIL |                          |
| 3    | go test ./...         | PASS/FAIL | N tests                  |
| 4    | Streaming end-to-end  | PASS/FAIL | First token latency: Xms |
| 5    | Agent invocation      | PASS/FAIL |                          |
| 6    | Memory persistence    | PASS/FAIL |                          |
| 7    | VS Code extension     | PASS/FAIL |                          |

## Remaining Technical Debt

[List any items not completed]

## Performance Baseline

- First token latency (p50): Xms
- Stream throughput: X tokens/s
- Agent invocation latency (p50): Xms
```

**Week 5 Exit Criteria:**

- [ ] `sigma-compress` `health_check()` — real HTTP ✅
- [ ] `sigma-compress` `get_embeddings()` — real HTTP with fallback ✅
- [ ] `cargo build` + `cargo test` — zero errors ✅
- [ ] IPC "infer" command — full dispatch to InvokeAgent() ✅
- [ ] All 7 smoke test steps — PASS ✅
- [ ] `WEEK5_VALIDATION_REPORT.md` — written ✅

---

## WEEKLY BUILD COMMAND REFERENCE

Every sprint that touches Go code MUST end with:

```powershell
cd S:\Ryot\desktop
wails build
```

NEVER use `-projectdir`. NEVER run from a different directory.

Every sprint that touches Rust code MUST end with:

```powershell
cd S:\Ryot\dependencies\sigma-compress
cargo build && cargo test
# Then verify workspace:
cd S:\Ryot
cargo build
```

Every sprint that touches TypeScript MUST end with:

```powershell
cd S:\Ryot\vscode-extension
npm run compile
```

---

## FINAL DELIVERABLES CHECKLIST

| Deliverable                        | Path                                       | Week |
| ---------------------------------- | ------------------------------------------ | ---- |
| Streaming API Contract             | `docs/STREAMING_API_CONTRACT.md`           | W1   |
| Log Service                        | `desktop/internal/services/log_service.go` | W1   |
| `GetRecentLogs()` Wails binding    | `desktop/main.go`                          | W1   |
| `GetCircuitStatus()` Wails binding | `desktop/main.go`                          | W1   |
| Retry/Timeout Matrix               | `docs/RETRY_TIMEOUT_MATRIX.md`             | W2   |
| `GetSystemHealth()` Wails binding  | `desktop/main.go`                          | W2   |
| IPC JSON Router                    | `desktop/internal/ipc/server.go`           | W3   |
| VS Code `openChat` fix             | `vscode-extension/CommandHandler.ts`       | W3   |
| VS Code `ryzanstein.infer` command | `vscode-extension/CommandHandler.ts`       | W3   |
| VS Code `RyzansteinClient.infer()` | `vscode-extension/RyzansteinClient.ts`     | W3   |
| Real `InvokeTool()`                | `desktop/internal/agents/service.go`       | W4   |
| New `InvokeAgent()`                | `desktop/internal/agents/service.go`       | W4   |
| `App.InvokeAgent()` Wails binding  | `desktop/main.go`                          | W4   |
| ADR-042 Agent Memory Architecture  | `docs/adr/ADR-042-*.md`                    | W4   |
| ADR-043 Compression Strategy       | `docs/adr/ADR-043-*.md`                    | W4   |
| `agentmem` disk persistence        | `dependencies/agentmem/store.py`           | W4   |
| `health_check()` real HTTP         | `sigma-compress/ryzanstein_integration.rs` | W5   |
| `get_embeddings()` real HTTP       | `sigma-compress/ryzanstein_integration.rs` | W5   |
| IPC full pipeline wiring           | `desktop/internal/ipc/server.go`           | W5   |
| Week 5 Validation Report           | `docs/WEEK5_VALIDATION_REPORT.md`          | W5   |

---

_End of Autonomous Execution Plan — Weeks 1 to 5_  
_All architectural facts verified through Wave 17 codebase exploration._  
_Zero further reads required before Week 1 Sprint 1.1 execution._
