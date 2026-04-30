# RYZANSTEIN LLM — NEXT STEPS MASTER ACTION PLAN

## Maximizing Autonomy & Automation | April 2026

**Project:** Ryzanstein LLM v2.0.0  
**Date:** April 11, 2026  
**Current Completion:** ~74%  
**Document Type:** Executable Action Plan — Priority-Ordered  
**Philosophy:** _Smallest delta → biggest user-visible impact, highest automation ratio_

---

## PLAN OVERVIEW

```
╔════════════════════════════════════════════════════════════════╗
║           MASTER ACTION PLAN — APRIL TO JUNE 2026             ║
╠════════════════════════════════════════════════════════════════╣
║  SPRINT A  [NOW]     Desktop ↔ Inference Connection  (P0)     ║
║  SPRINT B  [+1 week] Resilience Layer Completion     (P0)     ║
║  SPRINT C  [+2 weeks] Log Aggregation + VS Code MCP  (P1)     ║
║  SPRINT D  [+3 weeks] Model Optimization (INT4/prune) (P2)    ║
║  SPRINT E  [+4 weeks] Advanced Scheduling / NUMA      (P2)    ║
║  SPRINT F  [+5 weeks] Dependency Ecosystem Phase 1   (P2)     ║
║  SPRINT G  [+7 weeks] Enterprise Features             (P3)    ║
╚════════════════════════════════════════════════════════════════╝
```

**Principle:** Every sprint targets ≥80% automation via scripts, CI/CD, or self-contained code. Human decisions are minimized to architectural choices only.

---

## SPRINT A — DESKTOP ↔ INFERENCE CONNECTION (P0)

### Timeline: Now | Autonomy: 85% | Effort: 3–4 days

**Goal:** Users can open the desktop app and run real inference through the C++ engine with a Svelte UI.

### A.1 — Wire ClientManager to Real HTTP (Day 1)

The `ClientManager` in `desktop/internal/services/client_manager.go` currently holds `interface{}` stubs for its REST and gRPC clients. Replace with real `*http.Client` and `*grpc.ClientConn`.

**Automated:** This is a mechanical code change — the method signatures and config struct are already in place.

```go
// desktop/internal/services/client_manager.go
// Replace interface{} stubs with concrete types:

import (
    "net/http"
    "google.golang.org/grpc"
    pb "github.com/iamthegreatdestroyer/Ryzanstein/mcp/proto"
)

type ClientManager struct {
    config     *config.AppConfig
    restClient *http.Client          // real HTTP client
    grpcConn   *grpc.ClientConn      // real gRPC connection
    mcpClient  pb.RyzansteinClient   // generated proto client
    // ... rest unchanged
}

func (cm *ClientManager) Initialize() error {
    // HTTP client (calls port 8000)
    cm.restClient = &http.Client{Timeout: 30 * time.Second}

    // gRPC client (calls ports 8001-8003)
    conn, err := grpc.Dial(cm.config.MCPEndpoint,
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        return fmt.Errorf("grpc dial: %w", err)
    }
    cm.grpcConn = conn
    cm.mcpClient = pb.NewRyzansteinClient(conn)
    cm.initialized = true
    return nil
}
```

### A.2 — Wire InferenceService to Real API Calls (Day 1)

`InferenceService.Execute()` in `inference_service.go` has correct structs but makes no real HTTP call. Wire it to the REST API at `http://localhost:8000/v1/completions`.

```go
func (is *InferenceService) Execute(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
    // Build OpenAI-compatible request body
    body := map[string]interface{}{
        "model":       req.ModelID,
        "prompt":      req.Prompt,
        "max_tokens":  req.MaxTokens,
        "temperature": req.Temperature,
        "stream":      false,
    }
    // POST to localhost:8000/v1/completions via cm.restClient
    // Unmarshal response into InferenceResponse
    // Update metrics
}
```

### A.3 — Implement Wails IPC Bridge (Day 2)

Create `desktop/internal/ipc/bridge.go` — the Wails-exposed methods that the Svelte frontend calls via `window.go`:

```go
// desktop/internal/ipc/bridge.go
package ipc

type Bridge struct {
    inference *services.InferenceService
    models    *services.ModelService
    agents    *services.AgentService // new
}

// These methods are auto-exported to the Svelte frontend by Wails:

func (b *Bridge) RunInference(prompt string, modelID string) (*services.InferenceResponse, error) {
    return b.inference.Execute(context.Background(), &services.InferenceRequest{
        Prompt:  prompt,
        ModelID: modelID,
        MaxTokens: 512,
    })
}

func (b *Bridge) ListModels() ([]services.ModelInfo, error) {
    return b.models.List(context.Background())
}

func (b *Bridge) GetSystemStatus() (*services.SystemStatus, error) {
    return b.inference.Status()
}
```

### A.4 — Connect Svelte ChatPanel to IPC (Day 2–3)

`ChatPanel.svelte` currently has UI but no live data. Wire via Wails-generated JS bindings:

```svelte
<!-- frontend/src/components/ChatPanel.svelte -->
<script>
  import { RunInference, GetSystemStatus } from '../../wailsjs/go/ipc/Bridge';

  let messages = [];
  let inputText = '';
  let isLoading = false;

  async function sendMessage() {
    isLoading = true;
    const response = await RunInference(inputText, selectedModel);
    messages = [...messages, { role: 'user', text: inputText },
                              { role: 'assistant', text: response.Text }];
    inputText = '';
    isLoading = false;
  }
</script>
```

### A.5 — Streaming Support in Desktop (Day 3–4)

Wire SSE streaming from `streamer.go` through Wails event system to real-time token display in UI:

```go
// Use Wails runtime.EventsEmit for streaming tokens
func (b *Bridge) RunInferenceStream(prompt, modelID string) {
    go func() {
        tokens := b.inference.ExecuteStream(ctx, req)
        for token := range tokens {
            runtime.EventsEmit(ctx, "inference:token", token)
        }
        runtime.EventsEmit(ctx, "inference:done", nil)
    }()
}
```

### A.6 — Build & Test Desktop App (Day 4)

```powershell
# Automated build script: scripts/build_desktop.ps1
Set-Location s:\Ryot\desktop

# Install Go dependencies
go mod tidy

# Build Wails app (compiles Go + bundles Svelte)
wails build -platform windows/amd64 -o ryzanstein-desktop.exe

# Verify build
if (Test-Path "build/bin/ryzanstein-desktop.exe") {
    Write-Host "✅ Desktop build successful"
    & ".\build\bin\ryzanstein-desktop.exe" --test-mode
}
```

**Success Criteria:**

- [ ] Desktop app launches without errors
- [ ] Chat panel sends a prompt and receives a real response from the C++ engine
- [ ] Model selector populates from live `/v1/models` API
- [ ] Token streaming works in real time

---

## SPRINT B — RESILIENCE LAYER COMPLETION (P0)

### Timeline: +1 week | Autonomy: 90% | Effort: 2 days

**Goal:** The serving layer self-heals — circuit breaker uses real fallback, worker watchdog auto-restarts crashed processes.

### B.1 — Wire GracefulDegrader to Mock Engine (Day 1)

`GracefulDegrader` in `resilience.py` has a stub `fallback_engine`. Implement the mock engine fallback:

```python
# src/serving/resilience.py — GracefulDegrader enhancement

class MockFallbackEngine:
    """Emergnecy fallback when C++ engine is unavailable."""

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        return (
            "[Engine temporarily unavailable — degraded mode active. "
            f"Request queued. Prompt: '{prompt[:50]}...']"
        )

class GracefulDegrader:
    def __init__(self, primary_engine, fallback_engine=None):
        self._primary = primary_engine
        self._fallback = fallback_engine or MockFallbackEngine()
        self._degraded = False

    async def execute(self, prompt: str, **kwargs):
        try:
            if self._degraded:
                return self._fallback.generate(prompt)
            return await self._primary.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Primary engine failed: {e}. Activating degraded mode.")
            self._degraded = True
            return self._fallback.generate(prompt)

    async def probe_recovery(self):
        """Periodically test if primary engine has recovered."""
        try:
            _ = await self._primary.health_check()
            self._degraded = False
            logger.info("Primary engine recovered — exiting degraded mode")
        except Exception:
            pass  # Still degraded
```

### B.2 — Connect WorkerWatchdog to Serving Engine (Day 1)

```python
# src/serving/resilience.py — WorkerWatchdog wiring

class WorkerWatchdog:
    def __init__(self, worker_factory: Callable, check_interval: float = 5.0):
        self._worker_factory = worker_factory
        self._worker: Optional[asyncio.Task] = None
        self._check_interval = check_interval
        self._running = False

    async def start(self):
        self._running = True
        self._worker = asyncio.create_task(self._watch_loop())

    async def _watch_loop(self):
        while self._running:
            if self._worker is None or self._worker.done():
                logger.warning("Worker crashed — restarting...")
                self._worker = asyncio.create_task(self._worker_factory())
            await asyncio.sleep(self._check_interval)
```

### B.3 — Integrate Resilience into DistributedServingEngine (Day 2)

```python
# src/serving/distributed_serving.py
# Add at initialization:

from serving.resilience import CircuitBreaker, GracefulDegrader, WorkerWatchdog

class DistributedServingEngine:
    def __init__(self, config):
        self._engine = BitNetEngine(config)
        self._circuit_breaker = CircuitBreaker("bitnet-engine")
        self._degrader = GracefulDegrader(self._engine)
        self._watchdog = WorkerWatchdog(self._restart_worker)

    async def generate(self, prompt: str, **kwargs) -> str:
        async with self._circuit_breaker:
            return await self._degrader.execute(prompt, **kwargs)
```

### B.4 — Configure Log Rotation (Day 2)

```python
# src/serving/lockfree_logger.py — Add rotation handler

import logging.handlers

def setup_rotating_logger(log_dir: str = "/var/log/ryzanstein"):
    handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/ryzanstein.log",
        maxBytes=100 * 1024 * 1024,  # 100MB per file
        backupCount=7,               # 7 days of rotation
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
    ))
    return handler
```

**Success Criteria:**

- [ ] Circuit breaker catches engine failures, trips to OPEN state
- [ ] GracefulDegrader returns useful error response instead of 500
- [ ] WorkerWatchdog restarts crashed workers within 10 seconds
- [ ] Log files rotate at 100MB with 7-day retention

---

## SPRINT C — LOG AGGREGATION + VS CODE MCP (P1)

### Timeline: +2 weeks | Autonomy: 80% | Effort: 3 days

### C.1 — Loki Log Aggregation Pipeline (2 days)

Add Loki to `docker-compose.yml` + configure Promtail to ship logs from all services:

```yaml
# docker-compose.yml — Add Loki + Promtail
loki:
  image: grafana/loki:2.9.0
  ports: ["3100:3100"]

promtail:
  image: grafana/promtail:2.9.0
  volumes:
    - /var/log/ryzanstein:/var/log/ryzanstein
    - ./config/promtail.yml:/etc/promtail/config.yml
  command: -config.file=/etc/promtail/config.yml
```

Add Loki as Grafana data source and create a log explorer dashboard.

### C.2 — VS Code Extension Agent Commands (1 day)

Complete the 30% remaining MCP command palette in the VS Code extension:

```typescript
// vscode-extension/src/agents.ts — New file

import * as vscode from "vscode";
import { MCPClient } from "./mcp_client";

export function registerAgentCommands(context: vscode.ExtensionContext) {
  const mcp = new MCPClient();

  // Command: @APEX code review
  context.subscriptions.push(
    vscode.commands.registerCommand("ryzanstein.invokeApex", async () => {
      const editor = vscode.window.activeTextEditor;
      const code = editor?.document.getText(editor.selection) || "";
      const response = await mcp.invokeAgent("APEX", { task: "review", code });
      vscode.window.showInformationMessage(response.text);
    }),
  );

  // Command: Ask Ryzanstein
  context.subscriptions.push(
    vscode.commands.registerCommand("ryzanstein.askInline", async () => {
      const question = await vscode.window.showInputBox({
        prompt: "Ask Ryzanstein...",
      });
      if (question) {
        const response = await mcp.inference(question);
        // Show in webview panel
        showResponsePanel(context, response.text);
      }
    }),
  );
}
```

---

## SPRINT D — MODEL OPTIMIZATION: INT4 + PRUNING (P2)

### Timeline: +3 weeks | Autonomy: 75% | Effort: 5 days

### D.1 — INT4 GPTQ-Style Quantization (3 days)

Add INT4 quantization pipeline to `RYZEN-LLM/src/optimization/`:

```cpp
// src/optimization/quantization/int4.h — New file
// GPTQ-style per-group INT4 quantization

class INT4Quantizer {
public:
    // Quantize weight matrix W to INT4 symmetric per-group
    QuantizedMatrix quantize(const float* W, int rows, int cols, int group_size = 128);

    // Dequantize for inference
    void dequantize(const QuantizedMatrix& q, float* out, int rows, int cols);

    // Forward pass using quantized weights
    void matmul_int4(const QuantizedMatrix& W, const int8_t* x,
                     float* out, int M, int N, int K);
};
```

Python binding for quantization pipeline:

```python
# RYZEN-LLM/python/ryzanstein_llm/quantize.py
from .ryzen_llm_bindings import INT4Quantizer

def quantize_model(model_path: str, output_path: str, group_size: int = 128):
    """Convert BitNet 1.58b checkpoint to INT4 GPTQ format."""
    quantizer = INT4Quantizer()
    # Load weights, quantize per-layer, save compressed checkpoint
    # Target: 2× memory reduction at <1% accuracy loss
```

### D.2 — Magnitude Pruning with Accuracy Gate (2 days)

```python
# RYZEN-LLM/python/ryzanstein_llm/pruning.py
class MagnitudePruner:
    def __init__(self, accuracy_gate: float = 0.99):
        """Prune weights below threshold; reject if accuracy drops >1%."""
        self.accuracy_gate = accuracy_gate

    def prune(self, model, sparsity: float = 0.3) -> bool:
        """Returns True if pruned within accuracy gate."""
        baseline_ppl = self._measure_perplexity(model)
        pruned = self._apply_magnitude_pruning(model, sparsity)
        pruned_ppl = self._measure_perplexity(pruned)
        accuracy_ratio = baseline_ppl / pruned_ppl
        if accuracy_ratio >= self.accuracy_gate:
            return True  # Accept pruning
        logger.warning(f"Pruning rejected: accuracy_ratio={accuracy_ratio:.3f}")
        return False
```

**Expected gains from Sprint D:**

- Memory: 34 MB → ~20 MB (INT4 compression)
- Speed: additional 20-30% throughput improvement
- Target: **70-80 tok/s**

---

## SPRINT E — ADVANCED SCHEDULING: NUMA + HOT-SWAP (P2)

### Timeline: +4 weeks | Autonomy: 70% | Effort: 4 days

### E.1 — NUMA-Aware CPU Pinning (2 days)

```python
# src/serving/numa_scheduler.py — New file
import psutil

class NUMAScheduler:
    """Pin worker processes to NUMA nodes for memory locality."""

    def __init__(self):
        self.numa_nodes = self._detect_numa_topology()

    def pin_worker(self, pid: int, numa_node: int):
        """Pin process to cores in specified NUMA node."""
        cores = self.numa_nodes[numa_node]
        p = psutil.Process(pid)
        p.cpu_affinity(cores)
        logger.info(f"Pinned worker {pid} to NUMA node {numa_node}: cores {cores}")

    def optimal_node_for_model(self, model_size_gb: float) -> int:
        """Pick NUMA node with most free memory for model loading."""
        return max(range(len(self.numa_nodes)),
                   key=lambda n: psutil.virtual_memory().available)
```

### E.2 — Model Hot-Swap (1 day)

```python
# src/serving/model_manager.py — Enhancement
class ModelManager:
    async def hot_swap(self, new_model_path: str) -> bool:
        """Replace active model with zero downtime."""
        # 1. Load new model in shadow
        new_engine = await self._load_shadow_engine(new_model_path)
        # 2. Drain in-flight requests (graceful period = 5s)
        await asyncio.sleep(5)
        # 3. Atomic pointer swap
        old_engine, self._active_engine = self._active_engine, new_engine
        # 4. Shutdown old engine
        await old_engine.shutdown()
        return True
```

### E.3 — Multi-Model Router + SLA Tiers (1 day)

```python
# src/serving/router.py — New file
class ModelRouter:
    """Route requests to models based on SLA tier and capability."""

    TIERS = {
        "premium":  {"models": ["bitnet-7B"], "max_latency_ms": 100},
        "standard": {"models": ["bitnet-3B"], "max_latency_ms": 500},
        "economy":  {"models": ["bitnet-1.5B"], "max_latency_ms": 2000},
    }

    def select_model(self, request: InferenceRequest) -> str:
        tier = request.metadata.get("sla_tier", "standard")
        return self.TIERS[tier]["models"][0]
```

---

## SPRINT F — DEPENDENCY ECOSYSTEM PHASE 1 (P2)

### Timeline: +5 weeks | Autonomy: 70% | Effort: 2 weeks

Focus exclusively on the **6 highest-value dependencies** that unlock cross-pillar integration.

### F.1 — Priority Dependency Matrix

| Priority | Library          | Why Critical                        | Target |
| -------- | ---------------- | ----------------------------------- | ------ |
| F1       | `sigma-compress` | ΣLANG token compression — Pillar 2  | 80%    |
| F2       | `mcp-mesh`       | Agent routing — Pillar 3 completion | 85%    |
| F3       | `agentmem`       | MNEMONIC memory — agent learning    | 70%    |
| F4       | `ann-hybrid`     | Semantic search for all libraries   | 65%    |
| F5       | `cpu-infer`      | Rust inference bindings alternative | 60%    |
| F6       | `semlog`         | Log analysis feeds observability    | 65%    |

### F.2 — sigma-compress: Complete ΣLANG Integration

The `sigma-compress` Rust library implements the core compression engine. Target:

- Complete Huffman + entropy encoding pipeline
- Add streaming compression API
- Wire to the Python `sigma-compress` PyPI wrapper
- Benchmark: target 50× compression on typical LLM output tokens

### F.3 — mcp-mesh: Complete Gossip Discovery

```go
// mcp-mesh/registry.go — Complete gossip-based discovery
// Currently uses static registry. Add:
// - Gossip protocol for self-healing service discovery
// - Dynamic port assignment
// - Agent capability advertisement
// - Load-based routing (not just round-robin)
```

### F.4 — agentmem: Complete MNEMONIC Cross-Agent Sharing

```python
# agentmem/src/agentmem/consolidation.py — Complete cross-agent tier sharing
# Currently individual agent memory works.
# Need: tier-level memory propagation (all Tier 1 agents share insights)
```

---

## SPRINT G — ENTERPRISE FEATURES (P3)

### Timeline: +7 weeks | Autonomy: 60% | Effort: 2 weeks

### G.1 — Multi-Tenancy

```python
# src/serving/tenancy.py — New file
class TenantIsolator:
    """Namespace-based tenant isolation for multi-org deployment."""
    def create_namespace(self, tenant_id: str) -> TenantNamespace: ...
    def enforce_quota(self, tenant_id: str, tokens_per_minute: int): ...
    def get_usage(self, tenant_id: str) -> UsageReport: ...
```

### G.2 — RBAC

```python
# src/serving/auth/rbac.py
class RBACEnforcer:
    ROLES = {
        "admin":    ["read", "write", "admin"],
        "developer":["read", "write"],
        "viewer":   ["read"],
    }
    def check_permission(self, user_id: str, action: str) -> bool: ...
```

### G.3 — SOC2 Audit Logging

Wire `zkaudit` Rust library through Python bindings to create an immutable, tamper-evident audit trail of all API requests with Merkle proof.

---

## AUTOMATION INFRASTRUCTURE

### CI/CD Enhancements

Add these GitHub Actions workflows for maximum autonomy:

```yaml
# .github/workflows/auto-benchmark.yml
# Triggers: every commit to main
# Actions: Rebuild C++ → Run benchmarks → Comment PR with regression stats

# .github/workflows/auto-build-desktop.yml
# Triggers: every commit touching desktop/
# Actions: wails build → upload artifact

# .github/workflows/dep-ecosystem-ci.yml
# Triggers: changes in dependencies/
# Actions: cargo test (Rust libs) + go test (Go libs) + pytest (Python libs)
```

### Automated Health Dashboard Script

```powershell
# scripts/health_check_all.ps1
# Run daily to check status of all 18 libraries + core engine

$components = @(
    @{ name="C++ Engine"; cmd="python -c 'from ryzanstein_llm.ryzen_llm_bindings import BitNetEngine; print(\"OK\")'"; path="s:\Ryot\RYZEN-LLM\python" }
    @{ name="API Server"; cmd="curl -s http://localhost:8000/health | jq .status"; }
    @{ name="MCP Server"; cmd="curl -s http://localhost:8001/health | jq .status"; }
    @{ name="Desktop App"; cmd="Test-Path 's:\Ryot\desktop\build\bin\ryzanstein-desktop.exe'"; }
)
foreach ($c in $components) {
    # Test and report status
}
```

---

## EXECUTION PRIORITIES SUMMARY

### This Week (P0 — Non-Negotiable)

```
1. [ ] Wire ClientManager REST calls → port 8000 (2 hours)
2. [ ] Wire InferenceService HTTP body (2 hours)
3. [ ] Create IPC Bridge in Wails (3 hours)
4. [ ] Connect ChatPanel.svelte to IPC (3 hours)
5. [ ] Wire GracefulDegrader fallback (2 hours)
6. [ ] Connect WorkerWatchdog to serving engine (2 hours)
```

### Next Week (P1)

```
7. [ ] Log rotation configuration (~1 hour)
8. [ ] Add Loki to docker-compose + Promtail config (3 hours)
9. [ ] VS Code agent command palette (4 hours)
10.[ ] Desktop streaming via Wails events (3 hours)
```

### Weeks 3-4 (P2 — Model Performance)

```
11.[ ] INT4 quantization C++ + Python binding (3 days)
12.[ ] Magnitude pruning with accuracy gate (2 days)
13.[ ] NUMA-aware scheduler (2 days)
14.[ ] Model hot-swap (1 day)
```

### Weeks 5-7 (P2 — Ecosystem)

```
15.[ ] sigma-compress streaming API (3 days)
16.[ ] mcp-mesh gossip protocol (3 days)
17.[ ] agentmem cross-tier memory sharing (2 days)
18.[ ] ann-hybrid persistent index storage (2 days)
```

---

## SUCCESS METRICS

| Milestone              | Metric                             | Target                | Sprint |
| ---------------------- | ---------------------------------- | --------------------- | ------ |
| Desktop inference live | Users can chat with LLM from app   | ✅ functional         | A      |
| Resilience complete    | Engine crash → auto-recovery < 10s | Zero 500s             | B      |
| Log aggregation        | All services in Grafana/Loki       | Full visibility       | C      |
| INT4 model             | Memory reduction + speed gain      | 20 MB / 70+ tok/s     | D      |
| NUMA scheduler         | CPU utilization vs baseline        | +15% throughput       | E      |
| sigma-compress         | Integration test with core engine  | 50× token compression | F      |
| Enterprise ready       | Multi-tenant API with RBAC         | SOC2 audit trail      | G      |

---

## ARCHITECTURAL EVOLUTION ROADMAP

```
APRIL 2026:  Desktop connection (P0) → Resilience (P0) → Logging (P1)
             RESULT: "Complete working product — all user paths functional"

MAY 2026:    INT4/Pruning (P2) → NUMA (P2) → sigma-compress integration
             RESULT: "Performance tier achieved — 70+ tok/s, 3MB model size"

JUNE 2026:   Ecosystem P1 (mcp-mesh, agentmem) → Enterprise (P3)
             RESULT: "Enterprise-grade deployment with MCP agent orchestration"

Q3 2026:     Ecosystem completion → Open source release → Marketplace Extension
             RESULT: "Public platform — all 18 libraries production-ready"
```

---

_Document generated: April 11, 2026_  
_Next review: April 18, 2026 (after Sprint A completion)_  
_Maintained by: GitHub Copilot ARCHITECT-03_
