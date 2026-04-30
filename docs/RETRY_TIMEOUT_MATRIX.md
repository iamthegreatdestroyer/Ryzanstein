# Retry & Timeout Matrix

> Comprehensive inventory of all timeout, retry, and backoff values across the Ryzanstein desktop application.
> Generated from Sprint 2.1 code audit. Keep this document updated when values change.

---

## 1. API Client (`desktop/internal/client/ryzanstein_client.go`)

| Parameter            | Default                  | Runtime Override               | Notes                                              |
| -------------------- | ------------------------ | ------------------------------ | -------------------------------------------------- |
| `httpClient.Timeout` | 30 s                     | **150 s** (via `Startup()`)    | HTTP transport-level deadline                      |
| `timeout` field      | 30 s                     | **150 s** (via `SetTimeout()`) | Mirrors httpClient.Timeout                         |
| `maxRetries`         | 3                        | —                              | Total attempts = maxRetries (loop 0..maxRetries-1) |
| `retryDelay`         | 1 s                      | —                              | Base delay for exponential backoff                 |
| Backoff formula      | `retryDelay × 2^attempt` | —                              | 1 s → 2 s → 4 s                                    |

### Endpoints & Methods

| Method                   | HTTP | Path                   | Retry                          | Backoff       |
| ------------------------ | ---- | ---------------------- | ------------------------------ | ------------- |
| `Infer()`                | POST | `/v1/completions`      | ✅ maxRetries with exp backoff | 1 s, 2 s, 4 s |
| `ListModels()`           | GET  | `/v1/models`           | ✅ maxRetries with exp backoff | 1 s, 2 s, 4 s |
| `LoadModel()`            | POST | `/v1/models/load`      | ✅ maxRetries with exp backoff | 1 s, 2 s, 4 s |
| `ChatCompletion()`       | POST | `/v1/chat/completions` | via caller context             | —             |
| `ChatCompletionStream()` | POST | `/v1/chat/completions` | via caller context             | —             |
| `Health()`               | GET  | `/health`              | via caller context             | —             |

---

## 2. Application Layer (`desktop/main.go`)

| Method                | Context Timeout | Notes                                                                                      |
| --------------------- | --------------- | ------------------------------------------------------------------------------------------ |
| `Startup()`           | —               | Sets `apiClient.SetTimeout(150s)`                                                          |
| `SendMessage()`       | 30 s            | `context.WithTimeout(a.ctx, 30*time.Second)` → `ChatCompletion()`                          |
| `SendMessageStream()` | 120 s           | `context.WithTimeout(a.ctx, 120*time.Second)` → `ChatCompletionStream()`, tokenChan cap 64 |
| `CheckAPIHealth()`    | 5 s             | `context.WithTimeout(a.ctx, 5*time.Second)` → `Health()`                                   |

---

## 3. Client Manager (`desktop/internal/services/client_manager.go`)

| Parameter           | Value               | Notes                                                    |
| ------------------- | ------------------- | -------------------------------------------------------- |
| `Inference.Timeout` | 30 s                | Default from config; used for standalone inference calls |
| Infer path          | `/v1/completions`   | POST                                                     |
| ListModels path     | `/v1/models`        | GET                                                      |
| LoadModel path      | `/v1/models/load`   | POST                                                     |
| UnloadModel path    | `/v1/models/unload` | POST                                                     |

---

## 4. Async Model Manager (`desktop/internal/services/async_model_manager.go`)

| Parameter                | Value           | Notes                              |
| ------------------------ | --------------- | ---------------------------------- |
| `LoadTimeout`            | 30 s            | Per-model load deadline            |
| Preload coordinator tick | 100 ms          | Polling interval for preload queue |
| Concurrency              | Semaphore-based | Limits parallel model loads        |

---

## 5. Request Batcher (`desktop/internal/services/batcher.go`)

| Parameter               | Value | Notes                                    |
| ----------------------- | ----- | ---------------------------------------- |
| `BatchTimeout`          | 50 ms | Max wait before flushing a partial batch |
| `MaxBatchSize`          | 200   | Hard cap on batch size                   |
| `MinBatchSize`          | 10    | Minimum before eager flush               |
| Adaptive low threshold  | 5 ms  | Below this → increase batch size         |
| Adaptive high threshold | 10 ms | Above this → decrease batch size         |

---

## 6. Connection Pool (`desktop/internal/services/pool.go`)

| Parameter             | Value  | Notes                                                                                                             |
| --------------------- | ------ | ----------------------------------------------------------------------------------------------------------------- |
| `HealthCheckInterval` | 30 s   | Background health-check ticker                                                                                    |
| `IdleTimeout`         | 5 min  | Close connections idle longer than this                                                                           |
| `MaxConnAge`          | 10 min | Max lifetime of any connection                                                                                    |
| `IdleConnTimeout`     | 90 s   | HTTP transport idle connection timeout                                                                            |
| gRPC dial timeout     | 5 s    | **Uses deprecated `grpc.WithTimeout`** — migrate to `context.WithTimeout`                                         |
| gRPC insecure         | —      | **Uses deprecated `grpc.WithInsecure()`** — migrate to `grpc.WithTransportCredentials(insecure.NewCredentials())` |

---

## 7. Model Service (`desktop/internal/services/model_service.go`)

| Parameter       | Value          | Notes                        |
| --------------- | -------------- | ---------------------------- |
| `cacheExpiry`   | 5 min          | Model list cache TTL         |
| Fallback models | Hardcoded list | Used when API is unreachable |

---

## 8. Timeout Flow Diagram

```
User clicks "Send"
  │
  ├─ Non-streaming: SendMessage()
  │    context timeout = 30 s
  │    └─ apiClient.ChatCompletion()
  │         httpClient.Timeout = 150 s (transport)
  │         Effective deadline = min(30 s ctx, 150 s http) = 30 s
  │
  └─ Streaming: SendMessageStream()
       context timeout = 120 s
       └─ apiClient.ChatCompletionStream()
            httpClient.Timeout = 150 s (transport)
            Effective deadline = min(120 s ctx, 150 s http) = 120 s

Health check: CheckAPIHealth()
  context timeout = 5 s
  └─ apiClient.Health()
       httpClient.Timeout = 150 s (transport)
       Effective deadline = min(5 s ctx, 150 s http) = 5 s
```

---

## 9. Recommended Future Improvements

| Issue                            | Location             | Recommendation                                                 |
| -------------------------------- | -------------------- | -------------------------------------------------------------- |
| Deprecated `grpc.WithInsecure()` | pool.go              | Use `grpc.WithTransportCredentials(insecure.NewCredentials())` |
| Deprecated `grpc.WithTimeout()`  | pool.go              | Use `context.WithTimeout()` + `grpc.DialContext()`             |
| No jitter on retry backoff       | ryzanstein_client.go | Add ±20% random jitter to prevent thundering herd              |
| Hardcoded fallback models        | model_service.go     | Move to config file                                            |
| `SendMessage` 30 s may be tight  | main.go              | Consider increasing to 60 s for large prompts                  |

---

_Last updated: Sprint 2.2_
