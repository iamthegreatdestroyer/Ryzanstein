# Ryzanstein Streaming API Contract

> **Sprint 1.1 — API Contract Documentation**
> Generated from verified source code: `desktop/internal/client/ryzanstein_client.go` (494 lines) and `desktop/main.go`.

---

## Table of Contents

1. [Endpoint Reference](#1-endpoint-reference)
2. [Type Definitions](#2-type-definitions)
3. [Chat Request Construction Pattern](#3-chat-request-construction-pattern)
4. [SSE Wire Format](#4-sse-wire-format)
5. [The `[DONE]` Sentinel Behaviour](#5-the-done-sentinel-behaviour)
6. [The `splitLines()` Contract](#6-the-splitlines-contract)
7. [The `tokenChan` Buffering & Dual-Goroutine Pattern](#7-the-tokenchan-buffering--dual-goroutine-pattern)
8. [120-Second Context Timeout](#8-120-second-context-timeout)
9. [Error Handling & Retry Behaviour](#9-error-handling--retry-behaviour)
10. [Wails Event Sequence](#10-wails-event-sequence)
11. [Fallback Behaviours](#11-fallback-behaviours)
12. [Client Constructor Defaults](#12-client-constructor-defaults)

---

## 1. Endpoint Reference

All endpoints are relative to the configured `RyzansteinAPIURL` (default: `http://localhost:8000`).

| Method | HTTP Verb | Endpoint | Request Body | Response | Retry? |
|---|---|---|---|---|---|
| `Infer()` | POST | `/v1/completions` | `InferenceRequest` JSON | `InferenceResponse` | YES (3× exponential backoff) |
| `ListModels()` | GET | `/v1/models` | none | `{"data": []ModelInfo}` | YES (3× exponential backoff) |
| `LoadModel()` | POST | `/v1/models/load` | `{"model_id": "<id>"}` | 200 OK (no body) | YES (3× exponential backoff) |
| `UnloadModel()` | POST | `/v1/models/<id>/unload` | none | 200 OK (no body) | YES (3× exponential backoff) |
| `ChatCompletion()` | POST | `/v1/chat/completions` | `ChatCompletionRequest` JSON | `ChatCompletionResponse` | NO |
| `ChatCompletionStream()` | POST | `/v1/chat/completions` | `ChatCompletionRequest` (Stream forced `true`) | SSE stream → `tokenChan` | NO |
| `Health()` | GET | `/health` | none | `bool` (200 = healthy) | NO |

### Headers

| Header | Value | Applied To |
|---|---|---|
| `Content-Type` | `application/json` | All POST requests |
| `Accept` | `text/event-stream` | `ChatCompletionStream()` only |

---

## 2. Type Definitions

All types are defined in `desktop/internal/client/ryzanstein_client.go`. JSON tags are exact.

### InferenceRequest

```go
type InferenceRequest struct {
    Prompt      string  `json:"prompt"`
    Model       string  `json:"model"`
    Temperature float32 `json:"temperature,omitempty"`
    MaxTokens   int     `json:"max_tokens,omitempty"`
    TopP        float32 `json:"top_p,omitempty"`
    Stream      bool    `json:"stream,omitempty"`
}
```

### InferenceResponse

```go
type InferenceResponse struct {
    ID      string `json:"id"`
    Model   string `json:"model"`
    Choices []struct {
        Text         string `json:"text"`
        FinishReason string `json:"finish_reason"`
    } `json:"choices"`
    Usage struct {
        PromptTokens     int `json:"prompt_tokens"`
        CompletionTokens int `json:"completion_tokens"`
        TotalTokens      int `json:"total_tokens"`
    } `json:"usage"`
}
```

### ModelInfo

```go
type ModelInfo struct {
    ID              string `json:"id"`
    Name            string `json:"name"`
    ContextWindow   int    `json:"context_window"`
    MaxOutputTokens int    `json:"max_output_tokens"`
    Type            string `json:"type"`
    Quantization    string `json:"quantization"`
}
```

### ChatMessage

```go
type ChatMessage struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}
```

### ChatCompletionRequest

```go
type ChatCompletionRequest struct {
    Model       string        `json:"model"`
    Messages    []ChatMessage `json:"messages"`
    MaxTokens   int           `json:"max_tokens,omitempty"`
    Temperature float32       `json:"temperature,omitempty"`
    TopP        float32       `json:"top_p,omitempty"`
    Stream      bool          `json:"stream,omitempty"`
}
```

### ChatCompletionResponse

```go
type ChatCompletionResponse struct {
    ID      string `json:"id"`
    Object  string `json:"object"`
    Created int64  `json:"created"`
    Model   string `json:"model"`
    Choices []struct {
        Index        int         `json:"index"`
        Message      ChatMessage `json:"message"`
        FinishReason string      `json:"finish_reason"`
    } `json:"choices"`
    Usage struct {
        PromptTokens     int `json:"prompt_tokens"`
        CompletionTokens int `json:"completion_tokens"`
        TotalTokens      int `json:"total_tokens"`
    } `json:"usage"`
}
```

### RyzansteinError

```go
type RyzansteinError struct {
    Code    int    `json:"code"`
    Message string `json:"message"`
    Details string `json:"details"`
}
```

### SSE Stream Chunk (anonymous struct used in `ChatCompletionStream`)

```go
struct {
    Choices []struct {
        Delta struct {
            Content string `json:"content"`
        } `json:"delta"`
    } `json:"choices"`
}
```

---

## 3. Chat Request Construction Pattern

Both `SendMessage()` (non-streaming) and `SendMessageStream()` (streaming) in `main.go` construct identical `ChatCompletionRequest` payloads:

```go
chatReq := &client.ChatCompletionRequest{
    Model: modelID,
    Messages: []client.ChatMessage{
        {Role: "system", Content: systemPrompt},
        {Role: "user", Content: userMessage},
    },
    MaxTokens:   2048,
    Temperature: 0.7,
    TopP:        0.9,
}
```

### System Prompt

```go
systemPrompt := fmt.Sprintf(
    "You are %s, an elite AI agent. Respond helpfully and concisely.",
    agentCodename,
)
```

### Default Parameter Values

| Parameter | Value | Notes |
|---|---|---|
| `MaxTokens` | `2048` | Hardcoded in both paths |
| `Temperature` | `0.7` | Hardcoded in both paths |
| `TopP` | `0.9` | Hardcoded in both paths |
| `Stream` | Not set by caller | Forced to `true` inside `ChatCompletionStream()` before marshalling |

### Message Array Structure

The messages array always contains exactly 2 messages:
1. **System message** — agent persona prompt with codename
2. **User message** — the user's input text

---

## 4. SSE Wire Format

`ChatCompletionStream()` sends `POST /v1/chat/completions` with:
- `req.Stream = true` (forced before JSON marshal)
- Header: `Accept: text/event-stream`

### Response Format

Each SSE event is a line prefixed with `data: ` followed by a JSON payload:

```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

### Key Characteristics

- **Line delimiter:** `\n` (with possible `\r\n`)
- **Blank lines** between events are skipped by `splitLines()`
- **Prefix:** `data: ` (6 bytes, including trailing space)
- **Payload:** JSON object with `choices[].delta.content` containing the token
- **Sentinel:** `[DONE]` (literal string, not JSON)

### Read Buffer

```go
buf := make([]byte, 4096)
```

The response body is read in 4096-byte chunks. Each chunk is passed through `splitLines()` to extract individual SSE lines.

---

## 5. The `[DONE]` Sentinel Behaviour

After the final token, the server sends:

```
data: [DONE]
```

### Processing Logic

```go
data := line[6:]  // strip "data: " prefix
if data == "[DONE]" {
    return nil  // clean stream termination
}
```

- The `[DONE]` sentinel causes `ChatCompletionStream()` to return `nil` (success)
- The deferred `close(tokenChan)` fires, unblocking the `range tokenChan` loop in the consumer goroutine
- The consumer then assembles the full response and persists it to chat history

### Alternate Termination

If the server closes the connection without sending `[DONE]`:

```go
if err.Error() == "EOF" {
    return nil  // treat EOF as clean termination
}
```

Any other read error returns `fmt.Errorf("stream read error: %w", err)`.

---

## 6. The `splitLines()` Contract

`splitLines()` is a helper function in `ryzanstein_client.go` that parses raw byte chunks from the SSE stream into discrete lines.

### Algorithm

```
Input:  chunk string (raw bytes from HTTP body read)
Output: []string (non-empty lines)

1. Iterate byte-by-byte through chunk
2. When '\n' found:
   a. Extract line from start to current position
   b. Strip trailing '\r' if present (handles \r\n line endings)
   c. If line is non-empty, append to result
   d. Update start to position after '\n'
3. After loop: handle any trailing content after the last '\n'
4. Return slice of non-empty lines
```

### Guarantees

| Property | Behaviour |
|---|---|
| `\n` line endings | ✅ Correctly split |
| `\r\n` line endings | ✅ `\r` stripped before append |
| Empty lines | ✅ Skipped (only non-empty lines returned) |
| Trailing content (no final `\n`) | ✅ Captured as final line |
| Zero-length input | ✅ Returns empty slice |

### Usage in SSE Processing

```go
lines := splitLines(string(buf[:n]))
for _, line := range lines {
    if len(line) > 6 && line[:6] == "data: " {
        data := line[6:]
        // process data...
    }
}
```

Lines that don't start with `data: ` (e.g., SSE comments, event type lines) are silently ignored.

---

## 7. The `tokenChan` Buffering & Dual-Goroutine Pattern

### Channel Creation

```go
tokenChan := make(chan string, 64)
```

- **Buffer size:** 64 tokens
- **Created in:** outer goroutine of `SendMessageStream()` (`main.go:207`)
- **Type direction:** `chan<- string` (write-only) when passed to `ChatCompletionStream()`

### Dual-Goroutine Architecture

```
SendMessageStream()
│
├─ Outer Goroutine (consumer)
│   ├─ Creates tokenChan with buffer 64
│   ├─ Creates fullResponse strings.Builder
│   ├─ Spawns inner goroutine (producer)
│   ├─ Consumes: for token := range tokenChan
│   │   ├─ fullResponse.WriteString(token)
│   │   └─ runtime.EventsEmit("chat:streamToken", token)
│   ├─ Assembles final responseText
│   ├─ Applies offline fallback if empty
│   └─ Persists: a.chat.AddMessage(...)
│
└─ Inner Goroutine (producer)
    ├─ defer close(tokenChan)  ← ensures consumer loop terminates
    ├─ Calls a.apiClient.ChatCompletionStream(ctx, chatReq, tokenChan)
    └─ On error: emits chat:streamError event
```

### tokenChan Lifecycle (6 usage sites)

| Location | Line | Operation |
|---|---|---|
| `main.go` | 207 | **Create:** `make(chan string, 64)` |
| `main.go` | 211 | **Close:** `defer close(tokenChan)` |
| `main.go` | 212 | **Pass:** `ChatCompletionStream(ctx, chatReq, tokenChan)` |
| `main.go` | 219 | **Consume:** `for token := range tokenChan` |
| `ryzanstein_client.go` | 378 | **Signature:** `tokenChan chan<- string` (write-only) |
| `ryzanstein_client.go` | 432 | **Deliver:** `tokenChan <- choice.Delta.Content` |

### Data Flow

```
HTTP Body → Read(buf[4096]) → splitLines() → "data: " check → JSON unmarshal
    → choice.Delta.Content → tokenChan ← (buffer 64) → range loop
    → strings.Builder + EventsEmit("chat:streamToken") → UI render
```

---

## 8. 120-Second Context Timeout

### Declaration

```go
ctx, cancel := context.WithTimeout(a.ctx, 120*time.Second)
```

- **Location:** `SendMessageStream()` in `main.go`
- **Duration:** 120 seconds (2 minutes)
- **Cleanup:** `defer cancel()` in outer goroutine

### Timeout vs. HTTP Client Timeout

| Timeout | Value | Scope |
|---|---|---|
| `context.WithTimeout` | 120s | Per-stream operation in `SendMessageStream()` |
| `httpClient.Timeout` | 30s | Default HTTP client timeout on `RyzansteinClient` |

> **⚠️ Known Concern:** The 30-second `httpClient.Timeout` may terminate long-running streams before the 120-second context timeout expires. This is tracked for resolution in Week 2, Sprint 2.2.

### Context Propagation

```
App.ctx → context.WithTimeout(120s) → ChatCompletionStream(ctx, ...) → http.NewRequestWithContext(ctx, ...)
```

The context flows from the Wails application context through the timeout wrapper into the HTTP request, enabling cancellation at any point in the chain.

---

## 9. Error Handling & Retry Behaviour

### Retry-Enabled Endpoints

| Method | Retries | Backoff Formula |
|---|---|---|
| `Infer()` | 3 | `retryDelay × 2^attempt` (1s → 2s → 4s) |
| `ListModels()` | 3 | `retryDelay × 2^attempt` |
| `LoadModel()` | 3 | `retryDelay × 2^attempt` |
| `UnloadModel()` | 3 | `retryDelay × 2^attempt` |

### Non-Retry Endpoints

| Method | Reason |
|---|---|
| `ChatCompletion()` | Latency-sensitive, user-facing |
| `ChatCompletionStream()` | Long-running SSE connection |
| `Health()` | Simple connectivity check |

### HTTP Error Detection (Streaming)

```go
if httpResp.StatusCode >= 400 {
    // Decode response body as RyzansteinError
    // Return formatted error with code + message + details
}
```

### Stream Error Handling

```go
// In inner goroutine (producer):
err := a.apiClient.ChatCompletionStream(ctx, chatReq, tokenChan)
if err != nil {
    log.Printf("[Chat] Stream error: %v\n", err)
    runtime.EventsEmit(a.ctx, "chat:streamError", err.Error())
}
// defer close(tokenChan) fires regardless, unblocking consumer
```

---

## 10. Wails Event Sequence

### Streaming Flow (`SendMessageStream`)

```
1. chat:message        → {id, role:"user", content, timestamp}     (user message posted)
2. chat:streamStart    → nil                                        (stream begins)
3. chat:streamToken    → "<token>"                                  (repeated per token)
4. chat:streamError    → "<error message>"                          (only on error)
5. chat:response       → "<full assembled response>"               (stream complete)
```

### Other Application Events

| Event | Payload | When |
|---|---|---|
| `app:ready` | `{version: "1.0.0", timestamp: <now>}` | Application startup |

### Event Direction

All events flow **Go → Svelte** via `runtime.EventsEmit()`. The Svelte frontend subscribes using Wails' event system.

---

## 11. Fallback Behaviours

### Non-Streaming (`SendMessage`)

| Condition | Response |
|---|---|
| API call fails | `"[Offline Mode] The Ryzanstein inference API at <URL> is not reachable. Start backend with: docker-compose up -d"` |
| Success with choices | `chatResp.Choices[0].Message.Content` |
| Empty choices array | `"[Error] Empty response from inference API."` |

### Streaming (`SendMessageStream`)

| Condition | Response |
|---|---|
| Stream error | `chat:streamError` event emitted with error string; `tokenChan` closed |
| Empty final response | `"[Offline Mode] Streaming not available. Start backend with: docker-compose up -d"` |

In all fallback cases, the response is persisted to chat history via `a.chat.AddMessage()`.

---

## 12. Client Constructor Defaults

```go
RyzansteinClient{
    baseURL:    <provided>,
    httpClient: &http.Client{Timeout: 30 * time.Second},
    timeout:    30 * time.Second,
    maxRetries: 3,
    retryDelay: time.Second,  // 1s base, exponential: 1s → 2s → 4s
}
```

---

## Full Streaming Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        STREAMING PIPELINE                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Svelte UI                                                           │
│    │                                                                 │
│    ▼                                                                 │
│  SendMessageStream(userMessage, modelID, agentCodename)              │
│    │                                                                 │
│    ├─ ctx = WithTimeout(app.ctx, 120s)                               │
│    ├─ Emit: chat:message (user msg)                                  │
│    ├─ Emit: chat:streamStart                                         │
│    ├─ Build chatReq {Model, Messages[system+user],                   │
│    │                  MaxTokens:2048, Temp:0.7, TopP:0.9}            │
│    │                                                                 │
│    └─ go func() {  // outer goroutine                                │
│         tokenChan := make(chan string, 64)                            │
│         │                                                            │
│         ├─ go func() {  // inner goroutine                           │
│         │     defer close(tokenChan)                                 │
│         │     │                                                      │
│         │     ▼                                                      │
│         │   ChatCompletionStream(ctx, chatReq, tokenChan)            │
│         │     │                                                      │
│         │     ├─ req.Stream = true (forced)                          │
│         │     ├─ POST /v1/chat/completions                           │
│         │     │   Content-Type: application/json                     │
│         │     │   Accept: text/event-stream                          │
│         │     │                                                      │
│         │     └─ Read Loop:                                          │
│         │         buf[4096] → splitLines(chunk)                      │
│         │         for line in lines:                                 │
│         │           if line[:6] == "data: ":                         │
│         │             data = line[6:]                                │
│         │             if data == "[DONE]" → return nil               │
│         │             unmarshal → tokenChan <- delta.content         │
│         │         on EOF → return nil                                │
│         │         on error → return stream read error                │
│         │   }()                                                      │
│         │                                                            │
│         └─ for token := range tokenChan {                            │
│               fullResponse.WriteString(token)                        │
│               Emit: chat:streamToken(token)  ──────► Svelte UI       │
│           }                                                          │
│           │                                                          │
│           ├─ if empty → "[Offline Mode]..."                          │
│           └─ a.chat.AddMessage("assistant", response, model, agent)  │
│       }()                                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

*Document generated as Sprint 1.1, Action #1 of the Autonomous Execution Plan (Weeks 1–5).*
*Source files: `desktop/internal/client/ryzanstein_client.go` (494 lines), `desktop/main.go` (lines 125–230).*
