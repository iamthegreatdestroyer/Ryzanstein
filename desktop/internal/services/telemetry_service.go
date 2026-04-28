package services

import (
	"sync"
	"sync/atomic"
	"time"
)

// TelemetryService tracks inference, chat, and agent operation metrics
// in the Desktop runtime. Mirrors the sigma-telemetry span/metrics model
// so snapshots can be correlated with the Rust-layer telemetry pipeline.
type TelemetryService struct {
	mu sync.RWMutex

	// Atomic counters for hot-path operations.
	inferenceRequests  atomic.Int64
	inferenceErrors    atomic.Int64
	chatMessages       atomic.Int64
	agentInvocations   atomic.Int64
	streamingRequests  atomic.Int64

	// Latency histograms (nanoseconds) per operation type.
	histograms map[string]*latencyHistogram

	// Named gauges (e.g., "active_connections", "loaded_model_size_mb").
	gauges map[string]float64
}

type latencyHistogram struct {
	mu      sync.Mutex
	samples []int64 // nanoseconds
}

func (h *latencyHistogram) record(ns int64) {
	h.mu.Lock()
	h.samples = append(h.samples, ns)
	h.mu.Unlock()
}

func (h *latencyHistogram) stats() HistogramStats {
	h.mu.Lock()
	defer h.mu.Unlock()

	n := len(h.samples)
	if n == 0 {
		return HistogramStats{}
	}

	var sum int64
	min, max := h.samples[0], h.samples[0]
	for _, v := range h.samples {
		sum += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	return HistogramStats{
		Count:   n,
		MeanMs:  float64(sum) / float64(n) / 1e6,
		MinMs:   float64(min) / 1e6,
		MaxMs:   float64(max) / 1e6,
		Samples: n,
	}
}

// HistogramStats summarises a latency histogram.
type HistogramStats struct {
	Count   int     `json:"count"`
	MeanMs  float64 `json:"mean_ms"`
	MinMs   float64 `json:"min_ms"`
	MaxMs   float64 `json:"max_ms"`
	Samples int     `json:"samples"`
}

// TelemetrySnapshot is the full point-in-time state returned to the UI.
type TelemetrySnapshot struct {
	Timestamp          string                    `json:"timestamp"`
	InferenceRequests  int64                     `json:"inference_requests"`
	InferenceErrors    int64                     `json:"inference_errors"`
	ChatMessages       int64                     `json:"chat_messages"`
	AgentInvocations   int64                     `json:"agent_invocations"`
	StreamingRequests  int64                     `json:"streaming_requests"`
	Latencies          map[string]HistogramStats `json:"latencies"`
	Gauges             map[string]float64        `json:"gauges"`
}

// NewTelemetryService creates a ready-to-use TelemetryService.
func NewTelemetryService() *TelemetryService {
	return &TelemetryService{
		histograms: make(map[string]*latencyHistogram),
		gauges:     make(map[string]float64),
	}
}

// RecordInference records a completed inference call.
// durationMs is the wall-clock time in milliseconds.
func (t *TelemetryService) RecordInference(durationMs float64, errored bool) {
	t.inferenceRequests.Add(1)
	if errored {
		t.inferenceErrors.Add(1)
	}
	t.recordLatency("inference", int64(durationMs*1e6))
}

// RecordChatMessage records a user or assistant message event.
func (t *TelemetryService) RecordChatMessage() {
	t.chatMessages.Add(1)
}

// RecordAgentInvocation records an agent tool invocation.
func (t *TelemetryService) RecordAgentInvocation(durationMs float64) {
	t.agentInvocations.Add(1)
	t.recordLatency("agent_invoke", int64(durationMs*1e6))
}

// RecordStreaming records a streaming request completion.
func (t *TelemetryService) RecordStreaming(durationMs float64) {
	t.streamingRequests.Add(1)
	t.recordLatency("streaming", int64(durationMs*1e6))
}

// SetGauge sets a named gauge value (e.g., "active_connections").
func (t *TelemetryService) SetGauge(name string, value float64) {
	t.mu.Lock()
	t.gauges[name] = value
	t.mu.Unlock()
}

// Snapshot returns a point-in-time copy of all telemetry state.
func (t *TelemetryService) Snapshot() TelemetrySnapshot {
	t.mu.RLock()
	gauges := make(map[string]float64, len(t.gauges))
	for k, v := range t.gauges {
		gauges[k] = v
	}
	latencies := make(map[string]HistogramStats, len(t.histograms))
	for k, h := range t.histograms {
		latencies[k] = h.stats()
	}
	t.mu.RUnlock()

	return TelemetrySnapshot{
		Timestamp:         time.Now().UTC().Format(time.RFC3339),
		InferenceRequests: t.inferenceRequests.Load(),
		InferenceErrors:   t.inferenceErrors.Load(),
		ChatMessages:      t.chatMessages.Load(),
		AgentInvocations:  t.agentInvocations.Load(),
		StreamingRequests: t.streamingRequests.Load(),
		Latencies:         latencies,
		Gauges:            gauges,
	}
}

func (t *TelemetryService) recordLatency(operation string, ns int64) {
	t.mu.RLock()
	h := t.histograms[operation]
	t.mu.RUnlock()

	if h == nil {
		t.mu.Lock()
		if t.histograms[operation] == nil {
			t.histograms[operation] = &latencyHistogram{}
		}
		h = t.histograms[operation]
		t.mu.Unlock()
	}

	h.record(ns)
}
