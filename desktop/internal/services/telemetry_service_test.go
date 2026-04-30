package services

import (
	"testing"
)

func TestTelemetryService_InitialSnapshot(t *testing.T) {
	svc := NewTelemetryService()
	snap := svc.Snapshot()

	if snap.InferenceRequests != 0 {
		t.Errorf("expected 0 inference requests, got %d", snap.InferenceRequests)
	}
	if snap.Timestamp == "" {
		t.Error("expected non-empty timestamp")
	}
}

func TestTelemetryService_RecordInference(t *testing.T) {
	svc := NewTelemetryService()
	svc.RecordInference(42.5, false)
	svc.RecordInference(10.0, true)

	snap := svc.Snapshot()
	if snap.InferenceRequests != 2 {
		t.Errorf("expected 2, got %d", snap.InferenceRequests)
	}
	if snap.InferenceErrors != 1 {
		t.Errorf("expected 1 error, got %d", snap.InferenceErrors)
	}

	stats, ok := snap.Latencies["inference"]
	if !ok {
		t.Fatal("expected 'inference' in latencies")
	}
	if stats.Count != 2 {
		t.Errorf("expected 2 latency samples, got %d", stats.Count)
	}
	if stats.MeanMs <= 0 {
		t.Errorf("expected positive mean latency, got %f", stats.MeanMs)
	}
}

func TestTelemetryService_RecordChatMessage(t *testing.T) {
	svc := NewTelemetryService()
	for range 5 {
		svc.RecordChatMessage()
	}
	if svc.Snapshot().ChatMessages != 5 {
		t.Errorf("expected 5 chat messages")
	}
}

func TestTelemetryService_RecordAgentInvocation(t *testing.T) {
	svc := NewTelemetryService()
	svc.RecordAgentInvocation(15.0)
	svc.RecordAgentInvocation(20.0)

	snap := svc.Snapshot()
	if snap.AgentInvocations != 2 {
		t.Errorf("expected 2 agent invocations, got %d", snap.AgentInvocations)
	}
	if _, ok := snap.Latencies["agent_invoke"]; !ok {
		t.Error("expected 'agent_invoke' latency histogram")
	}
}

func TestTelemetryService_RecordStreaming(t *testing.T) {
	svc := NewTelemetryService()
	svc.RecordStreaming(100.0)

	snap := svc.Snapshot()
	if snap.StreamingRequests != 1 {
		t.Errorf("expected 1 streaming request, got %d", snap.StreamingRequests)
	}
}

func TestTelemetryService_SetGauge(t *testing.T) {
	svc := NewTelemetryService()
	svc.SetGauge("active_connections", 3.0)
	svc.SetGauge("loaded_model_size_mb", 575.0)

	snap := svc.Snapshot()
	if snap.Gauges["active_connections"] != 3.0 {
		t.Errorf("expected 3.0, got %f", snap.Gauges["active_connections"])
	}
	if snap.Gauges["loaded_model_size_mb"] != 575.0 {
		t.Errorf("expected 575.0, got %f", snap.Gauges["loaded_model_size_mb"])
	}
}

func TestTelemetryService_HistogramMinMaxMean(t *testing.T) {
	svc := NewTelemetryService()
	svc.RecordInference(10.0, false)
	svc.RecordInference(20.0, false)
	svc.RecordInference(30.0, false)

	stats := svc.Snapshot().Latencies["inference"]
	if stats.MinMs > stats.MeanMs || stats.MeanMs > stats.MaxMs {
		t.Errorf("histogram invariant violated: min=%f mean=%f max=%f",
			stats.MinMs, stats.MeanMs, stats.MaxMs)
	}
	if stats.Count != 3 {
		t.Errorf("expected 3 samples, got %d", stats.Count)
	}
}

func TestTelemetryService_Percentiles(t *testing.T) {
	svc := NewTelemetryService()
	// Record 100 evenly-spaced samples: 1ms, 2ms, ..., 100ms.
	for i := 1; i <= 100; i++ {
		svc.RecordInference(float64(i), false)
	}

	stats := svc.Snapshot().Latencies["inference"]

	// P50 should be ~50ms, P95 ~95ms, P99 ~99ms (nearest-rank).
	if stats.P50Ms < 49 || stats.P50Ms > 51 {
		t.Errorf("P50 out of range: %f", stats.P50Ms)
	}
	if stats.P95Ms < 94 || stats.P95Ms > 96 {
		t.Errorf("P95 out of range: %f", stats.P95Ms)
	}
	if stats.P99Ms < 98 || stats.P99Ms > 100 {
		t.Errorf("P99 out of range: %f", stats.P99Ms)
	}
	// Ordering invariant: P50 <= P95 <= P99.
	if stats.P50Ms > stats.P95Ms || stats.P95Ms > stats.P99Ms {
		t.Errorf("percentile ordering violated: P50=%f P95=%f P99=%f",
			stats.P50Ms, stats.P95Ms, stats.P99Ms)
	}
}
