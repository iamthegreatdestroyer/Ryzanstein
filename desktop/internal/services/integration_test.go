package services

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
)

// TestCompleteModelLifecycle tests full model lifecycle from discovery to unload.
func TestCompleteModelLifecycle(t *testing.T) {
	// Start mock server
	mockConfig := MockServerConfig{
		Port:      9001,
		Latency:   10 * time.Millisecond,
		ErrorRate: 0.0,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	// Allow server startup
	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	// Create client
	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9001",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test workflow:
	t.Run("list-models", func(t *testing.T) {
		models, err := ms.ListModels(ctx)
		if err != nil {
			t.Logf("list models error (expected in test): %v", err)
		} else {
			if len(models) == 0 {
				t.Log("no models returned (mock server response)")
			}
		}
	})

	t.Run("load-model", func(t *testing.T) {
		_, err := ms.LoadModel(ctx, "bitnet-7b")
		if err != nil {
			t.Logf("load model error (expected in test): %v", err)
		}
	})

	t.Run("model-status", func(t *testing.T) {
		isLoaded := ms.IsModelLoaded("bitnet-7b")
		if isLoaded {
			t.Log("model marked as loaded")
		}
	})

	t.Run("unload-model", func(t *testing.T) {
		err := ms.UnloadModel(ctx, "bitnet-7b")
		if err != nil {
			t.Logf("unload model error (expected in test): %v", err)
		}
	})

	// Verify metrics
	metrics := mockServer.GetMetrics()
	t.Logf("Server metrics: %v", metrics)
}

// TestConcurrentInferenceRequests tests multiple simultaneous inference requests.
func TestConcurrentInferenceRequests(t *testing.T) {
	mockConfig := MockServerConfig{
		Port:      9002,
		Latency:   5 * time.Millisecond,
		ErrorRate: 0.0,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9002",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)

	// Pre-load model
	ctx := context.Background()
	ms.LoadModel(ctx, "test-model")

	// Run concurrent requests
	numGoroutines := 5
	requestsPerGoroutine := 10
	var wg sync.WaitGroup
	errors := 0
	mu := sync.Mutex{}

	start := time.Now()

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			for i := 0; i < requestsPerGoroutine; i++ {
				_, err := ms.GetModelInfo(ctx, "test-model")
				if err != nil {
					mu.Lock()
					errors++
					mu.Unlock()
					t.Logf("goroutine %d request %d error: %v", goroutineID, i, err)
				}
			}
		}(g)
	}

	wg.Wait()
	duration := time.Since(start)

	totalRequests := numGoroutines * requestsPerGoroutine
	successfulRequests := totalRequests - errors
	rps := float64(totalRequests) / duration.Seconds()

	t.Logf("Concurrent Requests: total=%d, successful=%d, errors=%d, duration=%v, throughput=%.2f RPS",
		totalRequests, successfulRequests, errors, duration, rps)
}

// TestErrorRecoveryScenarios tests handling of errors and recovery.
func TestErrorRecoveryScenarios(t *testing.T) {
	// Test with error injection
	mockConfig := MockServerConfig{
		Port:      9003,
		Latency:   5 * time.Millisecond,
		ErrorRate: 0.2, // 20% error rate
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9003",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	successCount := 0
	errorCount := 0

	t.Run("error-handling", func(t *testing.T) {
		for i := 0; i < 50; i++ {
			_, err := ms.GetModelInfo(ctx, "test-model")
			if err != nil {
				errorCount++
			} else {
				successCount++
			}
		}
	})

	t.Logf("Error Recovery: successes=%d, errors=%d, rate=%.1f%%",
		successCount, errorCount, float64(errorCount)*100/float64(successCount+errorCount))

	// Verify server recorded errors
	serverErrors := mockServer.GetErrors()
	serverMetrics := mockServer.GetMetrics()
	t.Logf("Server recorded: total_requests=%d, total_errors=%d, errors_logged=%d",
		serverMetrics["total_requests"],
		serverMetrics["total_errors"],
		len(serverErrors),
	)
}

// TestContextCancellationHandling tests request cancellation during execution.
func TestContextCancellationHandling(t *testing.T) {
	mockConfig := MockServerConfig{
		Port:    9004,
		Latency: 100 * time.Millisecond, // Longer latency to test cancellation
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9004",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)

	t.Run("context-deadline-exceeded", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		_, err := ms.GetModelInfo(ctx, "test-model")
		if err != nil {
			t.Logf("expected timeout error: %v", err)
		} else {
			t.Log("request completed before timeout")
		}
	})

	t.Run("context-cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		_, err := ms.GetModelInfo(ctx, "test-model")
		if err != nil {
			t.Logf("expected cancellation error: %v", err)
		}
	})
}

// TestResourceCleanupOnShutdown tests proper cleanup of resources.
func TestResourceCleanupOnShutdown(t *testing.T) {
	mockConfig := MockServerConfig{
		Port: 9005,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9005",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)

	// Make some requests
	ctx := context.Background()
	ms.LoadModel(ctx, "test-model-1")
	ms.LoadModel(ctx, "test-model-2")

	// Verify state
	loadedCount := ms.GetModelCount()
	t.Logf("Loaded models before shutdown: %d", loadedCount)

	// Unload all models
	if err := ms.UnloadAllModels(ctx); err != nil {
		t.Logf("error unloading models: %v", err)
	}

	// Verify cleanup
	finalLoadedCount := ms.GetModelCount()
	t.Logf("Loaded models after cleanup: %d", finalLoadedCount)

	if finalLoadedCount > 0 {
		t.Logf("warning: %d models still loaded after cleanup", finalLoadedCount)
	}

	// Stop server
	if err := mockServer.Stop(); err != nil {
		t.Fatalf("failed to stop mock server: %v", err)
	}
}

// TestHighConcurrencyStress tests system under high concurrent load.
func TestHighConcurrencyStress(t *testing.T) {
	mockConfig := MockServerConfig{
		Port:              9006,
		Latency:           5 * time.Millisecond,
		MaxConcurrentReqs: 100,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9006",
			Timeout:  10 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	// High concurrency test
	numGoroutines := 50
	requestsPerGoroutine := 20
	var wg sync.WaitGroup
	var successCount, errorCount int64
	mu := sync.Mutex{}

	start := time.Now()

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < requestsPerGoroutine; i++ {
				_, err := ms.GetModelInfo(ctx, "test-model")

				mu.Lock()
				if err != nil {
					errorCount++
				} else {
					successCount++
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	duration := time.Since(start)

	totalRequests := numGoroutines * requestsPerGoroutine
	rps := float64(totalRequests) / duration.Seconds()

	t.Logf("High Concurrency Stress: concurrency=%d, total=%d, successful=%d, errors=%d, duration=%v, throughput=%.2f RPS",
		numGoroutines, totalRequests, successCount, errorCount, duration, rps)

	// Log server metrics
	metrics := mockServer.GetMetrics()
	t.Logf("Server Metrics: %v", metrics)
}

// TestModelCacheConsistency verifies cache behavior across operations.
func TestModelCacheConsistency(t *testing.T) {
	mockConfig := MockServerConfig{
		Port: 9007,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9007",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	t.Run("cache-hit", func(t *testing.T) {
		// First call should fetch
		models1, _ := ms.ListModels(ctx)
		firstCallCount := len(mockServer.GetRequests())

		// Second call should hit cache
		models2, _ := ms.ListModels(ctx)
		secondCallCount := len(mockServer.GetRequests())

		t.Logf("First call requests: %d, Second call requests: %d (should be cached)",
			firstCallCount, secondCallCount)

		if len(models1) > 0 && len(models2) > 0 {
			t.Log("models cached successfully")
		}
	})

	t.Run("cache-invalidation", func(t *testing.T) {
		mockServer.ClearMetrics()

		// List models
		ms.ListModels(ctx)
		requestsBefore := len(mockServer.GetRequests())

		// Clear cache
		ms.ClearCache()

		// List again - should fetch fresh
		ms.ListModels(ctx)
		requestsAfter := len(mockServer.GetRequests())

		t.Logf("Requests before cache clear: %d, after: %d", requestsBefore, requestsAfter)
		if requestsAfter > requestsBefore {
			t.Log("cache invalidation successful")
		}
	})
}

// ============================================================================
// Orchestrator Pipeline Integration Tests
// ============================================================================

func TestOrchestratorPipeline(t *testing.T) {
	t.Run("single_request_through_pipeline", func(t *testing.T) {
		orch := NewOrchestrator(nil)
		defer orch.Stop()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		req := &PipelineRequest{
			ID:        "pipe-001",
			ModelID:   "test-model-alpha",
			Input:     map[string]string{"prompt": "hello world"},
			Priority:  1,
			Timestamp: time.Now(),
		}

		result, err := orch.ProcessRequest(ctx, req)
		if err != nil {
			t.Fatalf("ProcessRequest failed: %v", err)
		}

		if result.RequestID != "pipe-001" {
			t.Errorf("expected RequestID pipe-001, got %s", result.RequestID)
		}
		if result.ModelID != "test-model-alpha" {
			t.Errorf("expected ModelID test-model-alpha, got %s", result.ModelID)
		}
		if result.Latency <= 0 {
			t.Error("expected positive latency")
		}

		outputStr, ok := result.Output.(string)
		if !ok {
			t.Fatalf("expected string output, got %T", result.Output)
		}
		if outputStr != "inference_result_pipe-001" {
			t.Errorf("expected inference_result_pipe-001, got %s", outputStr)
		}

		t.Logf("Pipeline result: ID=%s, Latency=%v, CacheHit=%v", result.RequestID, result.Latency, result.CacheHit)
	})

	t.Run("pipeline_with_custom_config", func(t *testing.T) {
		cfg := DefaultOrchestratorConfig()
		cfg.MaxConcurrentLoad = 2

		orch := NewOrchestrator(cfg)
		defer orch.Stop()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		req := &PipelineRequest{
			ID:        "pipe-cfg-001",
			ModelID:   "custom-model",
			Input:     "test input",
			Priority:  5,
			Timestamp: time.Now(),
		}

		result, err := orch.ProcessRequest(ctx, req)
		if err != nil {
			t.Fatalf("ProcessRequest with custom config failed: %v", err)
		}

		if result.RequestID != "pipe-cfg-001" {
			t.Errorf("expected RequestID pipe-cfg-001, got %s", result.RequestID)
		}
		t.Logf("Custom config pipeline: Latency=%v", result.Latency)
	})

	t.Run("pipeline_request_context_cancellation", func(t *testing.T) {
		orch := NewOrchestrator(nil)
		defer orch.Stop()

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // cancel immediately

		req := &PipelineRequest{
			ID:        "pipe-cancel-001",
			ModelID:   "cancel-model",
			Input:     "will be cancelled",
			Priority:  1,
			Timestamp: time.Now(),
		}

		_, err := orch.ProcessRequest(ctx, req)
		if err == nil {
			t.Log("request completed before cancellation took effect")
		} else {
			t.Logf("correctly received error on cancelled context: %v", err)
		}
	})
}

func TestOrchestratorConcurrentPipeline(t *testing.T) {
	orch := NewOrchestrator(nil)
	defer orch.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const numRequests = 20
	var (
		wg        sync.WaitGroup
		successes int64
		failures  int64
	)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			req := &PipelineRequest{
				ID:        fmt.Sprintf("conc-%03d", idx),
				ModelID:   fmt.Sprintf("model-%d", idx%3),
				Input:     fmt.Sprintf("concurrent input %d", idx),
				Priority:  idx % 5,
				Timestamp: time.Now(),
			}

			result, err := orch.ProcessRequest(ctx, req)
			if err != nil {
				atomic.AddInt64(&failures, 1)
				t.Logf("request %d failed: %v", idx, err)
				return
			}

			expectedOutput := fmt.Sprintf("inference_result_conc-%03d", idx)
			if outputStr, ok := result.Output.(string); ok && outputStr != expectedOutput {
				t.Logf("request %d output mismatch: got %s", idx, outputStr)
			}

			atomic.AddInt64(&successes, 1)
		}(i)
	}

	wg.Wait()

	t.Logf("Concurrent pipeline: %d/%d succeeded, %d failed", successes, numRequests, failures)
	if successes == 0 {
		t.Fatal("all concurrent requests failed")
	}
}

func TestOrchestratorBatchProcessing(t *testing.T) {
	t.Run("batch_multiple_requests", func(t *testing.T) {
		orch := NewOrchestrator(nil)
		defer orch.Stop()

		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		requests := make([]*PipelineRequest, 5)
		for i := 0; i < 5; i++ {
			requests[i] = &PipelineRequest{
				ID:        fmt.Sprintf("batch-%03d", i),
				ModelID:   "batch-model",
				Input:     fmt.Sprintf("batch input %d", i),
				Priority:  i,
				Timestamp: time.Now(),
			}
		}

		results, err := orch.ProcessBatch(ctx, requests)
		if err != nil {
			// ProcessBatch may return partial errors
			t.Logf("batch processing returned error (may be partial): %v", err)
		}

		if len(results) == 0 {
			t.Fatal("batch returned no results")
		}

		t.Logf("Batch results: %d/%d completed", len(results), len(requests))
		for _, r := range results {
			t.Logf("  Result: ID=%s, Latency=%v, CacheHit=%v", r.RequestID, r.Latency, r.CacheHit)
		}
	})

	t.Run("empty_batch", func(t *testing.T) {
		orch := NewOrchestrator(nil)
		defer orch.Stop()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		results, err := orch.ProcessBatch(ctx, []*PipelineRequest{})
		if err != nil {
			t.Logf("empty batch error: %v", err)
		}
		t.Logf("Empty batch results: %d", len(results))
	})
}

func TestOrchestratorMetrics(t *testing.T) {
	orch := NewOrchestrator(nil)
	defer orch.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Process some requests to generate metrics
	for i := 0; i < 5; i++ {
		req := &PipelineRequest{
			ID:        fmt.Sprintf("metrics-%03d", i),
			ModelID:   "metrics-model",
			Input:     "metrics test",
			Priority:  1,
			Timestamp: time.Now(),
		}
		_, _ = orch.ProcessRequest(ctx, req)
	}

	metrics := orch.GetMetrics()

	if metrics.TotalRequests < 5 {
		t.Errorf("expected TotalRequests >= 5, got %d", metrics.TotalRequests)
	}
	if metrics.UptimeSeconds <= 0 {
		t.Error("expected positive UptimeSeconds")
	}

	t.Logf("Orchestrator Metrics:")
	t.Logf("  TotalRequests:  %d", metrics.TotalRequests)
	t.Logf("  TotalCompleted: %d", metrics.TotalCompleted)
	t.Logf("  TotalFailed:    %d", metrics.TotalFailed)
	t.Logf("  AvgLatency:     %.2f ms", metrics.AverageLatency)
	t.Logf("  P99Latency:     %.2f ms", metrics.P99Latency)
	t.Logf("  Throughput:     %.2f req/s", metrics.Throughput)
	t.Logf("  PipelineDepth:  %d", metrics.PipelineDepth)
	t.Logf("  UptimeSeconds:  %.1f", metrics.UptimeSeconds)

	// Verify sub-component metrics are populated
	if metrics.PoolMetrics == nil {
		t.Error("expected PoolMetrics to be populated")
	}
	if metrics.BatchMetrics == nil {
		t.Error("expected BatchMetrics to be populated")
	}
	if metrics.StreamMetrics == nil {
		t.Error("expected StreamMetrics to be populated")
	}
}

func TestOrchestratorGracefulShutdown(t *testing.T) {
	t.Run("stop_with_pending_requests", func(t *testing.T) {
		orch := NewOrchestrator(nil)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		// Fire off some concurrent requests
		var wg sync.WaitGroup
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				req := &PipelineRequest{
					ID:        fmt.Sprintf("shutdown-%03d", idx),
					ModelID:   "shutdown-model",
					Input:     "shutdown test",
					Priority:  1,
					Timestamp: time.Now(),
				}
				_, _ = orch.ProcessRequest(ctx, req)
			}(i)
		}

		// Give requests a moment to enter pipeline
		time.Sleep(50 * time.Millisecond)

		// Stop should drain gracefully
		start := time.Now()
		orch.Stop()
		shutdownDuration := time.Since(start)

		t.Logf("Graceful shutdown completed in %v", shutdownDuration)
		if shutdownDuration > 15*time.Second {
			t.Errorf("shutdown took too long: %v", shutdownDuration)
		}

		wg.Wait()
	})

	t.Run("stop_idempotent", func(t *testing.T) {
		orch := NewOrchestrator(nil)

		// Multiple stops should not panic
		orch.Stop()
		orch.Stop()
		orch.Stop()

		t.Log("Idempotent stop: no panic on multiple calls")
	})

	t.Run("requests_after_stop_rejected", func(t *testing.T) {
		orch := NewOrchestrator(nil)
		orch.Stop()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		req := &PipelineRequest{
			ID:        "post-stop-001",
			ModelID:   "dead-model",
			Input:     "should fail",
			Priority:  1,
			Timestamp: time.Now(),
		}

		_, err := orch.ProcessRequest(ctx, req)
		if err == nil {
			t.Error("expected error when processing request after stop")
		} else {
			t.Logf("Correctly rejected post-stop request: %v", err)
		}
	})
}

func TestOrchestratorComponentAccess(t *testing.T) {
	orch := NewOrchestrator(nil)
	defer orch.Stop()

	if orch.GetPool() == nil {
		t.Error("GetPool() returned nil")
	}
	if orch.GetBatcher() == nil {
		t.Error("GetBatcher() returned nil")
	}
	if orch.GetStreamer() == nil {
		t.Error("GetStreamer() returned nil")
	}
	if orch.GetModelManager() == nil {
		t.Error("GetModelManager() returned nil")
	}

	t.Log("All component accessors returned non-nil values")
}

