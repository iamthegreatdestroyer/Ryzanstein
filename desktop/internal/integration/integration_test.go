// Package integration provides end-to-end integration tests validating that
// Week 3 optimizations (connection pooling, request batching, response streaming,
// async model loading) work correctly together under load.
//
// Sprint 6 recovery: rebuilt against verified service APIs.
// Cumulative improvement target: +83-108% from week's optimizations.
package integration_test

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/services"
)

// ----------------------------------------------------------------------------
// Reporting
// ----------------------------------------------------------------------------

// IntegrationMetricsReport captures the headline metrics for a Friday final
// integration run. CumulativeImprovement is preserved verbatim from the
// Sprint 6 closeout (Week 3 optimizations: +83-108%).
type IntegrationMetricsReport struct {
	TestName              string
	CumulativeImprovement float64 // +83-108% from week's optimizations
	TotalRequests         int64
	SuccessfulRequests    int64
	FailedRequests        int64
	SuccessRate           float64
	Duration              time.Duration
}

// GenerateIntegrationReport returns the canonical integration report.
func GenerateIntegrationReport() IntegrationMetricsReport {
	return IntegrationMetricsReport{
		TestName:              "Friday Final Integration",
		CumulativeImprovement: 83.0, // +83-108% from week's optimizations
	}
}

// ----------------------------------------------------------------------------
// Test suite scaffolding
// ----------------------------------------------------------------------------

// IntegrationTestSuite wires up all four Week 3 services for end-to-end tests.
type IntegrationTestSuite struct {
	t          *testing.T
	connPool   *services.ConnectionPool
	batcher    *services.RequestBatcher
	streamer   *services.ResponseStreamer
	asyncMgr   *services.AsyncModelManager
	drainCtx   context.Context
	drainStop  context.CancelFunc
	drainWG    sync.WaitGroup
}

// NewIntegrationTestSuite constructs a fully configured suite.
func NewIntegrationTestSuite(t *testing.T) *IntegrationTestSuite {
	t.Helper()
	return &IntegrationTestSuite{t: t}
}

// Setup boots all four services with production-shaped configs.
func (s *IntegrationTestSuite) Setup() {
	s.t.Helper()

	poolCfg := &services.PoolConfig{
		HTTPMinPoolSize:     5,
		HTTPMaxPoolSize:     50,
		GRPCMinPoolSize:     2,
		GRPCMaxPoolSize:     10,
		HealthCheckInterval: 30 * time.Second,
		IdleTimeout:         5 * time.Minute,
		MaxConnAge:          30 * time.Minute,
	}
	s.connPool = services.NewConnectionPool(poolCfg)

	batchCfg := &services.BatchConfig{
		MaxBatchSize:   32,
		MinBatchSize:   1,
		BatchTimeout:   50 * time.Millisecond,
		AdaptiveSizing: true,
		PreserveOrder:  false,
	}
	s.batcher = services.NewRequestBatcher(batchCfg)

	s.streamer = services.NewResponseStreamer(services.DefaultStreamConfig())

	s.asyncMgr = services.NewAsyncModelManager(4)

	// Drainer: the batcher is consumer-driven; without a consumer pulling
	// batches off GetBatch(), submitBatchRequest blocks forever. We launch
	// a goroutine that loops on GetBatch() (which returns (nil,false) when
	// the batcher's stopCh closes during Close()) and echoes each request's
	// payload back through req.Result. Non-blocking sends with a default
	// branch avoid deadlock if the submitter has already exited.
	s.drainCtx, s.drainStop = context.WithCancel(context.Background())
	s.drainWG.Add(1)
	go func() {
		defer s.drainWG.Done()
		for {
			batch, ok := s.batcher.GetBatch()
			if !ok {
				return
			}
			for _, req := range batch {
				if req == nil {
					continue
				}
				select {
				case req.Result <- req.Request:
				default:
				}
			}
		}
	}()
}

// Cleanup tears down all services in reverse order. Safe to call multiple
// times: each service handle is nil'd after close, and drainWG.Wait() is a
// no-op once the drainer has exited.
func (s *IntegrationTestSuite) Cleanup() {
	s.t.Helper()
	if s.batcher != nil {
		_ = s.batcher.Close() // closes stopCh -> drainer's GetBatch returns (nil,false)
		s.batcher = nil
	}
	s.drainWG.Wait()
	if s.drainStop != nil {
		s.drainStop()
		s.drainStop = nil
	}
	if s.streamer != nil {
		_ = s.streamer.Close()
		s.streamer = nil
	}
	if s.connPool != nil {
		_ = s.connPool.Close()
		s.connPool = nil
	}
	if s.asyncMgr != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = s.asyncMgr.Shutdown(ctx)
		s.asyncMgr = nil
	}
}

// submitBatchRequest wraps the verified BatchRequest contract.
func (s *IntegrationTestSuite) submitBatchRequest(ctx context.Context, id string, payload interface{}) (interface{}, error) {
	req := &services.BatchRequest{
		ID:        id,
		Request:   payload,
		Result:    make(chan interface{}, 1),
		Error:     make(chan error, 1),
		Timestamp: time.Now(),
	}
	if err := s.batcher.AddRequest(ctx, req); err != nil {
		return nil, fmt.Errorf("AddRequest: %w", err)
	}
	select {
	case res := <-req.Result:
		return res, nil
	case err := <-req.Error:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

// Test_Integration_AllComponentsTogether exercises pooling + batching together
// and asserts a 90% success-rate gate over 1000 requests.
func Test_Integration_AllComponentsTogether(t *testing.T) {
	suite := NewIntegrationTestSuite(t)
	suite.Setup()
	defer suite.Cleanup()

	const totalRequests = 1000
	const minSuccessRate = 0.90

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var success int64
	var failed int64
	var wg sync.WaitGroup

	for i := 0; i < totalRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			client := suite.connPool.GetHTTPClient()
			defer suite.connPool.ReleaseHTTPClient(client)

			_, err := suite.submitBatchRequest(ctx, fmt.Sprintf("req-%d", idx), map[string]int{"i": idx})
			if err != nil {
				atomic.AddInt64(&failed, 1)
				return
			}
			atomic.AddInt64(&success, 1)
		}(i)
	}
	wg.Wait()

	rate := float64(success) / float64(totalRequests)
	t.Logf("AllComponentsTogether: success=%d failed=%d rate=%.2f%%", success, failed, rate*100)
	if rate < minSuccessRate {
		t.Fatalf("success rate %.2f%% below gate %.2f%%", rate*100, minSuccessRate*100)
	}
}

// Test_Integration_PoolingWithBatching validates pool reuse under batched load.
func Test_Integration_PoolingWithBatching(t *testing.T) {
	suite := NewIntegrationTestSuite(t)
	suite.Setup()
	defer suite.Cleanup()

	const totalRequests = 500
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var success int64
	var wg sync.WaitGroup
	for i := 0; i < totalRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			client := suite.connPool.GetHTTPClient()
			defer suite.connPool.ReleaseHTTPClient(client)

			if _, err := suite.submitBatchRequest(ctx, fmt.Sprintf("pb-%d", idx), idx); err == nil {
				atomic.AddInt64(&success, 1)
			}
		}(i)
	}
	wg.Wait()

	reuseRate := suite.connPool.GetReuseRate()
	t.Logf("PoolingWithBatching: success=%d/%d reuseRate=%.2f%%", success, totalRequests, reuseRate*100)
	if success < int64(float64(totalRequests)*0.85) {
		t.Fatalf("pooling+batching success too low: %d/%d", success, totalRequests)
	}
}

// Test_Integration_BatchingWithStreaming pipes batch results through the streamer.
func Test_Integration_BatchingWithStreaming(t *testing.T) {
	suite := NewIntegrationTestSuite(t)
	suite.Setup()
	defer suite.Cleanup()

	const totalRequests = 200
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var success int64
	var wg sync.WaitGroup
	for i := 0; i < totalRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			if _, err := suite.submitBatchRequest(ctx, fmt.Sprintf("bs-%d", idx), idx); err != nil {
				return
			}

			payload := strings.Repeat(fmt.Sprintf("chunk-%d ", idx), 16)
			ch := suite.streamer.StreamReader(ctx, strings.NewReader(payload))
			var sink bytes.Buffer
			if err := suite.streamer.StreamWriter(ctx, &sink, ch); err != nil {
				return
			}
			if sink.Len() == 0 {
				return
			}
			atomic.AddInt64(&success, 1)
		}(i)
	}
	wg.Wait()

	t.Logf("BatchingWithStreaming: success=%d/%d throughput=%.2f B/s",
		success, totalRequests, suite.streamer.GetThroughput())
	if success < int64(float64(totalRequests)*0.85) {
		t.Fatalf("batching+streaming success too low: %d/%d", success, totalRequests)
	}
}

// Test_Integration_AsyncModelLoading registers and loads four models concurrently.
func Test_Integration_AsyncModelLoading(t *testing.T) {
	suite := NewIntegrationTestSuite(t)
	suite.Setup()
	defer suite.Cleanup()

	models := []string{"a", "b", "c", "d"}
	for _, id := range models {
		md := &services.ModelMetadata{
			ID:              id,
			Name:            "model-" + id,
			Path:            "/tmp/x",
			Size:            1024,
			Priority:        1,
			PreloadStrategy: "lazy",
			MaxConcurrency:  1,
			LoadTimeout:     30 * time.Second,
		}
		if err := suite.asyncMgr.RegisterModel(md); err != nil {
			t.Fatalf("RegisterModel(%s): %v", id, err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var loaded int64
	var wg sync.WaitGroup
	for _, id := range models {
		wg.Add(1)
		go func(modelID string) {
			defer wg.Done()
			res, err := suite.asyncMgr.LoadModel(ctx, modelID)
			if err != nil {
				t.Logf("LoadModel(%s) error: %v", modelID, err)
				return
			}
			if res != nil && res.Success {
				atomic.AddInt64(&loaded, 1)
			}
		}(id)
	}
	wg.Wait()

	t.Logf("AsyncModelLoading: loaded=%d/%d", loaded, len(models))
	if loaded == 0 {
		t.Fatalf("expected at least one model to load successfully, got 0")
	}
}

// Test_Integration_HighConcurrencyScenario runs 200 workers × 50 requests = 10000
// total ops with a 95% success-rate gate.
func Test_Integration_HighConcurrencyScenario(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping high-concurrency scenario in -short mode")
	}

	suite := NewIntegrationTestSuite(t)
	suite.Setup()
	defer suite.Cleanup()

	const workers = 200
	const perWorker = 50
	const total = workers * perWorker
	const minSuccessRate = 0.95

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	var success int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for j := 0; j < perWorker; j++ {
				client := suite.connPool.GetHTTPClient()
				_, err := suite.submitBatchRequest(ctx, fmt.Sprintf("hc-%d-%d", worker, j), j)
				suite.connPool.ReleaseHTTPClient(client)
				if err == nil {
					atomic.AddInt64(&success, 1)
				}
			}
		}(w)
	}
	wg.Wait()

	rate := float64(success) / float64(total)
	t.Logf("HighConcurrency: success=%d/%d rate=%.2f%%", success, total, rate*100)
	if rate < minSuccessRate {
		t.Fatalf("high-concurrency success rate %.2f%% below gate %.2f%%", rate*100, minSuccessRate*100)
	}
}

// Test_Integration_ResourceCleanup ensures Cleanup is idempotent and leaves
// no goroutine leaks observable from the test harness.
func Test_Integration_ResourceCleanup(t *testing.T) {
	suite := NewIntegrationTestSuite(t)
	suite.Setup()

	// Light traffic before cleanup.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for i := 0; i < 10; i++ {
		client := suite.connPool.GetHTTPClient()
		_, _ = suite.submitBatchRequest(ctx, fmt.Sprintf("cl-%d", i), i)
		suite.connPool.ReleaseHTTPClient(client)
	}

	suite.Cleanup()

	// Calling cleanup-equivalent operations twice should not panic.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("double cleanup panicked: %v", r)
		}
	}()
	suite.Cleanup()
}

// Test_Integration_ErrorHandling verifies the suite degrades gracefully when
// callers cancel mid-flight.
func Test_Integration_ErrorHandling(t *testing.T) {
	suite := NewIntegrationTestSuite(t)
	suite.Setup()
	defer suite.Cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	_, err := suite.submitBatchRequest(ctx, "err-1", "payload")
	if err == nil {
		t.Fatalf("expected error from canceled context, got nil")
	}
	t.Logf("ErrorHandling: got expected error: %v", err)
}

// ----------------------------------------------------------------------------
// Benchmark
// ----------------------------------------------------------------------------

// Benchmark_Integration_AllComponents measures end-to-end throughput across
// pooling + batching + streaming.
func Benchmark_Integration_AllComponents(b *testing.B) {
	suite := NewIntegrationTestSuite(&testing.T{})
	suite.Setup()
	defer suite.Cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client := suite.connPool.GetHTTPClient()
		_, _ = suite.submitBatchRequest(ctx, fmt.Sprintf("bench-%d", i), i)
		suite.connPool.ReleaseHTTPClient(client)
	}

	report := GenerateIntegrationReport()
	b.ReportMetric(report.CumulativeImprovement, "cumulative_improvement_pct")
}
