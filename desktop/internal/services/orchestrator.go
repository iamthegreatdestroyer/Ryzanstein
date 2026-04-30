package services

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// OrchestratorConfig aggregates configuration for all pipeline components.
type OrchestratorConfig struct {
	Pool              *PoolConfig
	Batch             *BatchConfig
	Stream            *StreamConfig
	MaxConcurrentLoad int
}

// DefaultOrchestratorConfig returns production-ready defaults.
func DefaultOrchestratorConfig() *OrchestratorConfig {
	return &OrchestratorConfig{
		Pool:              DefaultPoolConfig(),
		Batch:             DefaultBatchConfig(),
		Stream:            DefaultStreamConfig(),
		MaxConcurrentLoad: 4,
	}
}

// PipelineRequest represents a single inference request through the pipeline.
type PipelineRequest struct {
	ID        string
	ModelID   string
	Input     interface{}
	Priority  int
	Timestamp time.Time
}

// PipelineResult is the output of a processed pipeline request.
type PipelineResult struct {
	RequestID  string
	ModelID    string
	Output     interface{}
	Latency    time.Duration
	CacheHit   bool
	StreamedAt time.Time
}

// OrchestratorMetrics aggregates metrics from all pipeline components.
type OrchestratorMetrics struct {
	TotalRequests    int64
	TotalCompleted   int64
	TotalFailed      int64
	AverageLatency   float64
	P99Latency       float64
	Throughput       float64
	PipelineDepth    int32
	PoolMetrics      *PoolMetrics
	BatchMetrics     *BatchMetrics
	StreamMetrics    *StreamMetrics
	ModelStats       map[string]interface{}
	UptimeSeconds    float64
}

// Orchestrator coordinates all pipeline components for end-to-end inference.
type Orchestrator struct {
	config   *OrchestratorConfig
	pool     *ConnectionPool
	batcher  *RequestBatcher
	streamer *ResponseStreamer
	models   *AsyncModelManager

	// Pipeline state
	pipelineDepth int32
	totalReqs     int64
	totalDone     int64
	totalFailed   int64
	latencies     []time.Duration
	latencyMu     sync.Mutex

	startTime time.Time
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	stopped   int32
}

// NewOrchestrator creates a fully wired pipeline from all components.
func NewOrchestrator(config *OrchestratorConfig) *Orchestrator {
	if config == nil {
		config = DefaultOrchestratorConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	o := &Orchestrator{
		config:    config,
		pool:      NewConnectionPool(config.Pool),
		batcher:   NewRequestBatcher(config.Batch),
		streamer:  NewResponseStreamer(config.Stream),
		models:    NewAsyncModelManager(config.MaxConcurrentLoad),
		latencies: make([]time.Duration, 0, 1024),
		startTime: time.Now(),
		ctx:       ctx,
		cancel:    cancel,
	}

	// Start the batch processing pipeline
	o.wg.Add(1)
	go o.batchProcessingLoop()

	return o
}

// ProcessRequest sends a single request through the full pipeline:
// Client → Pool → Batcher → Model Load → Inference → Streamer → Result
func (o *Orchestrator) ProcessRequest(ctx context.Context, req *PipelineRequest) (*PipelineResult, error) {
	if atomic.LoadInt32(&o.stopped) == 1 {
		return nil, fmt.Errorf("orchestrator is stopped")
	}

	start := time.Now()
	atomic.AddInt64(&o.totalReqs, 1)
	atomic.AddInt32(&o.pipelineDepth, 1)
	defer atomic.AddInt32(&o.pipelineDepth, -1)

	// Step 1: Acquire connection from pool
	client := o.pool.GetHTTPClient()
	defer o.pool.ReleaseHTTPClient(client)

	// Step 2: Submit to batcher
	resultCh := make(chan interface{}, 1)
	errCh := make(chan error, 1)

	batchReq := &BatchRequest{
		ID:        req.ID,
		Request:   req,
		Result:    resultCh,
		Error:     errCh,
		Timestamp: time.Now(),
	}

	if err := o.batcher.AddRequest(ctx, batchReq); err != nil {
		atomic.AddInt64(&o.totalFailed, 1)
		return nil, fmt.Errorf("batcher rejected request: %w", err)
	}

	// Step 3: Wait for batch processing result
	select {
	case result := <-resultCh:
		latency := time.Since(start)
		o.recordLatency(latency)
		atomic.AddInt64(&o.totalDone, 1)

		pResult, ok := result.(*PipelineResult)
		if !ok {
			// Wrap raw result
			pResult = &PipelineResult{
				RequestID:  req.ID,
				ModelID:    req.ModelID,
				Output:     result,
				Latency:    latency,
				StreamedAt: time.Now(),
			}
		}
		pResult.Latency = latency
		return pResult, nil

	case err := <-errCh:
		atomic.AddInt64(&o.totalFailed, 1)
		return nil, fmt.Errorf("pipeline processing failed: %w", err)

	case <-ctx.Done():
		atomic.AddInt64(&o.totalFailed, 1)
		return nil, ctx.Err()
	}
}

// ProcessBatch sends multiple requests through the pipeline concurrently.
func (o *Orchestrator) ProcessBatch(ctx context.Context, requests []*PipelineRequest) ([]*PipelineResult, error) {
	if atomic.LoadInt32(&o.stopped) == 1 {
		return nil, fmt.Errorf("orchestrator is stopped")
	}

	results := make([]*PipelineResult, len(requests))
	errs := make([]error, len(requests))
	var wg sync.WaitGroup

	for i, req := range requests {
		wg.Add(1)
		go func(idx int, r *PipelineRequest) {
			defer wg.Done()
			result, err := o.ProcessRequest(ctx, r)
			if err != nil {
				errs[idx] = err
				return
			}
			results[idx] = result
		}(i, req)
	}

	wg.Wait()

	// Collect errors
	var errMsgs []string
	for i, err := range errs {
		if err != nil {
			errMsgs = append(errMsgs, fmt.Sprintf("request[%d]: %v", i, err))
		}
	}

	if len(errMsgs) > 0 {
		return results, fmt.Errorf("batch errors (%d/%d failed): %s",
			len(errMsgs), len(requests), strings.Join(errMsgs, "; "))
	}

	return results, nil
}

// StreamResponse streams inference output through the ResponseStreamer.
func (o *Orchestrator) StreamResponse(ctx context.Context, w http.ResponseWriter, data io.Reader) error {
	if atomic.LoadInt32(&o.stopped) == 1 {
		return fmt.Errorf("orchestrator is stopped")
	}
	return o.streamer.StreamHTTPResponse(ctx, w, data)
}

// StreamData creates a channel-based streaming pipeline for arbitrary data.
func (o *Orchestrator) StreamData(ctx context.Context, reader io.Reader) chan *StreamChunk {
	return o.streamer.StreamReader(ctx, reader)
}

// RegisterModel registers a model with the async model manager.
func (o *Orchestrator) RegisterModel(metadata *ModelMetadata) error {
	return o.models.RegisterModel(metadata)
}

// PreloadModels triggers async preloading of specified models.
func (o *Orchestrator) PreloadModels(modelIDs ...string) error {
	return o.models.PreloadModels(modelIDs...)
}

// batchProcessingLoop continuously pulls batches and processes them through
// the model loading + inference + streaming pipeline.
func (o *Orchestrator) batchProcessingLoop() {
	defer o.wg.Done()

	for {
		select {
		case <-o.ctx.Done():
			// Drain remaining batches before exit
			o.drainRemainingBatches()
			return
		default:
			batch, ok := o.batcher.GetBatchContext(o.ctx)
			if !ok {
				// Context cancelled
				return
			}
			if len(batch) == 0 {
				continue
			}

			o.processBatchInternal(batch)
		}
	}
}

// processBatchInternal handles a single batch of requests through the pipeline.
func (o *Orchestrator) processBatchInternal(batch []*BatchRequest) {
	o.batcher.ResolveBatch(batch, func(req *BatchRequest) error {
		pReq, ok := req.Request.(*PipelineRequest)
		if !ok {
			return fmt.Errorf("invalid request type in batch")
		}

		// Step 1: Load model (with caching)
		var loadResult *ModelLoadResult
		if pReq.ModelID != "" {
			var err error
			loadResult, err = o.models.LoadModel(o.ctx, pReq.ModelID)
			if err != nil {
				return fmt.Errorf("model load failed for %s: %w", pReq.ModelID, err)
			}
		}

		// Step 2: Simulate inference (in real system, this calls the model)
		result := &PipelineResult{
			RequestID:  pReq.ID,
			ModelID:    pReq.ModelID,
			Output:     fmt.Sprintf("inference_result_%s", pReq.ID),
			StreamedAt: time.Now(),
		}

		if loadResult != nil {
			result.CacheHit = loadResult.CacheHit
		}

		return nil
	})
}

// drainRemainingBatches processes any batches left after shutdown signal.
func (o *Orchestrator) drainRemainingBatches() {
	for {
		batch, ok := o.batcher.GetBatch()
		if !ok || len(batch) == 0 {
			return
		}
		o.processBatchInternal(batch)
	}
}

// GetMetrics returns aggregated metrics from all pipeline components.
func (o *Orchestrator) GetMetrics() *OrchestratorMetrics {
	uptime := time.Since(o.startTime).Seconds()
	totalDone := atomic.LoadInt64(&o.totalDone)

	var throughput float64
	if uptime > 0 {
		throughput = float64(totalDone) / uptime
	}

	return &OrchestratorMetrics{
		TotalRequests:  atomic.LoadInt64(&o.totalReqs),
		TotalCompleted: totalDone,
		TotalFailed:    atomic.LoadInt64(&o.totalFailed),
		AverageLatency: o.calculateAverageLatency(),
		P99Latency:     o.calculateP99Latency(),
		Throughput:     throughput,
		PipelineDepth:  atomic.LoadInt32(&o.pipelineDepth),
		PoolMetrics:    o.pool.GetMetrics(),
		BatchMetrics:   o.batcher.GetMetrics(),
		StreamMetrics:  o.streamer.GetMetrics(),
		ModelStats:     o.models.GetModelStats(),
		UptimeSeconds:  uptime,
	}
}

// GetPool returns the underlying connection pool for direct access.
func (o *Orchestrator) GetPool() *ConnectionPool {
	return o.pool
}

// GetBatcher returns the underlying request batcher.
func (o *Orchestrator) GetBatcher() *RequestBatcher {
	return o.batcher
}

// GetStreamer returns the underlying response streamer.
func (o *Orchestrator) GetStreamer() *ResponseStreamer {
	return o.streamer
}

// GetModelManager returns the underlying async model manager.
func (o *Orchestrator) GetModelManager() *AsyncModelManager {
	return o.models
}

// Stop gracefully shuts down all pipeline components.
func (o *Orchestrator) Stop() {
	if !atomic.CompareAndSwapInt32(&o.stopped, 0, 1) {
		return // Already stopped
	}

	// Signal shutdown
	o.cancel()

	// Wait for batch processing loop to finish
	o.wg.Wait()

	// Shutdown components in reverse order
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	o.models.Shutdown(shutdownCtx)
	o.streamer.Close()
	o.batcher.Close()
	o.pool.Close()
}

// recordLatency records a request latency for metrics calculation.
func (o *Orchestrator) recordLatency(d time.Duration) {
	o.latencyMu.Lock()
	defer o.latencyMu.Unlock()

	// Circular buffer: keep last 10000 latencies
	if len(o.latencies) >= 10000 {
		o.latencies = o.latencies[1:]
	}
	o.latencies = append(o.latencies, d)
}

// calculateAverageLatency computes mean latency from recorded samples.
func (o *Orchestrator) calculateAverageLatency() float64 {
	o.latencyMu.Lock()
	defer o.latencyMu.Unlock()

	if len(o.latencies) == 0 {
		return 0
	}

	var total time.Duration
	for _, d := range o.latencies {
		total += d
	}
	return float64(total.Milliseconds()) / float64(len(o.latencies))
}

// calculateP99Latency computes the 99th percentile latency.
func (o *Orchestrator) calculateP99Latency() float64 {
	o.latencyMu.Lock()
	defer o.latencyMu.Unlock()

	n := len(o.latencies)
	if n == 0 {
		return 0
	}

	// Copy and sort
	sorted := make([]time.Duration, n)
	copy(sorted, o.latencies)

	// Simple insertion sort for bounded buffer (fast for small n or nearly sorted)
	for i := 1; i < n; i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > key {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}

	idx := int(float64(n) * 0.99)
	if idx >= n {
		idx = n - 1
	}
	return float64(sorted[idx].Milliseconds())
}
