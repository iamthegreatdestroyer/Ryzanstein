package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ClientManager handles REST and gRPC client initialization and lifecycle
type ClientManager struct {
	config *config.AppConfig

	// REST client and base endpoint
	restClient  *http.Client
	restBaseURL string

	// gRPC client
	grpcConn *grpc.ClientConn

	// Lifecycle management
	mu          sync.RWMutex
	initialized bool
	closed      bool
	ctx         context.Context
	cancel      context.CancelFunc

	// Metrics
	createdAt time.Time
	requests  int64
}

// NewClientManager creates a new client manager with given config
func NewClientManager(cfg *config.AppConfig) *ClientManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &ClientManager{
		config:    cfg,
		ctx:       ctx,
		cancel:    cancel,
		createdAt: time.Now(),
		requests:  0,
	}
}

// Initialize sets up REST and gRPC clients based on configuration
func (cm *ClientManager) Initialize() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if cm.initialized {
		return fmt.Errorf("client manager already initialized")
	}

	if cm.closed {
		return fmt.Errorf("client manager is closed")
	}

	// Initialize based on server type configuration
	switch cm.config.Server.Protocol {
	case "rest":
		if err := cm.initializeRESTClient(); err != nil {
			return fmt.Errorf("failed to initialize REST client: %w", err)
		}
	case "grpc":
		if err := cm.initializeGRPCClient(); err != nil {
			return fmt.Errorf("failed to initialize gRPC client: %w", err)
		}
	case "hybrid":
		if err := cm.initializeRESTClient(); err != nil {
			return fmt.Errorf("failed to initialize REST client: %w", err)
		}
		if err := cm.initializeGRPCClient(); err != nil {
			// Log warning but don't fail on gRPC if REST works
			fmt.Printf("warning: gRPC initialization failed: %v\n", err)
		}
	default:
		return fmt.Errorf("unsupported protocol: %s", cm.config.Server.Protocol)
	}

	cm.initialized = true
	return nil
}

// initializeRESTClient sets up a real *http.Client with timeouts.
func (cm *ClientManager) initializeRESTClient() error {
	if cm.config.Server.Host == "" || cm.config.Server.Port == 0 {
		return fmt.Errorf("invalid REST configuration: host or port missing")
	}

	timeout := cm.config.Inference.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	cm.restBaseURL = fmt.Sprintf("http://%s:%d", cm.config.Server.Host, cm.config.Server.Port)
	cm.restClient = &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  false,
		},
	}

	// Probe the health endpoint to verify the server is reachable.
	healthURL := cm.restBaseURL + "/health"
	probeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, healthURL, nil)
	if err != nil {
		// Server may not be started yet — not a fatal error, but log it.
		fmt.Printf("[ClientManager] health probe request creation failed: %v (continuing)\n", err)
		return nil
	}

	resp, err := cm.restClient.Do(req)
	if err != nil {
		// Server offline at init time — record warning but don't block startup.
		fmt.Printf("[ClientManager] REST server unreachable at %s (will retry on first request): %v\n", cm.restBaseURL, err)
		return nil
	}
	resp.Body.Close()

	fmt.Printf("[ClientManager] REST client connected to %s (status %d)\n", cm.restBaseURL, resp.StatusCode)
	return nil
}

// initializeGRPCClient sets up gRPC client with proper configuration
func (cm *ClientManager) initializeGRPCClient() error {
	if cm.config.Server.Host == "" || cm.config.Server.Port == 0 {
		return fmt.Errorf("invalid gRPC configuration: host or port missing")
	}

	// Create gRPC connection with timeouts
	endpoint := fmt.Sprintf("%s:%d", cm.config.Server.Host, cm.config.Server.Port)
	timeout := cm.config.Inference.Timeout

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Attempt connection
	conn, err := grpc.DialContext(
		ctx,
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithReturnConnectionError(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to gRPC server %s: %w", endpoint, err)
	}

	cm.grpcConn = conn
	return nil
}

// GetRESTEndpoint returns the configured REST base URL.
func (cm *ClientManager) GetRESTEndpoint() (string, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if !cm.initialized {
		return "", fmt.Errorf("client manager not initialized")
	}

	if cm.restBaseURL == "" {
		return "", fmt.Errorf("REST client not available")
	}

	return cm.restBaseURL, nil
}

// GetGRPCConnection returns the gRPC connection
func (cm *ClientManager) GetGRPCConnection() (*grpc.ClientConn, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if !cm.initialized {
		return nil, fmt.Errorf("client manager not initialized")
	}

	if cm.grpcConn == nil {
		return nil, fmt.Errorf("gRPC client not available")
	}

	return cm.grpcConn, nil
}

// ExecuteWithRouting routes request based on configuration
func (cm *ClientManager) ExecuteWithRouting(ctx context.Context, operation string, data interface{}) (interface{}, error) {
	cm.mu.Lock()
	cm.requests++
	cm.mu.Unlock()

	if !cm.IsInitialized() {
		return nil, fmt.Errorf("client manager not initialized")
	}

	switch cm.config.Server.Protocol {
	case "rest":
		return cm.executeREST(ctx, operation, data)
	case "grpc":
		return cm.executeGRPC(ctx, operation, data)
	case "hybrid":
		// Try gRPC first, fall back to REST
		result, err := cm.executeGRPC(ctx, operation, data)
		if err != nil {
			return cm.executeREST(ctx, operation, data)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported protocol: %s", cm.config.Server.Protocol)
	}
}

// executeREST handles REST-based requests against the Ryzanstein API.
func (cm *ClientManager) executeREST(ctx context.Context, operation string, data interface{}) (interface{}, error) {
	if cm.restClient == nil {
		return nil, fmt.Errorf("REST client not initialized")
	}

	var urlPath string
	switch operation {
	case "infer":
		urlPath = "/v1/completions"
	case "infer_stream":
		urlPath = "/v1/completions"
	case "list_models":
		urlPath = "/v1/models"
	case "load_model":
		urlPath = "/v1/models/load"
	case "unload_model":
		urlPath = "/v1/models/unload"
	default:
		urlPath = "/v1/" + operation
	}

	fullURL := cm.restBaseURL + urlPath

	// GET-style operations carry no body.
	if operation == "list_models" {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, fullURL, nil)
		if err != nil {
			return nil, fmt.Errorf("build GET request: %w", err)
		}
		req.Header.Set("Accept", "application/json")

		resp, err := cm.restClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("GET %s: %w", fullURL, err)
		}
		defer resp.Body.Close()

		var result interface{}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, fmt.Errorf("decode GET response: %w", err)
		}
		return result, nil
	}

	// POST-style operations.
	bodyBytes, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fullURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("build POST request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := cm.restClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("POST %s: %w", fullURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	var result interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode POST response: %w", err)
	}
	return result, nil
}

// executeGRPC handles gRPC-based requests
func (cm *ClientManager) executeGRPC(ctx context.Context, operation string, data interface{}) (interface{}, error) {
	conn, err := cm.GetGRPCConnection()
	if err != nil {
		return nil, err
	}

	// Verify connection is still open
	if conn.GetState().String() == "SHUTDOWN" {
		return nil, fmt.Errorf("gRPC connection is shutdown")
	}

	// Simulate gRPC request with context
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("context cancelled during gRPC execution")
	default:
	}

	// Return operation result (would use actual gRPC client in real implementation)
	return map[string]interface{}{
		"operation": operation,
		"protocol":  "gRPC",
		"data":      data,
	}, nil
}

// IsInitialized returns whether client manager is ready
func (cm *ClientManager) IsInitialized() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.initialized && !cm.closed
}

// Close closes all client connections and resources
func (cm *ClientManager) Close() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if cm.closed {
		return fmt.Errorf("client manager already closed")
	}

	// Close gRPC connection if exists
	if cm.grpcConn != nil {
		cm.grpcConn.Close()
	}

	// Cancel context
	cm.cancel()

	cm.closed = true
	return nil
}

// GetMetrics returns current metrics
func (cm *ClientManager) GetMetrics() map[string]interface{} {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return map[string]interface{}{
		"initialized": cm.initialized,
		"requests":    cm.requests,
		"uptime":      time.Since(cm.createdAt).Seconds(),
		"server_type": cm.config.Server.Protocol,
	}
}
