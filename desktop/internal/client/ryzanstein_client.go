package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// RyzansteinClient handles REST API communication with Ryzanstein inference server
type RyzansteinClient struct {
	baseURL    string
	httpClient *http.Client
	timeout    time.Duration
	maxRetries int
	retryDelay time.Duration
}

// InferenceRequest represents a request to the inference API
type InferenceRequest struct {
	Prompt      string  `json:"prompt"`
	Model       string  `json:"model"`
	Temperature float32 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

// InferenceResponse represents a response from the inference API
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

// ModelInfo represents metadata about a model
type ModelInfo struct {
	ID              string `json:"id"`
	Name            string `json:"name"`
	ContextWindow   int    `json:"context_window"`
	MaxOutputTokens int    `json:"max_output_tokens"`
	Type            string `json:"type"`
	Quantization    string `json:"quantization"`
}

// RyzansteinError represents an error from the API
type RyzansteinError struct {
	Code    string                 `json:"code"`
	Message string                 `json:"message"`
	Details map[string]interface{} `json:"details,omitempty"`
}

func (e *RyzansteinError) Error() string {
	return fmt.Sprintf("Ryzanstein error [%s]: %s", e.Code, e.Message)
}

// NewRyzansteinClient creates a new client for the Ryzanstein API
func NewRyzansteinClient(baseURL string) *RyzansteinClient {
	return &RyzansteinClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		timeout:    30 * time.Second,
		maxRetries: 3,
		retryDelay: time.Second,
	}
}

// SetTimeout sets the request timeout
func (c *RyzansteinClient) SetTimeout(timeout time.Duration) {
	c.timeout = timeout
	c.httpClient.Timeout = timeout
}

// SetMaxRetries sets the maximum number of retries
func (c *RyzansteinClient) SetMaxRetries(maxRetries int) {
	c.maxRetries = maxRetries
}

// GetBaseURL returns the configured API base URL
func (c *RyzansteinClient) GetBaseURL() string {
	return c.baseURL
}

// Infer makes an inference request to the API
func (c *RyzansteinClient) Infer(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		resp, err := c.inferOnce(ctx, req)
		if err == nil {
			return resp, nil
		}

		lastErr = err
		if attempt < c.maxRetries {
			// Exponential backoff
			backoff := c.retryDelay * time.Duration(1<<uint(attempt))
			select {
			case <-time.After(backoff):
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	return nil, fmt.Errorf("inference failed after %d retries: %w", c.maxRetries, lastErr)
}

func (c *RyzansteinClient) inferOnce(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= 400 {
		var apiErr RyzansteinError
		if err := json.NewDecoder(httpResp.Body).Decode(&apiErr); err != nil {
			return nil, fmt.Errorf("API error (status %d): failed to parse error response", httpResp.StatusCode)
		}
		return nil, &apiErr
	}

	var resp InferenceResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// ListModels gets the list of available models
func (c *RyzansteinClient) ListModels(ctx context.Context) ([]ModelInfo, error) {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		models, err := c.listModelsOnce(ctx)
		if err == nil {
			return models, nil
		}

		lastErr = err
		if attempt < c.maxRetries {
			backoff := c.retryDelay * time.Duration(1<<uint(attempt))
			select {
			case <-time.After(backoff):
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	return nil, fmt.Errorf("list models failed after %d retries: %w", c.maxRetries, lastErr)
}

func (c *RyzansteinClient) listModelsOnce(ctx context.Context) ([]ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		var apiErr RyzansteinError
		if err := json.NewDecoder(httpResp.Body).Decode(&apiErr); err != nil {
			return nil, fmt.Errorf("API error (status %d)", httpResp.StatusCode)
		}
		return nil, &apiErr
	}

	var resp struct {
		Data []ModelInfo `json:"data"`
	}
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return resp.Data, nil
}

// LoadModel loads a model into memory
func (c *RyzansteinClient) LoadModel(ctx context.Context, modelID string) error {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		err := c.loadModelOnce(ctx, modelID)
		if err == nil {
			return nil
		}

		lastErr = err
		if attempt < c.maxRetries {
			backoff := c.retryDelay * time.Duration(1<<uint(attempt))
			select {
			case <-time.After(backoff):
				continue
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	return fmt.Errorf("load model failed after %d retries: %w", c.maxRetries, lastErr)
}

func (c *RyzansteinClient) loadModelOnce(ctx context.Context, modelID string) error {
	body := fmt.Sprintf(`{"model_id": "%s"}`, modelID)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/models/load", bytes.NewBufferString(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		var apiErr RyzansteinError
		if err := json.NewDecoder(httpResp.Body).Decode(&apiErr); err != nil {
			return fmt.Errorf("API error (status %d)", httpResp.StatusCode)
		}
		return &apiErr
	}

	return nil
}

// UnloadModel unloads a model from memory
func (c *RyzansteinClient) UnloadModel(ctx context.Context, modelID string) error {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		err := c.unloadModelOnce(ctx, modelID)
		if err == nil {
			return nil
		}

		lastErr = err
		if attempt < c.maxRetries {
			backoff := c.retryDelay * time.Duration(1<<uint(attempt))
			select {
			case <-time.After(backoff):
				continue
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	return fmt.Errorf("unload model failed after %d retries: %w", c.maxRetries, lastErr)
}

func (c *RyzansteinClient) unloadModelOnce(ctx context.Context, modelID string) error {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/v1/models/%s/unload", c.baseURL, modelID), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		var apiErr RyzansteinError
		if err := json.NewDecoder(httpResp.Body).Decode(&apiErr); err != nil {
			return fmt.Errorf("API error (status %d)", httpResp.StatusCode)
		}
		return &apiErr
	}

	return nil
}

// ChatMessage represents a message in the OpenAI-compatible chat format
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest represents an OpenAI-compatible chat completion request
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float32       `json:"temperature,omitempty"`
	TopP        float32       `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

// ChatCompletionResponse represents an OpenAI-compatible chat completion response
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

// ChatCompletion makes an OpenAI-compatible chat completion request
func (c *RyzansteinClient) ChatCompletion(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= 400 {
		var apiErr RyzansteinError
		if err := json.NewDecoder(httpResp.Body).Decode(&apiErr); err != nil {
			return nil, fmt.Errorf("API error (status %d)", httpResp.StatusCode)
		}
		return nil, &apiErr
	}

	var resp ChatCompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// ChatCompletionStream opens an SSE stream for chat completions and sends tokens to the channel
func (c *RyzansteinClient) ChatCompletionStream(ctx context.Context, req *ChatCompletionRequest, tokenChan chan<- string) error {
	req.Stream = true

	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= 400 {
		var apiErr RyzansteinError
		if err := json.NewDecoder(httpResp.Body).Decode(&apiErr); err != nil {
			return fmt.Errorf("API error (status %d)", httpResp.StatusCode)
		}
		return &apiErr
	}

	// Read SSE stream
	buf := make([]byte, 4096)
	for {
		n, err := httpResp.Body.Read(buf)
		if n > 0 {
			chunk := string(buf[:n])
			// Parse SSE data lines
			for _, line := range splitLines(chunk) {
				if len(line) > 6 && line[:6] == "data: " {
					data := line[6:]
					if data == "[DONE]" {
						return nil
					}
					// Parse the JSON chunk to extract content delta
					var streamChunk struct {
						Choices []struct {
							Delta struct {
								Content string `json:"content"`
							} `json:"delta"`
						} `json:"choices"`
					}
					if json.Unmarshal([]byte(data), &streamChunk) == nil {
						for _, choice := range streamChunk.Choices {
							if choice.Delta.Content != "" {
								tokenChan <- choice.Delta.Content
							}
						}
					}
				}
			}
		}
		if err != nil {
			if err.Error() == "EOF" {
				return nil
			}
			return fmt.Errorf("stream read error: %w", err)
		}
	}
}

// splitLines splits a string into lines, handling both \n and \r\n
func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			line := s[start:i]
			if len(line) > 0 && line[len(line)-1] == '\r' {
				line = line[:len(line)-1]
			}
			if line != "" {
				lines = append(lines, line)
			}
			start = i + 1
		}
	}
	if start < len(s) {
		line := s[start:]
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

// Health checks if the API is healthy
func (c *RyzansteinClient) Health(ctx context.Context) (bool, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return false, fmt.Errorf("failed to create request: %w", err)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return false, fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	return httpResp.StatusCode == http.StatusOK, nil
}
