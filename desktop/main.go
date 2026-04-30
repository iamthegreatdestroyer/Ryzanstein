package main

// This file is the entry point for Wails
// The actual implementation is in cmd/ryzanstein/main.go

import (
	"context"
	"embed"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/agents"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/chat"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/client"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/ipc"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/models"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/services"
	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

//go:embed all:frontend/dist
var assets embed.FS

// App struct is where we bind all application methods
type App struct {
	ctx       context.Context
	chat      *chat.Service
	models    *models.Service
	agents    *agents.Service
	config    *config.Manager
	ipc       *ipc.Server
	apiClient *client.RyzansteinClient
	logger    *services.LogService
	telemetry *services.TelemetryService
	mu        sync.RWMutex
	isRunning bool
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{
		isRunning: false,
	}
}

// Startup is called at application startup
func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
	a.isRunning = true
	log.Println("[Desktop] Application starting up")

	var err error
	a.config, err = config.NewManager()
	if err != nil {
		log.Printf("[Desktop] Failed to load config: %v\n", err)
		runtime.MessageDialog(ctx, runtime.MessageDialogOptions{
			Type:    runtime.ErrorDialog,
			Title:   "Startup Error",
			Message: fmt.Sprintf("Failed to load config: %v", err),
		})
		return
	}

	a.chat = chat.NewService()
	a.models = models.NewService(a.config)

	// Initialize the Ryzanstein API client before agents so we can inject it
	apiURL := a.config.GetConfig().RyzansteinAPIURL
	if apiURL == "" {
		apiURL = "http://localhost:8000"
	}
	a.apiClient = client.NewRyzansteinClient(apiURL)
	a.apiClient.SetTimeout(150 * time.Second)

	a.agents = agents.NewService(a.apiClient)

	a.ipc = ipc.NewServer(a.agents, a.models, a.apiClient)

	a.logger = services.NewLogService(services.LogLevelInfo)
	a.telemetry = services.NewTelemetryService()

	go a.startIPCServer()
	go a.models.LoadInstalledModels()

	runtime.EventsEmit(ctx, "app:ready", map[string]interface{}{
		"version":   "1.0.0",
		"timestamp": time.Now().Unix(),
	})

	log.Println("[Desktop] Application ready")
}

// Shutdown is called at application termination
func (a *App) Shutdown(ctx context.Context) {
	log.Println("[Desktop] Application shutting down")
	a.isRunning = false

	a.chat.Close()
	a.ipc.Close()

	log.Println("[Desktop] Application shutdown complete")
}

// ============================================================================
// Chat Service Methods
// ============================================================================

type Message struct {
	ID        string                 `json:"id"`
	Role      string                 `json:"role"`
	Content   string                 `json:"content"`
	Timestamp int64                  `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

func (a *App) SendMessage(userMessage string, modelID string, agentCodename string) (string, error) {
	log.Printf("[Chat] Sending message: %s (model: %s, agent: %s)\n",
		userMessage, modelID, agentCodename)

	inferenceStart := time.Now()
	ctx, cancel := context.WithTimeout(a.ctx, 30*time.Second)
	defer cancel()

	// Add user message to chat history
	a.chat.AddMessage(ctx, "user", userMessage, modelID, agentCodename)
	a.telemetry.RecordChatMessage()

	runtime.EventsEmit(a.ctx, "chat:message", Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   userMessage,
		Timestamp: time.Now().Unix(),
	})

	// Build system prompt with agent context
	systemPrompt := fmt.Sprintf("You are %s, an elite AI agent. Respond helpfully and concisely.", agentCodename)

	// Try real API first
	var responseText string
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

	chatResp, err := a.apiClient.ChatCompletion(ctx, chatReq)
	inferenceErrored := err != nil
	a.telemetry.RecordInference(float64(time.Since(inferenceStart).Milliseconds()), inferenceErrored)
	if err != nil {
		log.Printf("[Chat] API call failed, using fallback: %v\n", err)
		// Fallback to mock response when API is not available
		responseText = fmt.Sprintf("[Offline Mode] %s received your message. The inference API at %s is not reachable. Please start the backend with: docker-compose up -d",
			agentCodename, a.config.GetConfig().RyzansteinAPIURL)
	} else if len(chatResp.Choices) > 0 {
		responseText = chatResp.Choices[0].Message.Content
	} else {
		responseText = "[Error] Empty response from inference API."
	}

	// Add assistant response to history
	a.chat.AddMessage(ctx, "assistant", responseText, modelID, agentCodename)
	a.telemetry.RecordChatMessage()

	runtime.EventsEmit(a.ctx, "chat:response", Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "assistant",
		Content:   responseText,
		Timestamp: time.Now().Unix(),
	})

	return responseText, nil
}

// SendMessageStream sends a message and streams the response token-by-token
func (a *App) SendMessageStream(userMessage string, modelID string, agentCodename string) error {
	log.Printf("[Chat] Streaming message: %s (model: %s, agent: %s)\n",
		userMessage, modelID, agentCodename)

	streamStart := time.Now()
	ctx, cancel := context.WithTimeout(a.ctx, 120*time.Second)

	// Add user message to history
	a.chat.AddMessage(ctx, "user", userMessage, modelID, agentCodename)
	a.telemetry.RecordChatMessage()

	runtime.EventsEmit(a.ctx, "chat:message", Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   userMessage,
		Timestamp: time.Now().Unix(),
	})

	runtime.EventsEmit(a.ctx, "chat:streamStart", nil)

	systemPrompt := fmt.Sprintf("You are %s, an elite AI agent. Respond helpfully and concisely.", agentCodename)

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

	go func() {
		defer cancel()
		tokenChan := make(chan string, 64)
		var fullResponse strings.Builder

		go func() {
			defer close(tokenChan)
			err := a.apiClient.ChatCompletionStream(ctx, chatReq, tokenChan)
			if err != nil {
				log.Printf("[Chat] Stream error: %v\n", err)
				runtime.EventsEmit(a.ctx, "chat:streamError", err.Error())
			}
		}()

		for token := range tokenChan {
			fullResponse.WriteString(token)
			runtime.EventsEmit(a.ctx, "chat:streamToken", token)
		}

		responseText := fullResponse.String()
		if responseText == "" {
			responseText = "[Offline Mode] Streaming not available. Start backend with: docker-compose up -d"
		}

		a.telemetry.RecordStreaming(float64(time.Since(streamStart).Milliseconds()))
		a.chat.AddMessage(a.ctx, "assistant", responseText, modelID, agentCodename)
		a.telemetry.RecordChatMessage()

		runtime.EventsEmit(a.ctx, "chat:streamEnd", Message{
			ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
			Role:      "assistant",
			Content:   responseText,
			Timestamp: time.Now().Unix(),
		})
	}()

	return nil
}

// CheckAPIHealth checks if the Ryzanstein API is reachable
func (a *App) CheckAPIHealth() (bool, error) {
	ctx, cancel := context.WithTimeout(a.ctx, 5*time.Second)
	defer cancel()
	return a.apiClient.Health(ctx)
}

// GetCircuitStatus returns real-time API health and circuit breaker state
func (a *App) GetCircuitStatus() map[string]interface{} {
	a.mu.RLock()
	running := a.isRunning
	a.mu.RUnlock()
	return map[string]interface{}{
		"api_running":       running,
		"api_base_url":      a.apiClient.GetBaseURL(),
		"max_retries":       3,
		"retry_delay_ms":    1000,
		"stream_buffer":     64,
		"context_timeout_s": 120,
		"timestamp":         time.Now().UTC().Format(time.RFC3339),
	}
}

// GetSystemHealth returns a comprehensive health snapshot for UI diagnostics.
// It aggregates API reachability, circuit status, timeout configuration,
// and runtime state into a single map suitable for the health panel.
func (a *App) GetSystemHealth() map[string]interface{} {
	a.mu.RLock()
	running := a.isRunning
	a.mu.RUnlock()

	// Probe API health with a short timeout
	healthCtx, cancel := context.WithTimeout(a.ctx, 5*time.Second)
	defer cancel()
	healthy, apiErr := a.apiClient.Health(healthCtx)
	apiReachable := apiErr == nil && healthy
	apiStatus := "healthy"
	if !apiReachable {
		if apiErr != nil {
			apiStatus = fmt.Sprintf("unreachable: %v", apiErr)
		} else {
			apiStatus = "unhealthy"
		}
	}

	return map[string]interface{}{
		// Runtime state
		"is_running": running,
		"api_status": apiStatus,

		// Connection info
		"api_base_url": a.apiClient.GetBaseURL(),

		// Timeout configuration (seconds)
		"timeouts": map[string]interface{}{
			"http_client_s":    150,
			"send_message_s":   30,
			"stream_s":         120,
			"health_check_s":   5,
			"model_load_s":     30,
			"batch_ms":         50,
			"pool_health_s":    30,
			"pool_idle_s":      300,
			"pool_max_conn_s":  600,
			"grpc_dial_s":      5,
			"cache_expiry_s":   300,
		},

		// Retry configuration
		"retry": map[string]interface{}{
			"max_retries":    3,
			"retry_delay_ms": 1000,
			"strategy":       "exponential_backoff",
		},

		// Streaming
		"stream_buffer_size": 64,

		// Timestamp
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}
}

// GetRecentLogs returns recent log entries for the UI diagnostics panel.
func (a *App) GetRecentLogs(n int) []map[string]interface{} {
	entries := a.logger.GetRecentEntries(n, services.LogLevelDebug)
	result := make([]map[string]interface{}, len(entries))
	for i, e := range entries {
		result[i] = map[string]interface{}{
			"level":     e.Level.String(),
			"message":   e.Message,
			"timestamp": e.Timestamp.UTC().Format("2006-01-02T15:04:05Z"),
			"source":    e.Component,
		}
	}
	return result
}

func (a *App) GetHistory(limit int) ([]Message, error) {
	log.Printf("[Chat] Fetching history (limit: %d)\n", limit)
	chatHistory := a.chat.GetHistory(limit)
	messages := make([]Message, len(chatHistory))
	for i, h := range chatHistory {
		messages[i] = Message{
			ID:        h.ID,
			Role:      h.Role,
			Content:   h.Content,
			Timestamp: h.Timestamp,
		}
	}
	return messages, nil
}

func (a *App) ClearHistory() error {
	log.Println("[Chat] Clearing history")
	a.chat.ClearHistory()
	runtime.EventsEmit(a.ctx, "chat:cleared", nil)
	return nil
}

// ============================================================================
// Model Service Methods
// ============================================================================

type ModelInfo struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Size          string `json:"size"`
	ContextLength int    `json:"contextLength"`
	Loaded        bool   `json:"loaded"`
	Status        string `json:"status"`
}

func (a *App) ListModels() ([]ModelInfo, error) {
	log.Println("[Models] Listing models")
	svcModels := a.models.ListModels()
	models := make([]ModelInfo, len(svcModels))
	for i, m := range svcModels {
		models[i] = ModelInfo{
			ID:            m.ID,
			Name:          m.Name,
			Size:          m.Size,
			ContextLength: m.ContextLength,
			Loaded:        m.Loaded,
			Status:        m.Status,
		}
	}
	return models, nil
}

func (a *App) LoadModel(modelID string) error {
	log.Printf("[Models] Loading model: %s\n", modelID)
	go func() {
		err := a.models.LoadModel(modelID)
		if err != nil {
			log.Printf("[Models] Error loading model: %v\n", err)
			runtime.EventsEmit(a.ctx, "model:loadError", map[string]interface{}{
				"modelID": modelID,
				"error":   err.Error(),
			})
			return
		}
		log.Printf("[Models] Successfully loaded: %s\n", modelID)
		runtime.EventsEmit(a.ctx, "model:loaded", map[string]interface{}{
			"modelID": modelID,
		})
	}()
	return nil
}

func (a *App) UnloadModel(modelID string) error {
	log.Printf("[Models] Unloading model: %s\n", modelID)
	return a.models.UnloadModel(modelID)
}

// ============================================================================
// Agent Service Methods
// ============================================================================

type AgentInfo struct {
	Codename       string   `json:"codename"`
	Name           string   `json:"name"`
	Tier           int      `json:"tier"`
	Philosophy     string   `json:"philosophy"`
	Capabilities   []string `json:"capabilities"`
	MasteryDomains []string `json:"masteryDomains"`
}

func (a *App) ListAgents() ([]string, error) {
	log.Println("[Agents] Listing agents")
	return a.agents.ListAgents(), nil
}

func (a *App) InvokeAgent(agentCodename string, toolName string, parameters map[string]interface{}) (interface{}, error) {
	log.Printf("[Agents] Invoking %s.%s\n", agentCodename, toolName)
	agentStart := time.Now()
	ctx, cancel := context.WithTimeout(a.ctx, 30*time.Second)
	defer cancel()
	result, err := a.agents.InvokeTool(ctx, agentCodename, toolName, parameters)
	a.telemetry.RecordAgentInvocation(float64(time.Since(agentStart).Milliseconds()))
	if err != nil {
		log.Printf("[Agents] Error invoking agent: %v\n", err)
		return nil, err
	}
	runtime.EventsEmit(a.ctx, "agent:invoked", map[string]interface{}{
		"agent":  agentCodename,
		"tool":   toolName,
		"result": result,
	})
	return result, nil
}

// InvokeAgentChat sends a chat message to an agent and returns the response
func (a *App) InvokeAgentChat(agentCodename string, message string) (string, error) {
	ctx, cancel := context.WithTimeout(a.ctx, 60*time.Second)
	defer cancel()

	response, err := a.agents.InvokeAgentChat(ctx, agentCodename, message)
	if err != nil {
		runtime.EventsEmit(a.ctx, "agent:chat:error", map[string]interface{}{
			"agent": agentCodename,
			"error": err.Error(),
		})
		return "", err
	}

	runtime.EventsEmit(a.ctx, "agent:chat:response", map[string]interface{}{
		"agent":   agentCodename,
		"message": message,
		"response": response,
	})
	return response, nil
}

// ============================================================================
// Config Service Methods
// ============================================================================

type ConfigData struct {
	Theme             string `json:"theme"`
	DefaultModel      string `json:"defaultModel"`
	DefaultAgent      string `json:"defaultAgent"`
	RyzansteinAPIURL  string `json:"ryzansteinApiUrl"`
	MCPServerURL      string `json:"mcpServerUrl"`
	AutoLoadLastModel bool   `json:"autoLoadLastModel"`
	EnableSystemTray  bool   `json:"enableSystemTray"`
	MinimizeToTray    bool   `json:"minimizeToTray"`
}

func (a *App) GetConfig() (ConfigData, error) {
	log.Println("[Config] Fetching configuration")
	svcCfg := a.config.GetConfig()
	return ConfigData{
		Theme:             svcCfg.Theme,
		DefaultModel:      svcCfg.DefaultModel,
		DefaultAgent:      svcCfg.DefaultAgent,
		RyzansteinAPIURL:  svcCfg.RyzansteinAPIURL,
		MCPServerURL:      svcCfg.MCPServerURL,
		AutoLoadLastModel: svcCfg.AutoLoadLastModel,
		EnableSystemTray:  svcCfg.EnableSystemTray,
		MinimizeToTray:    svcCfg.MinimizeToTray,
	}, nil
}

func (a *App) SaveConfig(cfg ConfigData) error {
	log.Println("[Config] Saving configuration")
	svcCfg := config.ConfigData{
		Theme:             cfg.Theme,
		DefaultModel:      cfg.DefaultModel,
		DefaultAgent:      cfg.DefaultAgent,
		RyzansteinAPIURL:  cfg.RyzansteinAPIURL,
		MCPServerURL:      cfg.MCPServerURL,
		AutoLoadLastModel: cfg.AutoLoadLastModel,
		EnableSystemTray:  cfg.EnableSystemTray,
		MinimizeToTray:    cfg.MinimizeToTray,
	}
	err := a.config.SaveConfig(svcCfg)
	if err == nil {
		runtime.EventsEmit(a.ctx, "config:saved", cfg)
	}
	return err
}

// ============================================================================
// System Methods
// ============================================================================

func (a *App) Greet(name string) string {
	log.Printf("[System] Greet: %s\n", name)
	return fmt.Sprintf("Hello %s, let's build amazing AI applications!", name)
}

func (a *App) GetVersion() string {
	return "1.0.0"
}

func (a *App) GetSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"arch": os.Getenv("PROCESSOR_ARCHITECTURE"),
	}
}

// GetTelemetrySnapshot returns a point-in-time snapshot of Desktop runtime
// telemetry: inference counts, error rates, latency histograms, and gauges.
// Mirrors the sigma-telemetry SpanRecord model from the Rust inference layer.
func (a *App) GetTelemetrySnapshot() map[string]interface{} {
	snap := a.telemetry.Snapshot()
	return map[string]interface{}{
		"timestamp":           snap.Timestamp,
		"inference_requests":  snap.InferenceRequests,
		"inference_errors":    snap.InferenceErrors,
		"chat_messages":       snap.ChatMessages,
		"agent_invocations":   snap.AgentInvocations,
		"streaming_requests":  snap.StreamingRequests,
		"latencies":           snap.Latencies,
		"gauges":              snap.Gauges,
	}
}

// ============================================================================
// IPC Server
// ============================================================================

func (a *App) startIPCServer() {
	err := a.ipc.Start()
	if err != nil {
		log.Printf("[IPC] Error starting server: %v\n", err)
		runtime.EventsEmit(a.ctx, "ipc:error", err.Error())
		return
	}
	log.Println("[IPC] Server started")
}

// ============================================================================
// Main Entry Point
// ============================================================================

func main() {
	app := NewApp()

	// If RYZANSTEIN_DEBUG_PORT is set, forward it as WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS
	// so Playwright's CDP harness can connect. This must be set before wails.Run.
	if port := os.Getenv("RYZANSTEIN_DEBUG_PORT"); port != "" {
		existing := os.Getenv("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS")
		debugFlag := fmt.Sprintf("--remote-debugging-port=%s", port)
		if existing != "" {
			os.Setenv("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", existing+" "+debugFlag)
		} else {
			os.Setenv("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", debugFlag)
		}
		log.Printf("[Desktop] CDP debug port enabled: %s", port)
	}

	// Create application with options
	err := wails.Run(&options.App{
		Title:      "Ryzanstein",
		Width:      1400,
		Height:     900,
		MinWidth:   800,
		MinHeight:  600,
		Assets:     assets,
		OnStartup:  app.Startup,
		OnShutdown: app.Shutdown,
		OnDomReady: app.onDomReady,
		Bind: []interface{}{
			app,
		},
	})

	if err != nil {
		log.Fatalf("Fatal error: %v", err)
	}
}

func (a *App) onDomReady(ctx context.Context) {
	log.Println("[Desktop] DOM ready")
	runtime.EventsEmit(ctx, "dom:ready", nil)
}
