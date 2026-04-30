package ipc

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/agents"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/client"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/models"
)

// IPCCommand represents an incoming JSON command from VS Code
type IPCCommand struct {
	Command string                 `json:"command"`
	Params  map[string]interface{} `json:"params,omitempty"`
}

// Server handles IPC communication between desktop and VS Code
type Server struct {
	listener  net.Listener
	clients   map[string]net.Conn
	mu        sync.RWMutex
	isRunning bool
	agents    *agents.Service
	models    *models.Service
	apiClient *client.RyzansteinClient
}

// NewServer creates a new IPC server with injected services
func NewServer(agentsSvc *agents.Service, modelsSvc *models.Service, apiClient *client.RyzansteinClient) *Server {
	return &Server{
		clients:   make(map[string]net.Conn),
		agents:    agentsSvc,
		models:    modelsSvc,
		apiClient: apiClient,
	}
}

// Start starts the IPC server
func (s *Server) Start() error {
	log.Println("[IPC] Starting server on localhost:9001")

	listener, err := net.Listen("tcp", "localhost:9001")
	if err != nil {
		return fmt.Errorf("failed to start IPC server: %w", err)
	}

	s.listener = listener
	s.isRunning = true

	// Accept connections
	go s.acceptConnections()

	return nil
}

// acceptConnections accepts incoming client connections
func (s *Server) acceptConnections() {
	for s.isRunning {
		conn, err := s.listener.Accept()
		if err != nil {
			if s.isRunning {
				log.Printf("[IPC] Accept error: %v\n", err)
			}
			continue
		}

		clientID := conn.RemoteAddr().String()
		s.mu.Lock()
		s.clients[clientID] = conn
		s.mu.Unlock()

		log.Printf("[IPC] Client connected: %s\n", clientID)

		// Handle client in goroutine
		go s.handleClient(clientID, conn)
	}
}

// handleClient handles a client connection with JSON command dispatch
func (s *Server) handleClient(clientID string, conn net.Conn) {
	defer func() {
		conn.Close()
		s.mu.Lock()
		delete(s.clients, clientID)
		s.mu.Unlock()
		log.Printf("[IPC] Client disconnected: %s\n", clientID)
	}()

	conn.SetDeadline(time.Now().Add(30 * time.Second)) //nolint:errcheck
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	for scanner.Scan() {
		// D-3: reset read deadline on each message
		conn.SetDeadline(time.Now().Add(30 * time.Second)) //nolint:errcheck

		line := scanner.Text()

		var cmd IPCCommand
		if err := json.Unmarshal([]byte(line), &cmd); err != nil {
			s.sendResponse(conn, IPCResponse{Status: "error", Error: fmt.Sprintf("invalid JSON: %v", err)})
			continue
		}

		// I-1: log command type only, never payload content
		log.Printf("[IPC] cmd=%s client=%s\n", cmd.Command, clientID)

		resp := s.dispatchCommand(cmd)
		s.sendResponse(conn, resp)
	}
}

// dispatchCommand routes an IPCCommand to the appropriate handler
func (s *Server) dispatchCommand(cmd IPCCommand) IPCResponse {

	switch cmd.Command {
	case "health":
		return IPCResponse{Status: "ok", Data: map[string]interface{}{"status": "ok", "version": "1.0.0"}}

	case "infer":
		return s.handleInfer(cmd.Params)

	case "list_agents":
		return s.handleListAgents()

	case "list_models":
		return s.handleListModels()

	default:
		return IPCResponse{Status: "error", Error: fmt.Sprintf("unknown command: %s", cmd.Command)}
	}
}

// handleInfer sends an inference request via the Ryzanstein API client
func (s *Server) handleInfer(params map[string]interface{}) IPCResponse {
	prompt, _ := params["prompt"].(string)
	if prompt == "" {
		return IPCResponse{Status: "error", Error: "missing required param: prompt"}
	}

	model, _ := params["model"].(string)
	if model == "" {
		model = "default"
	}

	req := &client.InferenceRequest{
		Prompt: prompt,
		Model:  model,
	}

	resp, err := s.apiClient.Infer(context.Background(), req)
	if err != nil {
		return IPCResponse{Status: "error", Error: fmt.Sprintf("inference failed: %v", err)}
	}

	responseText := ""
	if len(resp.Choices) > 0 {
		responseText = resp.Choices[0].Text
	}

	return IPCResponse{Status: "ok", Data: map[string]interface{}{
		"response": responseText,
		"model":    resp.Model,
		"usage": map[string]interface{}{
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
			"total_tokens":      resp.Usage.TotalTokens,
		},
	}}
}

// handleListAgents returns registered agents
func (s *Server) handleListAgents() IPCResponse {
	agentList := s.agents.ListAgents()
	return IPCResponse{Status: "ok", Data: agentList}
}

// handleListModels returns available models
func (s *Server) handleListModels() IPCResponse {
	modelList := s.models.ListModels()
	return IPCResponse{Status: "ok", Data: modelList}
}

// sendResponse marshals and writes a response to the connection
func (s *Server) sendResponse(conn net.Conn, resp IPCResponse) {
	data, err := json.Marshal(resp)
	if err != nil {
		log.Printf("[IPC] Error marshaling response: %v\n", err)
		return
	}
	data = append(data, '\n')
	if _, err := conn.Write(data); err != nil {
		log.Printf("[IPC] Error writing response: %v\n", err)
	}
}

// Broadcast sends message to all connected clients
func (s *Server) Broadcast(message string) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for clientID, conn := range s.clients {
		if _, err := conn.Write([]byte(message)); err != nil {
			log.Printf("[IPC] Error sending to %s: %v\n", clientID, err)
		}
	}
}

// SendToClient sends message to specific client
func (s *Server) SendToClient(clientID string, message string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	conn, ok := s.clients[clientID]
	if !ok {
		return fmt.Errorf("client not found: %s", clientID)
	}

	_, err := conn.Write([]byte(message))
	return err
}

// Close closes the IPC server
func (s *Server) Close() error {
	log.Println("[IPC] Closing server")
	s.isRunning = false

	// Close all client connections
	s.mu.Lock()
	for _, conn := range s.clients {
		conn.Close()
	}
	s.clients = make(map[string]net.Conn)
	s.mu.Unlock()

	// Close listener
	if s.listener != nil {
		return s.listener.Close()
	}

	return nil
}

// GetConnectedClients returns count of connected clients
func (s *Server) GetConnectedClients() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.clients)
}
