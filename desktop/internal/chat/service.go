package chat

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// Service handles chat operations
type Service struct {
	history []Message
	mu      sync.RWMutex
}

// Message represents a chat message
type Message struct {
	ID        string
	Role      string // "user" or "assistant"
	Content   string
	Timestamp int64
	Metadata  map[string]interface{}
}

// NewService creates a new chat service
func NewService() *Service {
	return &Service{
		history: make([]Message, 0),
	}
}

// SendMessage sends a message and returns response
func (s *Service) SendMessage(ctx context.Context, message string, modelID string, agentCodename string) (string, error) {
	log.Printf("[ChatService] Processing message (model: %s, agent: %s)\n", modelID, agentCodename)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Add user message to history
	userMsg := Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   message,
		Timestamp: time.Now().Unix(),
		Metadata: map[string]interface{}{
			"model": modelID,
			"agent": agentCodename,
		},
	}
	s.history = append(s.history, userMsg)

	// Simulate inference (will be replaced with actual MCP call)
	response := fmt.Sprintf("Response from %s (model: %s): Processed your request about '%s'",
		agentCodename, modelID, message)

	// Add assistant message to history
	assistantMsg := Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "assistant",
		Content:   response,
		Timestamp: time.Now().Unix(),
		Metadata: map[string]interface{}{
			"agent": agentCodename,
			"model": modelID,
		},
	}
	s.history = append(s.history, assistantMsg)

	return response, nil
}

// GetHistory returns chat history
func (s *Service) GetHistory(limit int) []Message {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 || limit > len(s.history) {
		return s.history
	}

	return s.history[len(s.history)-limit:]
}

// ClearHistory clears chat history
func (s *Service) ClearHistory() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history = make([]Message, 0)
}

// Close closes the service
func (s *Service) Close() {
	log.Println("[ChatService] Closing")
	s.ClearHistory()
}
