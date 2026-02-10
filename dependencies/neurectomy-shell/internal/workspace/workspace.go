package workspace

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Workspace represents a confidential dev environment.
type Workspace struct {
	ID           string
	Name         string
	Path         string
	Confidential bool
}

// Manager handles workspace lifecycle.
type Manager struct {
	baseDir    string
	workspaces map[string]*Workspace
	mu         sync.RWMutex
}

// NewManager creates a new workspace manager.
func NewManager(baseDir string) *Manager {
	return &Manager{
		baseDir:    baseDir,
		workspaces: make(map[string]*Workspace),
	}
}

// Create provisions a new workspace.
func (m *Manager) Create(name string, confidential bool) (*Workspace, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := generateID()
	wsPath := filepath.Join(m.baseDir, id)

	if err := os.MkdirAll(wsPath, 0o700); err != nil {
		return nil, fmt.Errorf("create workspace dir: %w", err)
	}

	ws := &Workspace{
		ID:           id,
		Name:         name,
		Path:         wsPath,
		Confidential: confidential,
	}

	m.workspaces[id] = ws
	return ws, nil
}

// Get returns a workspace by ID.
func (m *Manager) Get(id string) *Workspace {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.workspaces[id]
}

// List returns all workspaces.
func (m *Manager) List() []*Workspace {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*Workspace, 0, len(m.workspaces))
	for _, ws := range m.workspaces {
		result = append(result, ws)
	}
	return result
}

// Delete removes a workspace.
func (m *Manager) Delete(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	ws, ok := m.workspaces[id]
	if !ok {
		return fmt.Errorf("workspace %s not found", id)
	}

	if err := os.RemoveAll(ws.Path); err != nil {
		return fmt.Errorf("remove workspace: %w", err)
	}

	delete(m.workspaces, id)
	return nil
}

func generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}
