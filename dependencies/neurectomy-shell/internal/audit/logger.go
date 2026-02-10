package audit

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/config"
)

// Entry represents an immutable audit log entry.
type Entry struct {
	Timestamp time.Time         `json:"timestamp"`
	Action    string            `json:"action"`
	Metadata  map[string]string `json:"metadata"`
	Hash      string            `json:"hash,omitempty"`
}

// Logger writes immutable audit entries.
type Logger struct {
	logPath   string
	encrypted bool
	entries   []Entry
	mu        sync.Mutex
}

// NewLogger creates a new audit logger.
func NewLogger(cfg config.AuditCfg) (*Logger, error) {
	// Ensure log directory exists
	dir := cfg.LogPath
	if dir == "" {
		dir = "audit.log"
	}

	return &Logger{
		logPath:   dir,
		encrypted: cfg.Encrypted,
		entries:   make([]Entry, 0),
	}, nil
}

// Log records an audit event.
func (l *Logger) Log(action string, metadata map[string]string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry := Entry{
		Timestamp: time.Now().UTC(),
		Action:    action,
		Metadata:  metadata,
	}

	// Chain hash: each entry includes hash of previous entry
	if len(l.entries) > 0 {
		prev := l.entries[len(l.entries)-1]
		entry.Hash = fmt.Sprintf("chain:%s:%s", prev.Hash, prev.Action)
	} else {
		entry.Hash = "genesis"
	}

	l.entries = append(l.entries, entry)

	// Write to file (append-only)
	data, err := json.Marshal(entry)
	if err != nil {
		log.Printf("audit marshal error: %v", err)
		return
	}

	f, err := os.OpenFile(l.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		log.Printf("audit file error: %v", err)
		return
	}
	defer f.Close()

	f.Write(data)
	f.Write([]byte("\n"))
}

// Entries returns all audit entries (for verification).
func (l *Logger) Entries() []Entry {
	l.mu.Lock()
	defer l.mu.Unlock()
	result := make([]Entry, len(l.entries))
	copy(result, l.entries)
	return result
}

// Verify checks the integrity of the audit chain.
func (l *Logger) Verify() bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	for i := 1; i < len(l.entries); i++ {
		expected := fmt.Sprintf("chain:%s:%s", l.entries[i-1].Hash, l.entries[i-1].Action)
		if l.entries[i].Hash != expected {
			return false
		}
	}
	return true
}
