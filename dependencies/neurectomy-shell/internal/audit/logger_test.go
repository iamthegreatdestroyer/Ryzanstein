package audit

import (
	"os"
	"testing"

	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/config"
)

func newTestLogger(t *testing.T) *Logger {
	t.Helper()
	tmpFile, err := os.CreateTemp("", "audit-test-*.log")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	tmpFile.Close()
	t.Cleanup(func() { os.Remove(tmpFile.Name()) })

	logger, err := NewLogger(config.AuditCfg{
		LogPath:   tmpFile.Name(),
		Encrypted: false,
	})
	if err != nil {
		t.Fatalf("create logger: %v", err)
	}
	return logger
}

func TestLogAndRetrieve(t *testing.T) {
	logger := newTestLogger(t)

	logger.Log("test.action", map[string]string{"key": "value"})
	logger.Log("test.action2", nil)

	entries := logger.Entries()
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if entries[0].Action != "test.action" {
		t.Fatalf("unexpected action: %s", entries[0].Action)
	}
}

func TestChainIntegrity(t *testing.T) {
	logger := newTestLogger(t)

	logger.Log("action1", nil)
	logger.Log("action2", nil)
	logger.Log("action3", nil)

	if !logger.Verify() {
		t.Fatal("audit chain verification failed")
	}
}

func TestGenesisHash(t *testing.T) {
	logger := newTestLogger(t)
	logger.Log("first", nil)

	entries := logger.Entries()
	if entries[0].Hash != "genesis" {
		t.Fatalf("expected genesis hash, got %s", entries[0].Hash)
	}
}
