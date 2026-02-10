package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/config"
)

func newTestServer(t *testing.T) *Server {
	t.Helper()

	tmpDir, err := os.MkdirTemp("", "neurectomy-test-*")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(tmpDir) })

	cfg := &config.Config{
		Port:    0,
		DataDir: tmpDir,
		Vault:   config.VaultCfg{Enabled: false},
		TEE:     config.TEECfg{Provider: "simulate"},
		Audit:   config.AuditCfg{LogPath: tmpDir + "/audit.log", Encrypted: false},
	}

	srv, err := New(cfg)
	if err != nil {
		t.Fatalf("create server: %v", err)
	}
	return srv
}

func TestHealthEndpoint(t *testing.T) {
	srv := newTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	srv.handleHealth(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	body, _ := io.ReadAll(rec.Body)
	if string(body) != `{"status": "ok"}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

func TestCreateWorkspace(t *testing.T) {
	srv := newTestServer(t)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workspace/create?name=test-ws&confidential=false", nil)
	rec := httptest.NewRecorder()
	srv.handleCreateWorkspace(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestCreateConfidentialWorkspace(t *testing.T) {
	srv := newTestServer(t)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workspace/create?name=secret-ws&confidential=true", nil)
	rec := httptest.NewRecorder()
	srv.handleCreateWorkspace(rec, req)

	if rec.Code != http.StatusOK {
		body, _ := io.ReadAll(rec.Body)
		t.Fatalf("expected 200, got %d: %s", rec.Code, body)
	}
}

func TestListWorkspaces(t *testing.T) {
	srv := newTestServer(t)

	// Create a workspace first
	req1 := httptest.NewRequest(http.MethodPost, "/api/v1/workspace/create?name=ws1&confidential=false", nil)
	rec1 := httptest.NewRecorder()
	srv.handleCreateWorkspace(rec1, req1)

	// List workspaces
	req2 := httptest.NewRequest(http.MethodGet, "/api/v1/workspace/list", nil)
	rec2 := httptest.NewRecorder()
	srv.handleListWorkspaces(rec2, req2)

	if rec2.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec2.Code)
	}
}

func TestCreateWorkspaceMethodNotAllowed(t *testing.T) {
	srv := newTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/workspace/create?name=test", nil)
	rec := httptest.NewRecorder()
	srv.handleCreateWorkspace(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rec.Code)
	}
}

func TestConnectNonExistentWorkspace(t *testing.T) {
	srv := newTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/workspace/connect?id=nonexistent", nil)
	rec := httptest.NewRecorder()
	srv.handleConnect(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rec.Code)
	}
}
