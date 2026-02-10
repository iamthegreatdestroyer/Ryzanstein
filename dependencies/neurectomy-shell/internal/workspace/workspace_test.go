package workspace

import (
	"os"
	"testing"
)

func TestWorkspaceCreateAndGet(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "ws-test-*")
	defer os.RemoveAll(tmpDir)

	mgr := NewManager(tmpDir)

	ws, err := mgr.Create("my-project", false)
	if err != nil {
		t.Fatalf("create workspace: %v", err)
	}
	if ws.Name != "my-project" {
		t.Fatalf("expected name my-project, got %s", ws.Name)
	}
	if ws.Confidential {
		t.Fatal("expected non-confidential workspace")
	}

	got := mgr.Get(ws.ID)
	if got == nil {
		t.Fatal("workspace not found")
	}
	if got.ID != ws.ID {
		t.Fatalf("ID mismatch: %s vs %s", got.ID, ws.ID)
	}
}

func TestWorkspaceList(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "ws-test-*")
	defer os.RemoveAll(tmpDir)

	mgr := NewManager(tmpDir)
	mgr.Create("ws1", false)
	mgr.Create("ws2", true)

	list := mgr.List()
	if len(list) != 2 {
		t.Fatalf("expected 2 workspaces, got %d", len(list))
	}
}

func TestWorkspaceDelete(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "ws-test-*")
	defer os.RemoveAll(tmpDir)

	mgr := NewManager(tmpDir)
	ws, _ := mgr.Create("to-delete", false)

	err := mgr.Delete(ws.ID)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}

	got := mgr.Get(ws.ID)
	if got != nil {
		t.Fatal("workspace should be deleted")
	}
}

func TestWorkspaceDeleteNotFound(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "ws-test-*")
	defer os.RemoveAll(tmpDir)

	mgr := NewManager(tmpDir)
	err := mgr.Delete("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent workspace")
	}
}

func TestWorkspaceConfidential(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "ws-test-*")
	defer os.RemoveAll(tmpDir)

	mgr := NewManager(tmpDir)
	ws, err := mgr.Create("secret", true)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if !ws.Confidential {
		t.Fatal("expected confidential workspace")
	}
}
