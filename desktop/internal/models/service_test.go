package models

import (
	"testing"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
)

func newTestService(t *testing.T) *Service {
	t.Helper()
	mgr, err := config.NewManager()
	if err != nil {
		t.Fatalf("config.NewManager: %v", err)
	}
	svc := NewService(mgr)
	svc.LoadInstalledModels()
	return svc
}

func TestLoadInstalledModels_PopulatesModels(t *testing.T) {
	svc := newTestService(t)

	models := svc.ListModels()
	if len(models) == 0 {
		t.Fatal("expected models to be populated after LoadInstalledModels")
	}
}

func TestListModels_AllHaveIDs(t *testing.T) {
	svc := newTestService(t)

	for _, m := range svc.ListModels() {
		if m.ID == "" {
			t.Error("model has empty ID")
		}
		if m.Name == "" {
			t.Errorf("model %q has empty Name", m.ID)
		}
	}
}

func TestLoadModel_MarksLoaded(t *testing.T) {
	svc := newTestService(t)

	if err := svc.LoadModel("ryzanstein-7b"); err != nil {
		t.Fatalf("LoadModel error: %v", err)
	}

	loaded := svc.GetLoadedModel()
	if loaded == nil {
		t.Fatal("GetLoadedModel returned nil after loading")
	}
	if loaded.ID != "ryzanstein-7b" {
		t.Errorf("expected 'ryzanstein-7b', got %q", loaded.ID)
	}
	if !loaded.Loaded {
		t.Error("expected Loaded=true")
	}
	if loaded.Status != "loaded" {
		t.Errorf("expected Status='loaded', got %q", loaded.Status)
	}
}

func TestLoadModel_UnknownModel(t *testing.T) {
	svc := newTestService(t)

	err := svc.LoadModel("unknown-model-xyz")
	if err == nil {
		t.Error("expected error for unknown model ID")
	}
}

func TestLoadModel_SwitchesActiveModel(t *testing.T) {
	svc := newTestService(t)

	if err := svc.LoadModel("ryzanstein-7b"); err != nil {
		t.Fatalf("LoadModel 7b: %v", err)
	}
	if err := svc.LoadModel("ryzanstein-13b"); err != nil {
		t.Fatalf("LoadModel 13b: %v", err)
	}

	loaded := svc.GetLoadedModel()
	if loaded == nil {
		t.Fatal("GetLoadedModel returned nil")
	}
	if loaded.ID != "ryzanstein-13b" {
		t.Errorf("expected 13b to be active, got %q", loaded.ID)
	}

	// Verify 7b is no longer flagged as loaded in the list.
	for _, m := range svc.ListModels() {
		if m.ID == "ryzanstein-7b" && m.Loaded {
			t.Error("ryzanstein-7b should be unloaded after switching to 13b")
		}
	}
}

func TestUnloadModel_ClearsLoaded(t *testing.T) {
	svc := newTestService(t)

	if err := svc.LoadModel("ryzanstein-7b"); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	if err := svc.UnloadModel("ryzanstein-7b"); err != nil {
		t.Fatalf("UnloadModel: %v", err)
	}

	if svc.GetLoadedModel() != nil {
		t.Error("expected no loaded model after UnloadModel")
	}

	// Status should revert to "ready".
	for _, m := range svc.ListModels() {
		if m.ID == "ryzanstein-7b" {
			if m.Status != "ready" {
				t.Errorf("expected Status='ready' after unload, got %q", m.Status)
			}
			if m.Loaded {
				t.Error("expected Loaded=false after unload")
			}
		}
	}
}

func TestGetLoadedModel_NoneLoaded(t *testing.T) {
	svc := newTestService(t)

	if svc.GetLoadedModel() != nil {
		t.Error("expected nil when no model is loaded")
	}
}

func TestUnloadModel_UnknownModelIsNoop(t *testing.T) {
	svc := newTestService(t)

	// Should not error — unloading a non-existent model is a no-op.
	if err := svc.UnloadModel("does-not-exist"); err != nil {
		t.Errorf("unexpected error for unknown model unload: %v", err)
	}
}
