package tee

import (
	"testing"

	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/config"
)

func TestNewManagerSimulate(t *testing.T) {
	mgr, err := NewManager(config.TEECfg{Provider: "simulate"})
	if err != nil {
		t.Fatalf("create manager: %v", err)
	}
	if mgr == nil {
		t.Fatal("manager is nil")
	}
}

func TestNewManagerUnsupported(t *testing.T) {
	_, err := NewManager(config.TEECfg{Provider: "magic"})
	if err == nil {
		t.Fatal("expected error for unsupported provider")
	}
}

func TestProvisionAndAttest(t *testing.T) {
	mgr, _ := NewManager(config.TEECfg{Provider: "simulate"})

	vm, err := mgr.Provision("ws-123")
	if err != nil {
		t.Fatalf("provision: %v", err)
	}
	if !vm.Running {
		t.Fatal("VM should be running")
	}

	report, err := mgr.Attest("ws-123")
	if err != nil {
		t.Fatalf("attest: %v", err)
	}
	if report == "" {
		t.Fatal("empty attestation report")
	}
}

func TestAttestNoVM(t *testing.T) {
	mgr, _ := NewManager(config.TEECfg{Provider: "simulate"})
	_, err := mgr.Attest("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent VM")
	}
}

func TestTerminate(t *testing.T) {
	mgr, _ := NewManager(config.TEECfg{Provider: "simulate"})
	mgr.Provision("ws-456")

	err := mgr.Terminate("ws-456")
	if err != nil {
		t.Fatalf("terminate: %v", err)
	}

	_, err = mgr.Attest("ws-456")
	if err == nil {
		t.Fatal("expected error after termination")
	}
}
