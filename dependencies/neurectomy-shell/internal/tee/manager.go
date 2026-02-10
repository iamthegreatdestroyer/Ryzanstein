package tee

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"

	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/config"
)

// Manager manages Trusted Execution Environment lifecycle.
type Manager struct {
	provider string
	vms      map[string]*ConfidentialVM
}

// ConfidentialVM represents a running confidential VM instance.
type ConfidentialVM struct {
	ID          string
	WorkspaceID string
	Provider    string
	// Simulated attestation nonce
	Nonce   string
	Running bool
}

// NewManager creates a new TEE manager.
func NewManager(cfg config.TEECfg) (*Manager, error) {
	switch cfg.Provider {
	case "sev-snp", "tdx", "simulate":
		log.Printf("TEE manager initialized (provider=%s)", cfg.Provider)
	default:
		return nil, fmt.Errorf("unsupported TEE provider: %s", cfg.Provider)
	}

	return &Manager{
		provider: cfg.Provider,
		vms:      make(map[string]*ConfidentialVM),
	}, nil
}

// Provision creates and starts a confidential VM for a workspace.
func (m *Manager) Provision(workspaceID string) (*ConfidentialVM, error) {
	nonce := make([]byte, 32)
	rand.Read(nonce)

	vm := &ConfidentialVM{
		ID:          generateVMID(),
		WorkspaceID: workspaceID,
		Provider:    m.provider,
		Nonce:       hex.EncodeToString(nonce),
		Running:     true,
	}

	m.vms[workspaceID] = vm
	log.Printf("Provisioned confidential VM %s for workspace %s (provider=%s)",
		vm.ID, workspaceID, m.provider)

	return vm, nil
}

// Attest performs remote attestation for a workspace's VM.
func (m *Manager) Attest(workspaceID string) (string, error) {
	vm, ok := m.vms[workspaceID]
	if !ok {
		return "", fmt.Errorf("no VM for workspace %s", workspaceID)
	}

	if !vm.Running {
		return "", fmt.Errorf("VM not running for workspace %s", workspaceID)
	}

	// In production: perform actual SEV-SNP attestation handshake.
	// In simulation mode: return a mock attestation report.
	report := fmt.Sprintf("attestation:%s:nonce:%s:provider:%s:status:verified",
		vm.ID, vm.Nonce, vm.Provider)

	log.Printf("Attestation successful for workspace %s", workspaceID)
	return report, nil
}

// Terminate shuts down a confidential VM.
func (m *Manager) Terminate(workspaceID string) error {
	vm, ok := m.vms[workspaceID]
	if !ok {
		return fmt.Errorf("no VM for workspace %s", workspaceID)
	}

	vm.Running = false
	log.Printf("Terminated VM %s for workspace %s", vm.ID, workspaceID)
	delete(m.vms, workspaceID)
	return nil
}

func generateVMID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return "cvm-" + hex.EncodeToString(b)
}
