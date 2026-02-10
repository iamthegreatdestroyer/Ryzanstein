package server

import (
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/audit"
	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/config"
	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/tee"
	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/workspace"
)

// Server is the neurectomy-shell orchestrator.
type Server struct {
	cfg        *config.Config
	workspaces *workspace.Manager
	teeManager *tee.Manager
	auditor    *audit.Logger
	mu         sync.RWMutex
}

// New creates a new Server.
func New(cfg *config.Config) (*Server, error) {
	ws := workspace.NewManager(cfg.DataDir)

	teeMgr, err := tee.NewManager(cfg.TEE)
	if err != nil {
		return nil, fmt.Errorf("tee manager init: %w", err)
	}

	auditor, err := audit.NewLogger(cfg.Audit)
	if err != nil {
		return nil, fmt.Errorf("audit logger init: %w", err)
	}

	return &Server{
		cfg:        cfg,
		workspaces: ws,
		teeManager: teeMgr,
		auditor:    auditor,
	}, nil
}

// Run starts the HTTP server.
func (s *Server) Run() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/workspace/create", s.handleCreateWorkspace)
	mux.HandleFunc("/api/v1/workspace/list", s.handleListWorkspaces)
	mux.HandleFunc("/api/v1/workspace/connect", s.handleConnect)
	mux.HandleFunc("/api/v1/attest", s.handleAttest)
	mux.HandleFunc("/health", s.handleHealth)

	addr := fmt.Sprintf(":%d", s.cfg.Port)
	return http.ListenAndServe(addr, mux)
}

func (s *Server) handleCreateWorkspace(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	name := r.URL.Query().Get("name")
	confidential := r.URL.Query().Get("confidential") == "true"

	s.auditor.Log("workspace.create", map[string]string{
		"name":         name,
		"confidential": fmt.Sprintf("%t", confidential),
	})

	ws, err := s.workspaces.Create(name, confidential)
	if err != nil {
		log.Printf("workspace create error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if confidential {
		if _, err := s.teeManager.Provision(ws.ID); err != nil {
			log.Printf("TEE provision error: %v", err)
			http.Error(w, "TEE provisioning failed", http.StatusInternalServerError)
			return
		}
	}

	fmt.Fprintf(w, `{"id": "%s", "name": "%s", "confidential": %t}`, ws.ID, ws.Name, ws.Confidential)
}

func (s *Server) handleListWorkspaces(w http.ResponseWriter, r *http.Request) {
	workspaces := s.workspaces.List()
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"workspaces": [`)
	for i, ws := range workspaces {
		if i > 0 {
			fmt.Fprint(w, ",")
		}
		fmt.Fprintf(w, `{"id":"%s","name":"%s","confidential":%t}`, ws.ID, ws.Name, ws.Confidential)
	}
	fmt.Fprintf(w, `]}`)
}

func (s *Server) handleConnect(w http.ResponseWriter, r *http.Request) {
	wsID := r.URL.Query().Get("id")
	ws := s.workspaces.Get(wsID)
	if ws == nil {
		http.Error(w, "Workspace not found", http.StatusNotFound)
		return
	}

	s.auditor.Log("workspace.connect", map[string]string{"id": wsID})

	if ws.Confidential {
		report, err := s.teeManager.Attest(wsID)
		if err != nil {
			http.Error(w, "Attestation failed", http.StatusForbidden)
			return
		}
		fmt.Fprintf(w, `{"connected": true, "attestation": "%s"}`, report)
		return
	}

	fmt.Fprintf(w, `{"connected": true}`)
}

func (s *Server) handleAttest(w http.ResponseWriter, r *http.Request) {
	wsID := r.URL.Query().Get("id")
	report, err := s.teeManager.Attest(wsID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	fmt.Fprintf(w, `{"attestation_report": "%s"}`, report)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	fmt.Fprint(w, `{"status": "ok"}`)
}
