package ipc

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/agents"
)

// MessageType represents the type of IPC message action.
type MessageType string

const (
	MessageTypeInference  MessageType = "inference"
	MessageTypeListModels MessageType = "list_models"
	MessageTypeLoadModel  MessageType = "load_model"
	MessageTypePing       MessageType = "ping"
	MessageTypeInvokeTool MessageType = "invoke_tool"
	MessageTypeListAgents MessageType = "list_agents"
	MessageTypeGetAgent   MessageType = "get_agent"
)

// IPCRequest represents an incoming JSON message from a client.
type IPCRequest struct {
	ID      string          `json:"id"`
	Action  MessageType     `json:"action"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

// IPCResponse represents an outgoing JSON response to a client.
type IPCResponse struct {
	ID     string      `json:"id"`
	Action MessageType `json:"action"`
	Status string      `json:"status"` // "ok" or "error"
	Data   interface{} `json:"data,omitempty"`
	Error  string      `json:"error,omitempty"`
}

// InferencePayload is the payload for inference requests.
type InferencePayload struct {
	Prompt  string `json:"prompt"`
	ModelID string `json:"model_id"`
}

// LoadModelPayload is the payload for load_model requests.
type LoadModelPayload struct {
	ModelID string `json:"model_id"`
}

// InvokeToolPayload is the payload for invoke_tool requests.
type InvokeToolPayload struct {
	AgentCodename string                 `json:"agent_codename"`
	ToolName      string                 `json:"tool_name"`
	Params        map[string]interface{} `json:"params,omitempty"`
}

// GetAgentPayload is the payload for get_agent requests.
type GetAgentPayload struct {
	Codename string `json:"codename"`
}

// Router dispatches incoming IPC messages to the appropriate Bridge methods.
type Router struct {
	bridge       *Bridge
	server       *Server
	agentService *agents.Service
}

// NewRouter creates a Router wired to the given Bridge, Server, and agent Service.
func NewRouter(bridge *Bridge, server *Server, agentSvc *agents.Service) *Router {
	return &Router{
		bridge:       bridge,
		server:       server,
		agentService: agentSvc,
	}
}

// HandleMessage parses a raw JSON message and routes it to the correct handler.
// Returns the JSON-encoded response bytes.
func (r *Router) HandleMessage(clientID string, raw []byte) []byte {
	var req IPCRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return r.errorResponse("", "", fmt.Errorf("invalid JSON: %w", err))
	}

	log.Printf("[Router] action=%s id=%s client=%s", req.Action, req.ID, clientID)

	var resp *IPCResponse
	switch req.Action {
	case MessageTypeInference:
		resp = r.handleInference(req)
	case MessageTypeListModels:
		resp = r.handleListModels(req)
	case MessageTypeLoadModel:
		resp = r.handleLoadModel(req)
	case MessageTypePing:
		resp = r.handlePing(req)
	case MessageTypeInvokeTool:
		resp = r.handleInvokeTool(req)
	case MessageTypeListAgents:
		resp = r.handleListAgents(req)
	case MessageTypeGetAgent:
		resp = r.handleGetAgent(req)
	default:
		resp = &IPCResponse{
			ID:     req.ID,
			Action: req.Action,
			Status: "error",
			Error:  fmt.Sprintf("unknown action: %s", req.Action),
		}
	}

	out, err := json.Marshal(resp)
	if err != nil {
		log.Printf("[Router] marshal failure: %v", err)
		return []byte(`{"status":"error","error":"internal marshal failure"}`)
	}
	return out
}

func (r *Router) handleInference(req IPCRequest) *IPCResponse {
	var payload InferencePayload
	if err := json.Unmarshal(req.Payload, &payload); err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: fmt.Sprintf("invalid inference payload: %v", err),
		}
	}

	if payload.Prompt == "" {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: "prompt is required",
		}
	}

	result, err := r.bridge.RunInference(payload.Prompt, payload.ModelID)
	if err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: err.Error(),
		}
	}

	return &IPCResponse{
		ID: req.ID, Action: req.Action, Status: "ok",
		Data: result,
	}
}

func (r *Router) handleListModels(req IPCRequest) *IPCResponse {
	models, err := r.bridge.ListModels()
	if err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: err.Error(),
		}
	}

	return &IPCResponse{
		ID: req.ID, Action: req.Action, Status: "ok",
		Data: models,
	}
}

func (r *Router) handleLoadModel(req IPCRequest) *IPCResponse {
	var payload LoadModelPayload
	if err := json.Unmarshal(req.Payload, &payload); err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: fmt.Sprintf("invalid load_model payload: %v", err),
		}
	}

	if payload.ModelID == "" {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: "model_id is required",
		}
	}

	if err := r.bridge.LoadModel(payload.ModelID); err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: err.Error(),
		}
	}

	return &IPCResponse{
		ID: req.ID, Action: req.Action, Status: "ok",
		Data: map[string]string{"model_id": payload.ModelID, "status": "loaded"},
	}
}

func (r *Router) handlePing(req IPCRequest) *IPCResponse {
	return &IPCResponse{
		ID: req.ID, Action: req.Action, Status: "ok",
		Data: map[string]string{"message": "pong"},
	}
}

func (r *Router) handleInvokeTool(req IPCRequest) *IPCResponse {
	var payload InvokeToolPayload
	if err := json.Unmarshal(req.Payload, &payload); err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: fmt.Sprintf("invalid invoke_tool payload: %v", err),
		}
	}

	if payload.AgentCodename == "" || payload.ToolName == "" {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: "agent_codename and tool_name are required",
		}
	}

	result, err := r.agentService.InvokeTool(
		context.Background(), payload.AgentCodename, payload.ToolName, payload.Params,
	)
	if err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: err.Error(),
		}
	}

	return &IPCResponse{
		ID: req.ID, Action: req.Action, Status: "ok",
		Data: result,
	}
}

func (r *Router) handleListAgents(req IPCRequest) *IPCResponse {
	agentList := r.agentService.ListAgents()
	return &IPCResponse{
		ID: req.ID, Action: req.Action, Status: "ok",
		Data: agentList,
	}
}

func (r *Router) handleGetAgent(req IPCRequest) *IPCResponse {
	var payload GetAgentPayload
	if err := json.Unmarshal(req.Payload, &payload); err != nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: fmt.Sprintf("invalid get_agent payload: %v", err),
		}
	}

	if payload.Codename == "" {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: "codename is required",
		}
	}

	agent := r.agentService.GetAgent(payload.Codename)
	if agent == nil {
		return &IPCResponse{
			ID: req.ID, Action: req.Action, Status: "error",
			Error: fmt.Sprintf("agent not found: %s", payload.Codename),
		}
	}

	return &IPCResponse{
		ID: req.ID, Action: req.Action, Status: "ok",
		Data: agent,
	}
}

// RouteAndRespond handles a message and sends the response back to the client
// via the IPC Server.
func (r *Router) RouteAndRespond(clientID string, raw []byte) {
	response := r.HandleMessage(clientID, raw)
	if err := r.server.SendToClient(clientID, string(response)); err != nil {
		log.Printf("[Router] send to %s failed: %v", clientID, err)
	}
}

func (r *Router) errorResponse(id string, action string, err error) []byte {
	resp := &IPCResponse{
		ID:     id,
		Action: MessageType(action),
		Status: "error",
		Error:  err.Error(),
	}
	out, _ := json.Marshal(resp)
	return out
}
