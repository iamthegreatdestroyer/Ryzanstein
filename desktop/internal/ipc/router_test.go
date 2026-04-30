package ipc

import (
	"encoding/json"
	"testing"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/agents"
)

// newTestRouter builds a Router with a nil bridge/server (sufficient for
// pure routing logic that does not touch the network).
func newTestRouter() *Router {
	agentSvc := agents.NewService(nil)
	return &Router{
		bridge:       nil,
		server:       nil,
		agentService: agentSvc,
	}
}

func parseResponse(t *testing.T, raw []byte) *IPCResponse {
	t.Helper()
	var resp IPCResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("failed to parse response: %v\nraw: %s", err, raw)
	}
	return &resp
}

func TestRouter_Ping(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"1","action":"ping"}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got %q", resp.Status)
	}
	if resp.ID != "1" {
		t.Errorf("expected id '1', got %q", resp.ID)
	}
	if resp.Action != MessageTypePing {
		t.Errorf("expected action 'ping', got %q", resp.Action)
	}
}

func TestRouter_UnknownAction(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"2","action":"fly_to_moon"}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "error" {
		t.Errorf("expected status 'error', got %q", resp.Status)
	}
	if resp.Error == "" {
		t.Error("expected non-empty error message")
	}
}

func TestRouter_InvalidJSON(t *testing.T) {
	r := newTestRouter()

	raw := r.HandleMessage("client1", []byte(`{not valid json`))
	resp := parseResponse(t, raw)

	if resp.Status != "error" {
		t.Errorf("expected status 'error' for invalid JSON, got %q", resp.Status)
	}
}

func TestRouter_ListAgents(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"3","action":"list_agents"}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got %q (err: %s)", resp.Status, resp.Error)
	}

	agentList, ok := resp.Data.([]interface{})
	if !ok {
		t.Fatalf("expected []interface{} data, got %T", resp.Data)
	}
	if len(agentList) == 0 {
		t.Error("expected at least one agent in list")
	}
}

func TestRouter_GetAgent_Found(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"4","action":"get_agent","payload":{"codename":"@APEX"}}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got %q (err: %s)", resp.Status, resp.Error)
	}
	if resp.Data == nil {
		t.Error("expected non-nil agent data")
	}
}

func TestRouter_GetAgent_NotFound(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"5","action":"get_agent","payload":{"codename":"@NOBODY"}}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "error" {
		t.Errorf("expected status 'error' for missing agent, got %q", resp.Status)
	}
}

func TestRouter_GetAgent_MissingCodename(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"6","action":"get_agent","payload":{}}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "error" {
		t.Errorf("expected status 'error' for missing codename, got %q", resp.Status)
	}
}

func TestRouter_InvokeTool_UnknownAgent(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"7","action":"invoke_tool","payload":{"agent_codename":"@GHOST","tool_name":"foo"}}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "error" {
		t.Errorf("expected error for unknown agent, got %q", resp.Status)
	}
}

func TestRouter_InvokeTool_MissingFields(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"8","action":"invoke_tool","payload":{"agent_codename":""}}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "error" {
		t.Errorf("expected error for missing fields, got %q", resp.Status)
	}
}

func TestRouter_Inference_MissingPrompt(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"9","action":"inference","payload":{"model_id":"test"}}`

	raw := r.HandleMessage("client1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.Status != "error" {
		t.Errorf("expected error for missing prompt, got %q", resp.Status)
	}
}

func TestRouter_IDEchoedInResponse(t *testing.T) {
	r := newTestRouter()
	req := `{"id":"req-abc-123","action":"ping"}`

	raw := r.HandleMessage("c1", []byte(req))
	resp := parseResponse(t, raw)

	if resp.ID != "req-abc-123" {
		t.Errorf("expected ID 'req-abc-123' echoed, got %q", resp.ID)
	}
}
