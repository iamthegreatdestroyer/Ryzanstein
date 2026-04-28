package agents

import (
	"testing"
)

func TestNewService_RegistersFallbackAgents(t *testing.T) {
	svc := NewService(nil)

	agents := svc.ListAgents()
	if len(agents) == 0 {
		t.Fatal("expected at least one agent to be registered")
	}
}

func TestListAgents_ReturnsCodenames(t *testing.T) {
	svc := NewService(nil)
	list := svc.ListAgents()

	if len(list) == 0 {
		t.Fatal("ListAgents returned empty slice")
	}
	for _, name := range list {
		if name == "" {
			t.Error("ListAgents returned empty codename")
		}
	}
}

func TestGetAgent_KnownAgent(t *testing.T) {
	svc := NewService(nil)

	agent := svc.GetAgent("@APEX")
	if agent == nil {
		t.Fatal("GetAgent(@APEX) returned nil")
	}
	if agent.Codename != "@APEX" {
		t.Errorf("expected codename @APEX, got %s", agent.Codename)
	}
	if agent.Tier != 1 {
		t.Errorf("expected tier 1, got %d", agent.Tier)
	}
	if agent.Name == "" {
		t.Error("expected non-empty agent name")
	}
}

func TestGetAgent_UnknownAgent(t *testing.T) {
	svc := NewService(nil)
	agent := svc.GetAgent("@NONEXISTENT")
	if agent != nil {
		t.Error("expected nil for unknown agent")
	}
}

func TestInvokeTool_UnknownAgent(t *testing.T) {
	svc := NewService(nil)
	ctx := t.Context()

	_, err := svc.InvokeTool(ctx, "@NONEXISTENT", "some_tool", nil)
	if err == nil {
		t.Error("expected error for unknown agent")
	}
}

func TestInvokeTool_UnknownTool(t *testing.T) {
	svc := NewService(nil)
	ctx := t.Context()

	// @APEX exists but has no tools registered by default.
	_, err := svc.InvokeTool(ctx, "@APEX", "nonexistent_tool", nil)
	if err == nil {
		t.Error("expected error for unknown tool")
	}
}

func TestInvokeTool_KnownTool(t *testing.T) {
	svc := NewService(nil)
	ctx := t.Context()

	// Register a tool manually then invoke it.
	svc.mu.Lock()
	if agent, ok := svc.agents["@APEX"]; ok {
		agent.Tools["analyze"] = &Tool{
			Name:        "analyze",
			Description: "Analyze code",
			InputSchema: map[string]interface{}{},
		}
	}
	svc.mu.Unlock()

	result, err := svc.InvokeTool(ctx, "@APEX", "analyze", map[string]interface{}{"code": "x := 1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestRegisterFallbackAgents_AllTiers(t *testing.T) {
	svc := NewService(nil)
	svc.mu.RLock()
	defer svc.mu.RUnlock()

	tierCounts := map[int]int{}
	for _, agent := range svc.agents {
		tierCounts[agent.Tier]++
	}

	if tierCounts[1] == 0 {
		t.Error("no tier-1 agents registered")
	}
	if tierCounts[2] == 0 {
		t.Error("no tier-2 agents registered")
	}
}
