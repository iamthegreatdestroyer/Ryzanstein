package agents

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// MemoryClient communicates with the agentmem MCP server via its REST API.
type MemoryClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewMemoryClient creates a client pointing to the agentmem MCP server.
func NewMemoryClient(baseURL string) *MemoryClient {
	if baseURL == "" {
		baseURL = "http://localhost:8100"
	}
	return &MemoryClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Close releases resources held by the client.
func (mc *MemoryClient) Close() {
	mc.httpClient.CloseIdleConnections()
}

// toolCallRequest matches the agentmem FastAPI ToolCallRequest model.
type toolCallRequest struct {
	Tool string                 `json:"tool"`
	Args map[string]interface{} `json:"args"`
}

// callTool posts a tool call to the MCP server and returns the parsed response.
func (mc *MemoryClient) callTool(ctx context.Context, toolName string, args map[string]interface{}) (map[string]interface{}, error) {
	body, err := json.Marshal(toolCallRequest{Tool: toolName, Args: args})
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, mc.baseURL+"/call", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := mc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http call: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if errMsg, ok := result["error"]; ok {
		return nil, fmt.Errorf("tool error: %v", errMsg)
	}

	return result, nil
}

// Healthy checks if the agentmem MCP server is reachable.
func (mc *MemoryClient) Healthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, mc.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := mc.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// --- Result types ---

// RecordResult is the response from memory_record.
type RecordResult struct {
	Status   string `json:"status"`
	MemoryID string `json:"memory_id"`
}

// MemoryEntry represents a single recalled memory.
type MemoryEntry struct {
	ID         string                 `json:"id"`
	Similarity float64                `json:"similarity"`
	Content    map[string]interface{} `json:"content"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// RecallResult is the response from memory_recall.
type RecallResult struct {
	Memories []MemoryEntry `json:"memories"`
}

// FactEntry represents a single recalled fact.
type FactEntry struct {
	ID         string                 `json:"id"`
	Similarity float64                `json:"similarity"`
	Content    map[string]interface{} `json:"content"`
}

// FactsResult is the response from memory_recall_facts.
type FactsResult struct {
	Facts []FactEntry `json:"facts"`
}

// AssertResult is the response from memory_assert_fact.
type AssertResult struct {
	Status string `json:"status"`
	FactID string `json:"fact_id"`
}

// WorkflowEntry represents a single workflow result.
type WorkflowEntry struct {
	ID          string        `json:"id"`
	Similarity  float64       `json:"similarity"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Steps       []interface{} `json:"steps"`
	SuccessRate float64       `json:"success_rate"`
	UsageCount  int           `json:"usage_count"`
}

// WorkflowResult is the response from memory_workflow_find.
type WorkflowResult struct {
	Workflows []WorkflowEntry `json:"workflows"`
}

// WorkflowStep is a step in a procedural workflow.
type WorkflowStep struct {
	Action string `json:"action"`
	Tool   string `json:"tool,omitempty"`
}

// AddWorkflowResult is the response from memory_workflow_add.
type AddWorkflowResult struct {
	Status     string `json:"status"`
	WorkflowID string `json:"workflow_id"`
}

// ConsolidateResult is the response from memory_consolidate.
type ConsolidateResult struct {
	Status              string `json:"status"`
	EpisodesProcessed   int    `json:"episodes_processed"`
	FactsExtracted      int    `json:"facts_extracted"`
	ContradictionsFound int    `json:"contradictions_found"`
	PatternsDiscovered  int    `json:"patterns_discovered"`
}

// StatsResult is the response from memory_stats.
type StatsResult struct {
	TotalMemories int                    `json:"total_memories"`
	Agents        []string               `json:"agents"`
	ByLayer       map[string]interface{} `json:"by_layer"`
}

// --- Tool methods ---

// RecordMemory records an episodic memory for an agent.
func (mc *MemoryClient) RecordMemory(ctx context.Context, agentID, task, outcome string, strategy, reasoning string) (*RecordResult, error) {
	args := map[string]interface{}{
		"agent_id": agentID,
		"task":     task,
		"outcome":  outcome,
	}
	if strategy != "" {
		args["strategy"] = strategy
	}
	if reasoning != "" {
		args["reasoning"] = reasoning
	}

	raw, err := mc.callTool(ctx, "memory_record", args)
	if err != nil {
		return nil, err
	}

	return &RecordResult{
		Status:   stringVal(raw, "status"),
		MemoryID: stringVal(raw, "memory_id"),
	}, nil
}

// RecallMemory performs semantic search over episodic memories.
func (mc *MemoryClient) RecallMemory(ctx context.Context, agentID, query string, topK int, outcomeFilter string) (*RecallResult, error) {
	args := map[string]interface{}{
		"agent_id": agentID,
		"query":    query,
		"top_k":    topK,
	}
	if outcomeFilter != "" {
		args["outcome_filter"] = outcomeFilter
	}

	raw, err := mc.callTool(ctx, "memory_recall", args)
	if err != nil {
		return nil, err
	}

	result := &RecallResult{}
	if memories, ok := raw["memories"].([]interface{}); ok {
		for _, m := range memories {
			if entry, ok := m.(map[string]interface{}); ok {
				result.Memories = append(result.Memories, MemoryEntry{
					ID:         stringVal(entry, "id"),
					Similarity: floatVal(entry, "similarity"),
					Content:    mapVal(entry, "content"),
					Metadata:   mapVal(entry, "metadata"),
				})
			}
		}
	}
	return result, nil
}

// RecallFacts queries the semantic/knowledge-graph layer.
func (mc *MemoryClient) RecallFacts(ctx context.Context, agentID, query string, topK int) (*FactsResult, error) {
	args := map[string]interface{}{
		"agent_id": agentID,
		"query":    query,
		"top_k":    topK,
	}

	raw, err := mc.callTool(ctx, "memory_recall_facts", args)
	if err != nil {
		return nil, err
	}

	result := &FactsResult{}
	if facts, ok := raw["facts"].([]interface{}); ok {
		for _, f := range facts {
			if entry, ok := f.(map[string]interface{}); ok {
				result.Facts = append(result.Facts, FactEntry{
					ID:         stringVal(entry, "id"),
					Similarity: floatVal(entry, "similarity"),
					Content:    mapVal(entry, "content"),
				})
			}
		}
	}
	return result, nil
}

// AssertFact adds a fact to the semantic memory layer.
func (mc *MemoryClient) AssertFact(ctx context.Context, agentID, subject, predicate, value string, confidence float64) (*AssertResult, error) {
	args := map[string]interface{}{
		"agent_id":   agentID,
		"subject":    subject,
		"predicate":  predicate,
		"value":      value,
		"confidence": confidence,
	}

	raw, err := mc.callTool(ctx, "memory_assert_fact", args)
	if err != nil {
		return nil, err
	}

	return &AssertResult{
		Status: stringVal(raw, "status"),
		FactID: stringVal(raw, "fact_id"),
	}, nil
}

// FindWorkflow searches for procedural workflows.
func (mc *MemoryClient) FindWorkflow(ctx context.Context, agentID, query string, topK int, minSuccessRate float64) (*WorkflowResult, error) {
	args := map[string]interface{}{
		"agent_id":         agentID,
		"query":            query,
		"top_k":            topK,
		"min_success_rate": minSuccessRate,
	}

	raw, err := mc.callTool(ctx, "memory_workflow_find", args)
	if err != nil {
		return nil, err
	}

	result := &WorkflowResult{}
	if workflows, ok := raw["workflows"].([]interface{}); ok {
		for _, w := range workflows {
			if entry, ok := w.(map[string]interface{}); ok {
				result.Workflows = append(result.Workflows, WorkflowEntry{
					ID:          stringVal(entry, "id"),
					Similarity:  floatVal(entry, "similarity"),
					Name:        stringVal(entry, "name"),
					Description: stringVal(entry, "description"),
					Steps:       sliceVal(entry, "steps"),
					SuccessRate: floatVal(entry, "success_rate"),
					UsageCount:  intVal(entry, "usage_count"),
				})
			}
		}
	}
	return result, nil
}

// AddWorkflow saves a procedural workflow.
func (mc *MemoryClient) AddWorkflow(ctx context.Context, agentID, name string, steps []WorkflowStep, description, applicability string) (*AddWorkflowResult, error) {
	stepMaps := make([]map[string]interface{}, len(steps))
	for i, s := range steps {
		sm := map[string]interface{}{"action": s.Action}
		if s.Tool != "" {
			sm["tool"] = s.Tool
		}
		stepMaps[i] = sm
	}

	args := map[string]interface{}{
		"agent_id": agentID,
		"name":     name,
		"steps":    stepMaps,
	}
	if description != "" {
		args["description"] = description
	}
	if applicability != "" {
		args["applicability"] = applicability
	}

	raw, err := mc.callTool(ctx, "memory_workflow_add", args)
	if err != nil {
		return nil, err
	}

	return &AddWorkflowResult{
		Status:     stringVal(raw, "status"),
		WorkflowID: stringVal(raw, "workflow_id"),
	}, nil
}

// Consolidate runs the memory consolidation pipeline.
func (mc *MemoryClient) Consolidate(ctx context.Context, agentIDs []string) (*ConsolidateResult, error) {
	args := map[string]interface{}{}
	if len(agentIDs) > 0 {
		args["agent_ids"] = agentIDs
	}

	raw, err := mc.callTool(ctx, "memory_consolidate", args)
	if err != nil {
		return nil, err
	}

	return &ConsolidateResult{
		Status:              stringVal(raw, "status"),
		EpisodesProcessed:   intVal(raw, "episodes_processed"),
		FactsExtracted:      intVal(raw, "facts_extracted"),
		ContradictionsFound: intVal(raw, "contradictions_found"),
		PatternsDiscovered:  intVal(raw, "patterns_discovered"),
	}, nil
}

// Stats returns memory store statistics.
func (mc *MemoryClient) Stats(ctx context.Context) (*StatsResult, error) {
	raw, err := mc.callTool(ctx, "memory_stats", map[string]interface{}{})
	if err != nil {
		return nil, err
	}

	result := &StatsResult{
		TotalMemories: intVal(raw, "total_memories"),
		ByLayer:       mapVal(raw, "by_layer"),
	}
	if agents, ok := raw["agents"].([]interface{}); ok {
		for _, a := range agents {
			if s, ok := a.(string); ok {
				result.Agents = append(result.Agents, s)
			}
		}
	}
	return result, nil
}

// --- JSON helpers ---

func stringVal(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func floatVal(m map[string]interface{}, key string) float64 {
	if v, ok := m[key]; ok {
		switch n := v.(type) {
		case float64:
			return n
		case int:
			return float64(n)
		}
	}
	return 0
}

func intVal(m map[string]interface{}, key string) int {
	if v, ok := m[key]; ok {
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		}
	}
	return 0
}

func mapVal(m map[string]interface{}, key string) map[string]interface{} {
	if v, ok := m[key]; ok {
		if sub, ok := v.(map[string]interface{}); ok {
			return sub
		}
	}
	return nil
}

func sliceVal(m map[string]interface{}, key string) []interface{} {
	if v, ok := m[key]; ok {
		if s, ok := v.([]interface{}); ok {
			return s
		}
	}
	return nil
}
