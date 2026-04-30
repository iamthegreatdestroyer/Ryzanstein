package chat

import (
	"context"
	"testing"
)

func TestNewService_EmptyHistory(t *testing.T) {
	svc := NewService()
	history := svc.GetHistory(0)
	if len(history) != 0 {
		t.Fatalf("expected empty history, got %d entries", len(history))
	}
}

func TestAddMessage_AppendsToHistory(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	svc.AddMessage(ctx, "user", "hello", "model-x", "atlas")
	history := svc.GetHistory(0)

	if len(history) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(history))
	}
	msg := history[0]
	if msg.Role != "user" {
		t.Errorf("expected role 'user', got %q", msg.Role)
	}
	if msg.Content != "hello" {
		t.Errorf("expected content 'hello', got %q", msg.Content)
	}
	if msg.ID == "" {
		t.Error("expected non-empty ID")
	}
	if msg.Timestamp <= 0 {
		t.Error("expected positive timestamp")
	}
}

func TestGetHistory_LimitPositive(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	for i := range 6 {
		_ = i
		svc.AddMessage(ctx, "user", "msg", "m", "a")
	}

	tail := svc.GetHistory(3)
	if len(tail) != 3 {
		t.Fatalf("expected 3 entries with limit=3, got %d", len(tail))
	}
}

func TestGetHistory_LimitZeroReturnsAll(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	for range 5 {
		svc.AddMessage(ctx, "user", "x", "m", "a")
	}

	all := svc.GetHistory(0)
	if len(all) != 5 {
		t.Fatalf("expected 5 entries with limit=0, got %d", len(all))
	}
}

func TestGetHistory_LimitExceedsLength(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	svc.AddMessage(ctx, "user", "only one", "m", "a")

	result := svc.GetHistory(100)
	if len(result) != 1 {
		t.Fatalf("expected 1 entry when limit > len, got %d", len(result))
	}
}

func TestGetHistory_LimitReturnsSuffix(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	contents := []string{"first", "second", "third", "fourth"}
	for _, c := range contents {
		svc.AddMessage(ctx, "user", c, "m", "a")
	}

	tail := svc.GetHistory(2)
	if len(tail) != 2 {
		t.Fatalf("expected 2, got %d", len(tail))
	}
	if tail[0].Content != "third" || tail[1].Content != "fourth" {
		t.Errorf("expected last 2 entries, got %q %q", tail[0].Content, tail[1].Content)
	}
}

func TestClearHistory_EmptiesSlice(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	svc.AddMessage(ctx, "user", "msg", "m", "a")
	svc.ClearHistory()

	history := svc.GetHistory(0)
	if len(history) != 0 {
		t.Fatalf("expected empty history after clear, got %d entries", len(history))
	}
}

func TestClearHistory_RepopulateAfterClear(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	svc.AddMessage(ctx, "user", "before", "m", "a")
	svc.ClearHistory()
	svc.AddMessage(ctx, "user", "after", "m", "a")

	history := svc.GetHistory(0)
	if len(history) != 1 {
		t.Fatalf("expected 1 entry after repopulate, got %d", len(history))
	}
	if history[0].Content != "after" {
		t.Errorf("expected 'after', got %q", history[0].Content)
	}
}

func TestAddMessage_UniqueIDs(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	for range 10 {
		svc.AddMessage(ctx, "user", "x", "m", "a")
	}

	history := svc.GetHistory(0)
	seen := map[string]bool{}
	for _, msg := range history {
		if seen[msg.ID] {
			t.Errorf("duplicate message ID: %s", msg.ID)
		}
		seen[msg.ID] = true
	}
}

func TestClose_CanBeCalledSafely(t *testing.T) {
	svc := NewService()
	ctx := context.Background()

	svc.AddMessage(ctx, "user", "data", "m", "a")
	svc.Close()

	// After close, history should be empty.
	if len(svc.GetHistory(0)) != 0 {
		t.Error("expected empty history after Close()")
	}
}
