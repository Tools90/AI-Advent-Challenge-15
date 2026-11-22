package main

import (
	"context"
	"strings"
	"testing"
)

type scriptedLLM struct {
	responses []string
	index     int
}

func (f *scriptedLLM) Chat(ctx context.Context, messages []Message) (LLMResponse, error) {
	content := ""
	if f.index < len(f.responses) {
		content = f.responses[f.index]
		f.index++
	}
	return LLMResponse{Message: Message{Role: "assistant", Content: content}, Usage: Usage{TotalTokens: 1}}, nil
}

func (f *scriptedLLM) Model() string { return "fake" }

func TestReplaceConflictAndPrompt(t *testing.T) {
	store, err := NewMemoryStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.ApplyUpdates([]MemoryUpdate{{Key: "deadline", Value: "15 марта", Mode: "replace", Layer: "working", Reason: "старт"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ApplyUpdates([]MemoryUpdate{{Key: "deadline", Value: "20 марта", Mode: "replace", Layer: "working", Reason: "срок сдвинулся"}}); err != nil {
		t.Fatal(err)
	}
	state := store.State()
	active := activeFacts(state.Facts, "working")
	if len(active) != 1 || active[0].Value != "20 марта" {
		t.Fatalf("active = %+v", active)
	}
	var replacement *MemoryLog
	for i := range state.Logs {
		if state.Logs[i].Action == "replace" && state.Logs[i].NewValue == "20 марта" {
			replacement = &state.Logs[i]
		}
	}
	if replacement == nil || replacement.OldValue != "15 марта" || replacement.NewValue != "20 марта" {
		t.Fatalf("logs = %+v", state.Logs)
	}
	messages := buildAnswerMessages("Какой срок?", []Message{{Role: "user", Content: "Какой срок?"}}, state.Facts)
	joined := messages[0].Content
	if !strings.Contains(joined, "20 марта") || strings.Contains(joined, "15 марта") {
		t.Fatalf("prompt contains stale value: %s", joined)
	}
}

func TestAppendAndDelete(t *testing.T) {
	store, err := NewMemoryStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	updates := []MemoryUpdate{
		{Key: "decision", Value: "REST", Mode: "append", Layer: "long", Reason: "выбрали REST"},
		{Key: "decision", Value: "REST", Mode: "append", Layer: "long", Reason: "дубль"},
		{Key: "decision", Value: "GraphQL", Mode: "append", Layer: "long", Reason: "проверяли GraphQL"},
	}
	if _, err := store.ApplyUpdates(updates); err != nil {
		t.Fatal(err)
	}
	if got := activeFacts(store.State().Facts, "long"); len(got) != 2 {
		t.Fatalf("append dedupe failed: %+v", got)
	}
	if _, err := store.ApplyUpdates([]MemoryUpdate{{Key: "decision", Mode: "delete", Layer: "long", Reason: "решение отменено"}}); err != nil {
		t.Fatal(err)
	}
	if got := activeFacts(store.State().Facts, "long"); len(got) != 0 {
		t.Fatalf("delete failed: %+v", got)
	}
}

func TestCompleteTaskClearsWorkingKeepsLong(t *testing.T) {
	store, err := NewMemoryStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.ApplyUpdates([]MemoryUpdate{
		{Key: "deadline", Value: "20 марта", Mode: "replace", Layer: "working"},
		{Key: "preference", Value: "короткие ответы", Mode: "replace", Layer: "long"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteTask(); err != nil {
		t.Fatal(err)
	}
	state := store.State()
	if len(activeFacts(state.Facts, "working")) != 0 {
		t.Fatalf("working not cleared: %+v", activeFacts(state.Facts, "working"))
	}
	if len(activeFacts(state.Facts, "long")) != 1 {
		t.Fatalf("long cleared: %+v", activeFacts(state.Facts, "long"))
	}
}

func TestExtractAndAnswerUseRequests(t *testing.T) {
	llm := &scriptedLLM{responses: []string{
		`{"updates":[{"key":"deadline","value":"20 марта","mode":"replace","layer":"working","reason":"сдвиг"}]}`,
		"Актуальный срок — 20 марта.",
	}}
	extraction, _, err := ExtractMemory(context.Background(), llm, "Срок теперь 20 марта", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(extraction.Updates) != 1 || extraction.Updates[0].Key != "deadline" {
		t.Fatalf("updates = %+v", extraction.Updates)
	}
	if len(extraction.Request) == 0 {
		t.Fatal("extraction request missing")
	}
	answer, _, err := AskAnswer(context.Background(), llm, "Какой срок?", []Message{{Role: "user", Content: "Какой срок?"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer.Reply == "" || len(answer.Request) == 0 {
		t.Fatalf("answer = %+v", answer)
	}
}

func TestKeyModeIsFixedAfterFirstUpdate(t *testing.T) {
	store, err := NewMemoryStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.ApplyUpdates([]MemoryUpdate{{Key: "deadline", Value: "15 марта", Mode: "append", Layer: "working"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ApplyUpdates([]MemoryUpdate{{Key: "deadline", Value: "20 марта", Mode: "replace", Layer: "working"}}); err != nil {
		t.Fatal(err)
	}
	state := store.State()
	policy := findKeyPolicy(state.Keys, "deadline")
	if policy == nil || policy.Mode != "append" {
		t.Fatalf("policy = %+v", policy)
	}
	active := activeFacts(state.Facts, "working")
	if len(active) != 2 {
		t.Fatalf("mode changed after registration: %+v", active)
	}
}

func TestFormatFactsShowsArrayIndexes(t *testing.T) {
	facts := []*Fact{
		{Key: "decisions", Value: "S3", Mode: "append", Status: "active", Layer: "long", CreatedAt: "2026-01-01T00:00:01Z"},
		{Key: "decisions", Value: "Redis", Mode: "append", Status: "active", Layer: "long", CreatedAt: "2026-01-01T00:00:02Z"},
	}
	text := formatFacts(activeFacts(facts, "long"))
	if !strings.Contains(text, "decisions[0]: S3") || !strings.Contains(text, "decisions[1]: Redis") {
		t.Fatalf("format = %s", text)
	}
}
