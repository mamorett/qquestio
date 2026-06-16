package main

import (
	"context"
	"testing"
)

func TestHandleSlashCmd_Collection(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	cmd := m.handleSlashCmd("/collection my_collection")
	if cmd == nil {
		t.Fatal("expected non-nil tea.Cmd")
	}

	msg := cmd()
	resultMsg, ok := msg.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msg)
	}

	if resultMsg.feedback != "Collection → my_collection" {
		t.Errorf("unexpected feedback: %s", resultMsg.feedback)
	}

	if m.collection != "my_collection" {
		t.Errorf("expected collection to be my_collection, got %s", m.collection)
	}
}

func TestHandleSlashCmd_Limit(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	cmd := m.handleSlashCmd("/limit 10")
	msg := cmd()
	resultMsg, ok := msg.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msg)
	}

	if resultMsg.feedback != "Search limit → 10" {
		t.Errorf("unexpected feedback: %s", resultMsg.feedback)
	}

	if m.searchLimit != 10 {
		t.Errorf("expected searchLimit to be 10, got %d", m.searchLimit)
	}

	// Test invalid limit value
	cmdInvalid := m.handleSlashCmd("/limit 25")
	msgInvalid := cmdInvalid()
	_, isErr := msgInvalid.(appErrMsg)
	if !isErr {
		t.Fatalf("expected appErrMsg for invalid limit, got %T", msgInvalid)
	}
}

func TestHandleSlashCmd_System(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	cmd := m.handleSlashCmd("/system You are a smart coder")
	msg := cmd()
	resultMsg, ok := msg.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msg)
	}

	if resultMsg.feedback != "System prompt updated" {
		t.Errorf("unexpected feedback: %s", resultMsg.feedback)
	}

	if m.systemPrompt != "You are a smart coder" {
		t.Errorf("expected systemPrompt to be updated, got %s", m.systemPrompt)
	}
}
