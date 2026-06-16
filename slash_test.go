package main

import (
	"context"
	"os"
	"strings"
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

func TestHandleSlashCmd_Save(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	// Populate history to have something to save
	m.history = append(m.history,
		ConversationTurn{Role: "user", Content: "Hello assistant"},
		ConversationTurn{Role: "assistant", Content: "Hello user, this is a response."},
	)

	tempDir, err := os.MkdirTemp("", "qquestio_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	tempFile := tempDir + "/test_response.md"
	cmd := m.handleSlashCmd("/save " + tempFile)
	msg := cmd()
	resultMsg, ok := msg.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msg)
	}
	if !strings.Contains(resultMsg.feedback, "Saved last response") {
		t.Errorf("unexpected feedback: %s", resultMsg.feedback)
	}

	// Verify file was written
	data, err := os.ReadFile(tempFile)
	if err != nil {
		t.Fatalf("failed to read written file: %v", err)
	}
	if string(data) != "Hello user, this is a response." {
		t.Errorf("unexpected file content: %s", string(data))
	}

	// Test save all
	tempAllFile := tempDir + "/test_all.md"
	cmdAll := m.handleSlashCmd("/save all " + tempAllFile)
	msgAll := cmdAll()
	resultAllMsg, ok := msgAll.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgAll)
	}
	if !strings.Contains(resultAllMsg.feedback, "Saved entire transcript") {
		t.Errorf("unexpected feedback: %s", resultAllMsg.feedback)
	}

	// Verify all file content
	dataAll, err := os.ReadFile(tempAllFile)
	if err != nil {
		t.Fatalf("failed to read all file: %v", err)
	}
	if !strings.Contains(string(dataAll), "# Conversation Transcript") || !strings.Contains(string(dataAll), "Hello assistant") {
		t.Errorf("unexpected file content for save all: %s", string(dataAll))
	}
}
