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
	cmdInvalid := m.handleSlashCmd("/limit 105")
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

func TestHandleSlashCmd_Mode(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	// Test default mode
	if m.ragMode != "strict" {
		t.Errorf("expected default ragMode to be strict, got %s", m.ragMode)
	}

	// Switch to hybrid
	cmd := m.handleSlashCmd("/mode hybrid")
	msg := cmd()
	resultMsg, ok := msg.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msg)
	}
	if resultMsg.feedback != "RAG mode → hybrid" {
		t.Errorf("unexpected feedback: %s", resultMsg.feedback)
	}
	if m.ragMode != "hybrid" {
		t.Errorf("expected ragMode to be hybrid, got %s", m.ragMode)
	}

	// Switch to strict
	cmdStrict := m.handleSlashCmd("/mode strict")
	msgStrict := cmdStrict()
	resultStrictMsg, ok := msgStrict.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgStrict)
	}
	if resultStrictMsg.feedback != "RAG mode → strict" {
		t.Errorf("unexpected feedback: %s", resultStrictMsg.feedback)
	}
	if m.ragMode != "strict" {
		t.Errorf("expected ragMode to be strict, got %s", m.ragMode)
	}

	// Invalid mode
	cmdInvalid := m.handleSlashCmd("/mode invalid")
	msgInvalid := cmdInvalid()
	_, isErr := msgInvalid.(appErrMsg)
	if !isErr {
		t.Fatalf("expected appErrMsg for invalid mode, got %T", msgInvalid)
	}
}

func TestHandleSlashCmd_Filter(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	// Test default filter
	if m.filterKey != "" || m.filterValue != "" {
		t.Errorf("expected default filter to be empty, got key=%s, val=%s", m.filterKey, m.filterValue)
	}

	// Apply specific key-value filter
	cmd := m.handleSlashCmd("/filter file_name guide.txt")
	msg := cmd()
	resultMsg, ok := msg.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msg)
	}
	if resultMsg.feedback != "Filter applied → file_name = guide.txt" {
		t.Errorf("unexpected feedback: %s", resultMsg.feedback)
	}
	if m.filterKey != "file_name" || m.filterValue != "guide.txt" {
		t.Errorf("expected filter to be file_name=guide.txt, got %s=%s", m.filterKey, m.filterValue)
	}

	// Clear filter
	cmdClear := m.handleSlashCmd("/filter clear")
	msgClear := cmdClear()
	resultClearMsg, ok := msgClear.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgClear)
	}
	if resultClearMsg.feedback != "Active metadata filter cleared" {
		t.Errorf("unexpected feedback: %s", resultClearMsg.feedback)
	}
	if m.filterKey != "" || m.filterValue != "" {
		t.Errorf("expected filter to be empty after clear, got %s=%s", m.filterKey, m.filterValue)
	}

	// Wildcard filter
	cmdWildcard := m.handleSlashCmd("/filter guide.txt")
	msgWildcard := cmdWildcard()
	resultWildcardMsg, ok := msgWildcard.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgWildcard)
	}
	if !strings.Contains(resultWildcardMsg.feedback, "[any document field] = guide.txt") {
		t.Errorf("unexpected feedback: %s", resultWildcardMsg.feedback)
	}
	if m.filterKey != "*" || m.filterValue != "guide.txt" {
		t.Errorf("expected filter to be *=guide.txt, got %s=%s", m.filterKey, m.filterValue)
	}
}

func TestHandleSlashCmd_Rerank(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	// Default state
	if m.disableReranker {
		t.Errorf("expected default disableReranker to be false, got true")
	}

	// Disable it
	cmdOff := m.handleSlashCmd("/rerank off")
	msgOff := cmdOff()
	resultOff, ok := msgOff.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgOff)
	}
	if resultOff.feedback != "Reranker disabled (bypassed)" {
		t.Errorf("unexpected feedback: %s", resultOff.feedback)
	}
	if !m.disableReranker {
		t.Errorf("expected disableReranker to be true, got false")
	}

	// Enable it
	cmdOn := m.handleSlashCmd("/rerank on")
	msgOn := cmdOn()
	resultOn, ok := msgOn.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgOn)
	}
	if resultOn.feedback != "Reranker enabled" {
		t.Errorf("unexpected feedback: %s", resultOn.feedback)
	}
	if m.disableReranker {
		t.Errorf("expected disableReranker to be false, got true")
	}

	// Invalid arg
	cmdInvalid := m.handleSlashCmd("/rerank invalid")
	msgInvalid := cmdInvalid()
	_, isErr := msgInvalid.(appErrMsg)
	if !isErr {
		t.Fatalf("expected appErrMsg for invalid arg, got %T", msgInvalid)
	}
}

func TestHandleSlashCmd_RerankerPool(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	// Default state in Model constructor: cfg.RerankerPool (which is 0 here)
	if m.rerankerPool != 0 {
		t.Errorf("expected default rerankerPool to be 0, got %d", m.rerankerPool)
	}

	// 1. Check showing status with no arguments
	cmdShow := m.handleSlashCmd("/rerankerpool")
	msgShow := cmdShow()
	resShow, ok := msgShow.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgShow)
	}
	if !strings.Contains(resShow.feedback, "auto") {
		t.Errorf("expected feedback to contain 'auto', got %s", resShow.feedback)
	}

	// 2. Set to custom size
	cmdSet := m.handleSlashCmd("/rerankerpool 120")
	msgSet := cmdSet()
	resSet, ok := msgSet.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgSet)
	}
	if !strings.Contains(resSet.feedback, "120") {
		t.Errorf("expected feedback to contain '120', got %s", resSet.feedback)
	}
	if m.rerankerPool != 120 {
		t.Errorf("expected rerankerPool to be 120, got %d", m.rerankerPool)
	}

	// 3. Check showing status again
	cmdShow2 := m.handleSlashCmd("/rerankerpool")
	msgShow2 := cmdShow2()
	resShow2, ok := msgShow2.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgShow2)
	}
	if !strings.Contains(resShow2.feedback, "120") {
		t.Errorf("expected feedback to contain '120', got %s", resShow2.feedback)
	}

	// 4. Set to auto
	cmdAuto := m.handleSlashCmd("/rerankerpool auto")
	msgAuto := cmdAuto()
	resAuto, ok := msgAuto.(slashResultMsg)
	if !ok {
		t.Fatalf("expected slashResultMsg, got %T", msgAuto)
	}
	if !strings.Contains(resAuto.feedback, "auto") {
		t.Errorf("expected feedback to contain 'auto', got %s", resAuto.feedback)
	}
	if m.rerankerPool != 0 {
		t.Errorf("expected rerankerPool to be 0, got %d", m.rerankerPool)
	}

	// 5. Test invalid argument
	cmdInvalid := m.handleSlashCmd("/rerankerpool invalid")
	msgInvalid := cmdInvalid()
	_, isErr := msgInvalid.(appErrMsg)
	if !isErr {
		t.Fatalf("expected appErrMsg for invalid arg, got %T", msgInvalid)
	}
}

