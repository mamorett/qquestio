package main

import (
	"context"
	"os"
	"path/filepath"
	"qquestio/internal/rag"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestDoubleEscapeCancel(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	// Set model to an active state
	m.state = stateStreaming
	m.lastQuery = "Who are you?"
	m.output = "I am an assistant..."

	// Send first Esc key
	escMsg := tea.KeyMsg{Type: tea.KeyEscape}
	model, cmd := m.Update(escMsg)
	m = model.(*Model)

	if m.state != stateStreaming {
		t.Errorf("expected state to remain stateStreaming after first Esc, got %d", m.state)
	}
	if m.escCount != 1 {
		t.Errorf("expected escCount to be 1, got %d", m.escCount)
	}
	if cmd != nil {
		t.Error("expected nil cmd after first Esc")
	}

	// Send second Esc key
	model, cmd = m.Update(escMsg)
	m = model.(*Model)

	if m.state != stateIdle {
		t.Errorf("expected state to transition to stateIdle after double Esc, got %d", m.state)
	}
	if m.escCount != 0 {
		t.Errorf("expected escCount to be reset to 0, got %d", m.escCount)
	}
	if m.statusMsg != "Generation stopped" {
		t.Errorf("expected statusMsg to be 'Generation stopped', got %s", m.statusMsg)
	}
	if m.lastQuery != "" || m.output != "" {
		t.Errorf("expected transient query and output to be cleared, got lastQuery=%s output=%s", m.lastQuery, m.output)
	}

	// Test that other keys reset escCount
	m.state = stateStreaming
	m.escCount = 1
	otherKeyMsg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("a")}
	model, _ = m.Update(otherKeyMsg)
	m = model.(*Model)
	if m.escCount != 0 {
		t.Errorf("expected escCount to reset to 0 after other key press, got %d", m.escCount)
	}
}

func TestFormatNumber(t *testing.T) {
	tests := []struct {
		in   int
		want string
	}{
		{0, "0"},
		{1, "1"},
		{999, "999"},
		{1000, "1,000"},
		{12345, "12,345"},
		{1234567, "1,234,567"},
		{-1234567, "-1,234,567"},
	}
	for _, tc := range tests {
		got := formatNumber(tc.in)
		if got != tc.want {
			t.Errorf("formatNumber(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestFormatReferencesDocumentTitle is a regression test for the "lost doc title"
// bug where the references panel showed "Document: ID <id>" instead of the
// actual filename for points whose doc name lives in a nested map or under
// an unusual key. The fix routes formatReferences through extractDocumentName
// (the same recursive helper that buildPromptMessages uses), so both the
// panel and the LLM prompt see the same titles.
func TestFormatReferencesDocumentTitle(t *testing.T) {
	// Case 1: top-level file_name key (the common case).
	p1 := rag.QdrantPoint{
		ID:    "1",
		Score: 0.95,
		Payload: map[string]interface{}{
			"file_name":   "report-2024.txt",
			"chunk_index": float64(0),
			"text":        "Q1 revenue was $1.2B.",
		},
	}
	// Case 2: doc name lives in a nested map (the bug-triggering case).
	p2 := rag.QdrantPoint{
		ID:    "2",
		Score: 0.88,
		Payload: map[string]interface{}{
			"metadata": map[string]interface{}{
				"source": "deeply-nested-doc.pdf",
			},
			"chunk_index": float64(1),
			"text":        "Margins improved 200bps.",
		},
	}
	// Case 3: doc name under an unusual key ("name" rather than "file_name").
	p3 := rag.QdrantPoint{
		ID:    "3",
		Score: 0.80,
		Payload: map[string]interface{}{
			"name":        "short-name.txt",
			"chunk_index": float64(2),
			"text":        "EPS of $1.05.",
		},
	}

	out := formatReferences([]rag.QdrantPoint{p1, p2, p3}, 100)
	if !strings.Contains(out, "report-2024.txt") {
		t.Errorf("references panel should contain top-level file_name; got:\n%s", out)
	}
	if !strings.Contains(out, "deeply-nested-doc.pdf") {
		t.Errorf("references panel should contain doc name from nested map; got:\n%s", out)
	}
	if !strings.Contains(out, "short-name.txt") {
		t.Errorf("references panel should contain doc name from unusual key; got:\n%s", out)
	}
	// And none of them should have been replaced with the "Document: ID <id>" fallback.
	if strings.Contains(out, "Document: ID") {
		t.Errorf("references panel still shows 'Document: ID' fallback for at least one point; got:\n%s", out)
	}
}

func TestComputeSearchDocs(t *testing.T) {
	// Case 1: Reranker disabled
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)
	m.disableReranker = true

	// docs = 10, expand = 0 -> 10
	if got := m.computeSearchDocs(10, 0); got != 10 {
		t.Errorf("Reranker disabled: expected 10, got %d", got)
	}
	// docs = 10, expand = 1 -> 10
	if got := m.computeSearchDocs(10, 1); got != 10 {
		t.Errorf("Reranker disabled with expand=1: expected 10, got %d", got)
	}
	// docs = 10, expand = 100 -> 10
	if got := m.computeSearchDocs(10, 100); got != 10 {
		t.Errorf("Reranker disabled with expand=100 (cap): expected 10, got %d", got)
	}

	// Case 2: Reranker enabled, URL set, default rerankerPool (0 = auto)
	cfgRerank := Config{
		DefaultCollection: "default",
		RerankerURL:       "http://localhost:8001",
	}
	mRerank := NewModel(context.Background(), cfgRerank)
	mRerank.disableReranker = false
	mRerank.rerankerPool = 0 // auto

	// docs = 5, expand = 0 -> basePool = max(5*5, 50) = 50. expand=0 -> 50.
	if got := mRerank.computeSearchDocs(5, 0); got != 50 {
		t.Errorf("Reranker auto, docs=5: expected 50, got %d", got)
	}
	// docs = 15, expand = 0 -> basePool = max(15*5, 50) = 75. expand=0 -> 75.
	if got := mRerank.computeSearchDocs(15, 0); got != 75 {
		t.Errorf("Reranker auto, docs=15: expected 75, got %d", got)
	}
	// docs = 5, expand = 1 -> basePool = 50. expand=1 -> 50 * 2 = 100.
	if got := mRerank.computeSearchDocs(5, 1); got != 100 {
		t.Errorf("Reranker auto with expand=1, docs=5: expected 100, got %d", got)
	}
	// docs = 5, expand = 10 -> basePool = 50. expand=10 -> 50 * 11 = 550 -> 500 (capped).
	if got := mRerank.computeSearchDocs(5, 10); got != 500 {
		t.Errorf("Reranker auto with expand=10, docs=5: expected 500, got %d", got)
	}

	// Case 3: Reranker enabled, user-specified custom pool (rerankerPool = 30)
	mRerankCustom := NewModel(context.Background(), cfgRerank)
	mRerankCustom.disableReranker = false
	mRerankCustom.rerankerPool = 30

	// docs = 10, expand = 0 -> 30
	if got := mRerankCustom.computeSearchDocs(10, 0); got != 30 {
		t.Errorf("Reranker custom pool 30, docs=10: expected 30, got %d", got)
	}
	// docs = 10, expand = 1 -> 30 * 2 = 60
	if got := mRerankCustom.computeSearchDocs(10, 1); got != 60 {
		t.Errorf("Reranker custom pool 30, docs=10, expand=1: expected 60, got %d", got)
	}

	// Case 4: Reranker enabled, user-specified large pool (rerankerPool = 600)
	mRerankLarge := NewModel(context.Background(), cfgRerank)
	mRerankLarge.disableReranker = false
	mRerankLarge.rerankerPool = 600

	// docs = 10, expand = 0 -> 600
	if got := mRerankLarge.computeSearchDocs(10, 0); got != 600 {
		t.Errorf("Reranker custom pool 600, docs=10: expected 600, got %d", got)
	}
	// docs = 10, expand = 1 -> 600 * 2 = 1200 -> capped at 600 (since capVal = max(500, rerankerPool))
	if got := mRerankLarge.computeSearchDocs(10, 1); got != 600 {
		t.Errorf("Reranker custom pool 600, docs=10, expand=1: expected 600, got %d", got)
	}
}

func TestSplitPanelFocusAndCopy(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	if m.focusRef {
		t.Error("expected initial focusRef to be false")
	}

	// Test Tab key focus toggle
	tabMsg := tea.KeyMsg{Type: tea.KeyTab}
	model, _ := m.Update(tabMsg)
	m = model.(*Model)

	if !m.focusRef {
		t.Error("expected focusRef to be true after Tab")
	}

	model, _ = m.Update(tabMsg)
	m = model.(*Model)

	if m.focusRef {
		t.Error("expected focusRef to be false after second Tab")
	}

	// Test Mouse click focus switch
	m.width = 90
	m.height = 30
	// Click on references panel (msg.X >= mainWidth, where mainWidth = 90 - 30 = 60)
	mouseRefClick := tea.MouseMsg{
		X: 70,
		Y: 10,
	}
	model, _ = m.Update(mouseRefClick)
	m = model.(*Model)
	if !m.focusRef {
		t.Error("expected focusRef to be true after clicking on the right panel")
	}

	// Click on main panel (msg.X < mainWidth)
	mouseMainClick := tea.MouseMsg{
		X: 30,
		Y: 10,
	}
	model, _ = m.Update(mouseMainClick)
	m = model.(*Model)
	if m.focusRef {
		t.Error("expected focusRef to be false after clicking on the left panel")
	}
}

func TestSessionSaveAndLoad(t *testing.T) {
	// Setup custom home dir for the test to avoid overwriting actual user config
	tempDir := t.TempDir()
	t.Setenv("HOME", tempDir)

	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)
	m.sessionID = "test-session-123"
	m.collection = "test-collection"
	m.searchLimit = 42
	m.history = []ConversationTurn{
		{Role: "user", Content: "Hello session test"},
		{Role: "assistant", Content: "I am tested", References: []rag.QdrantPoint{
			{ID: "point1", Score: 0.99, Payload: map[string]interface{}{"text": "sample"}},
		}},
	}

	// Save
	if err := m.saveSession(); err != nil {
		t.Fatalf("failed to save session: %v", err)
	}

	// Load into a new model
	m2 := NewModel(context.Background(), cfg)
	if err := m2.loadSession("test-session-123"); err != nil {
		t.Fatalf("failed to load session: %v", err)
	}

	// Verify loaded properties
	if m2.sessionID != "test-session-123" {
		t.Errorf("expected session ID 'test-session-123', got %q", m2.sessionID)
	}
	if m2.collection != "test-collection" {
		t.Errorf("expected collection 'test-collection', got %q", m2.collection)
	}
	if m2.searchLimit != 42 {
		t.Errorf("expected searchLimit 42, got %d", m2.searchLimit)
	}
	if len(m2.history) != 2 {
		t.Errorf("expected history length 2, got %d", len(m2.history))
	}
	if m2.history[0].Content != "Hello session test" {
		t.Errorf("expected turn content 'Hello session test', got %q", m2.history[0].Content)
	}
	if len(m2.history[1].References) != 1 {
		t.Errorf("expected 1 reference, got %d", len(m2.history[1].References))
	}
	if len(m2.inputHistory) != 1 || m2.inputHistory[0] != "Hello session test" {
		t.Errorf("expected inputHistory to contain ['Hello session test'], got %v", m2.inputHistory)
	}
	if m2.historyIndex != 1 {
		t.Errorf("expected historyIndex to be 1, got %d", m2.historyIndex)
	}
}

func TestMouseLeakFilter(t *testing.T) {
	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)

	// Send an SGR mouse scroll sequence: \x1b[<65;258;21M
	sequence := []tea.KeyMsg{
		{Type: tea.KeyEscape},
		{Type: tea.KeyRunes, Runes: []rune{'['}},
		{Type: tea.KeyRunes, Runes: []rune{'<'}},
		{Type: tea.KeyRunes, Runes: []rune{'6'}},
		{Type: tea.KeyRunes, Runes: []rune{'5'}},
		{Type: tea.KeyRunes, Runes: []rune{';'}},
		{Type: tea.KeyRunes, Runes: []rune{'2'}},
		{Type: tea.KeyRunes, Runes: []rune{'5'}},
		{Type: tea.KeyRunes, Runes: []rune{'8'}},
		{Type: tea.KeyRunes, Runes: []rune{';'}},
		{Type: tea.KeyRunes, Runes: []rune{'2'}},
		{Type: tea.KeyRunes, Runes: []rune{'1'}},
		{Type: tea.KeyRunes, Runes: []rune{'M'}},
	}

	for _, msg := range sequence {
		model, _ := m.Update(msg)
		m = model.(*Model)
	}

	// The textinput value should remain empty (none of the leaked characters should have entered it)
	if m.textInput.Value() != "" {
		t.Errorf("expected textinput to remain empty, but got %q", m.textInput.Value())
	}
}

func TestEmptySessionNotSaved(t *testing.T) {
	tempDir := t.TempDir()
	t.Setenv("HOME", tempDir)

	cfg := Config{DefaultCollection: "default"}
	m := NewModel(context.Background(), cfg)
	m.sessionID = "test-empty-session"

	// Add only a system/command turn, but no user prompt
	m.history = []ConversationTurn{
		{Role: "system", Content: "Command executed: help"},
	}

	if err := m.saveSession(); err != nil {
		t.Fatalf("saveSession returned unexpected error: %v", err)
	}

	// Verify that the file was not created
	dir, err := GetSessionsDir()
	if err != nil {
		t.Fatalf("failed to get sessions dir: %v", err)
	}
	filePath := filepath.Join(dir, "test-empty-session.json")
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Errorf("expected session file %s to not exist, but it was found", filePath)
	}
}

func TestTruncateMiddle(t *testing.T) {
	tests := []struct {
		input  string
		maxLen int
		expect string
	}{
		{"hello.txt", 10, "hello.txt"},
		{"abcdefghij.txt", 10, "abc...txt"},
		{"abcdefghijkl.txt", 12, "abcd....txt"},
		{"abcdef", 3, "abc"},
		{"abcdef", 5, "ab..."},
	}

	for _, tc := range tests {
		got := truncateMiddle(tc.input, tc.maxLen)
		if got != tc.expect {
			t.Errorf("truncateMiddle(%q, %d): expected %q, got %q", tc.input, tc.maxLen, tc.expect, got)
		}
	}
}

func TestFormatReferencesNegativeScores(t *testing.T) {
	p := rag.QdrantPoint{
		ID:    "1",
		Score: -0.2673,
		Payload: map[string]interface{}{
			"file_name":   "Desert.md",
			"chunk_index": float64(262),
			"text":        "Sun was setting.",
		},
	}

	out := formatReferences([]rag.QdrantPoint{p}, 40)
	if !strings.Contains(out, "Score: -0.2673") {
		t.Errorf("expected score formatting for negative value -0.2673, but output was:\n%s", out)
	}
	if !strings.Contains(out, "Desert.md") {
		t.Errorf("expected output to contain Desert.md, but got:\n%s", out)
	}
}

func TestLongPromptAndUserTurnWrap(t *testing.T) {
	m := NewModel(context.Background(), Config{})
	model, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	m = model.(*Model)
	longText := strings.Repeat("long-prompt ", 40)
	longUnbrokenText := strings.Repeat("https://example.com/a-very-long-path/", 20)
	model, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(longText), Paste: true})
	m = model.(*Model)

	if m.textInput.Height() < 2 {
		t.Fatalf("long prompt should occupy multiple rows, got height %d", m.textInput.Height())
	}
	inputView := m.textInput.View()
	if strings.Count(inputView, "\n") < 2 {
		t.Fatalf("long prompt view should contain wrapped rows, got:\n%s", inputView)
	}
	if strings.Count(inputView, "❯") != 1 {
		t.Fatalf("wrapped prompt should show one prompt marker, got:\n%s", inputView)
	}
	if strings.Count(m.renderFooter(), "\n") < 2 {
		t.Fatalf("footer should grow with the pasted prompt, got:\n%s", m.renderFooter())
	}
	if !strings.Contains(m.View(), "long-prompt") {
		t.Fatalf("full model view lost pasted prompt")
	}

	m.textInput.Reset()
	m.history = []ConversationTurn{{Role: "user", Content: longText}}
	m.updateViewport()
	if strings.Count(m.viewport.View(), "❯ You:") != 1 {
		t.Fatalf("user turn should keep one label, got:\n%s", m.viewport.View())
	}
	if !strings.Contains(m.viewport.View(), "long-prompt") {
		t.Fatalf("wrapped user turn lost its content")
	}
	if strings.Count(m.viewport.View(), "\n") < 2 {
		t.Fatalf("long user turn should wrap onto multiple rows, got:\n%s", m.viewport.View())
	}

	m.history = []ConversationTurn{{Role: "user", Content: longUnbrokenText}}
	m.updateViewport()
	if strings.Count(m.viewport.View(), "\n") < 2 {
		t.Fatalf("unbroken pasted user turn should hard-wrap onto multiple rows, got:\n%s", m.viewport.View())
	}

	m.history = nil
	m.lastQuery = longUnbrokenText
	m.state = stateEmbedding
	m.updateViewport()
	if strings.Count(m.viewport.View(), "❯ You:") != 1 || strings.Count(m.viewport.View(), "\n") < 2 {
		t.Fatalf("live user query should hard-wrap onto multiple rows, got:\n%s", m.viewport.View())
	}
}
