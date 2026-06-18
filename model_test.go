package main

import (
	"context"
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
