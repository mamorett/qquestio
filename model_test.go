package main

import (
	"context"
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
