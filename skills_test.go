package main

import (
	"context"
	"strings"
	"testing"
)

func TestParseCall(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantName string
		wantArgs string
		wantOk   bool
	}{
		{
			name:   "No call",
			input:  "Hello, how can I help you?",
			wantOk: false,
		},
		{
			name:     "Simple call",
			input:    "CALL: bash ls -la",
			wantName: "bash",
			wantArgs: "ls -la",
			wantOk:   true,
		},
		{
			name:     "Call with no args",
			input:    "CALL: bash",
			wantName: "bash",
			wantArgs: "",
			wantOk:   true,
		},
		{
			name:     "Multi-line call",
			input:    "I will list files now:\nCALL: bash ls\nDone.",
			wantName: "bash",
			wantArgs: "ls",
			wantOk:   true,
		},
		{
			name:     "Call with extra spacing",
			input:    "   CALL:   bash    ls -la   ",
			wantName: "bash",
			wantArgs: "ls -la",
			wantOk:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotName, gotArgs, gotOk := ParseCall(tc.input)
			if gotOk != tc.wantOk {
				t.Errorf("ParseOk = %v, want %v", gotOk, tc.wantOk)
			}
			if gotOk {
				if gotName != tc.wantName {
					t.Errorf("ParseName = %q, want %q", gotName, tc.wantName)
				}
				if gotArgs != tc.wantArgs {
					t.Errorf("ParseArgs = %q, want %q", gotArgs, tc.wantArgs)
				}
			}
		})
	}
}

func TestBashSkill_Execute(t *testing.T) {
	skill := BashSkill{}

	t.Run("Execute valid command", func(t *testing.T) {
		ctx := context.Background()
		output, err := skill.Execute(ctx, []byte("echo 'hello'"))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		trimmed := strings.TrimSpace(output)
		if trimmed != "hello" {
			t.Errorf("expected 'hello', got %q", trimmed)
		}
	})

	t.Run("Execute with JSON command", func(t *testing.T) {
		ctx := context.Background()
		args := []byte(`{"command": "echo 'json-hello'"}`)
		output, err := skill.Execute(ctx, args)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		trimmed := strings.TrimSpace(output)
		if trimmed != "json-hello" {
			t.Errorf("expected 'json-hello', got %q", trimmed)
		}
	})

	t.Run("Execute invalid command", func(t *testing.T) {
		ctx := context.Background()
		_, err := skill.Execute(ctx, []byte("non_existent_command_12345"))
		if err == nil {
			t.Error("expected command to fail with error, but got nil")
		}
	})
}

func TestSkillRegistry(t *testing.T) {
	r := NewSkillRegistry()
	skill, ok := r.Get("bash")
	if !ok {
		t.Fatal("expected registry to contain bash skill by default")
	}
	if skill.Name() != "bash" {
		t.Errorf("expected skill name to be 'bash', got %q", skill.Name())
	}
	if !strings.Contains(r.ForPrompt(), "bash") {
		t.Errorf("expected ForPrompt to describe bash tool, got: %s", r.ForPrompt())
	}
}
