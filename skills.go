package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"sort"
	"strings"
)

// Skill defines a local tool that can be executed during the generation loop.
type Skill interface {
	// Name returns the unique identifier for this skill.
	Name() string

	// Description returns a human-readable summary for LLM tool-use prompting.
	Description() string

	// Execute runs the skill with the given JSON arguments and returns the result.
	Execute(ctx context.Context, args []byte) (string, error)
}

type SkillRegistry struct {
	skills map[string]Skill
}

func NewSkillRegistry() SkillRegistry {
	r := SkillRegistry{skills: make(map[string]Skill)}
	r.Register(BashSkill{})
	return r
}

func (r *SkillRegistry) Register(s Skill) {
	r.skills[s.Name()] = s
}

func (r *SkillRegistry) Get(name string) (Skill, bool) {
	s, ok := r.skills[name]
	return s, ok
}

func (r *SkillRegistry) List() []Skill {
	list := make([]Skill, 0, len(r.skills))
	for _, s := range r.skills {
		list = append(list, s)
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].Name() < list[j].Name()
	})
	return list
}

// ForPrompt generates the tool-use system prompt fragment describing all registered skills.
func (r *SkillRegistry) ForPrompt() string {
	if len(r.skills) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("Available tools:\n")
	for _, s := range r.List() {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", s.Name(), s.Description()))
	}
	return sb.String()
}

// ParseCall parses a tool call string of the format "CALL: <name> <args>".
// Returns (name, args, ok).
func ParseCall(output string) (string, string, bool) {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "CALL:") {
			callText := strings.TrimSpace(strings.TrimPrefix(line, "CALL:"))
			parts := strings.Fields(callText)
			if len(parts) > 0 {
				name := parts[0]
				args := ""
				nameIdx := strings.Index(callText, name)
				if nameIdx != -1 {
					args = strings.TrimSpace(callText[nameIdx+len(name):])
				}
				return name, args, true
			}
		}
	}
	return "", "", false
}

// BashSkill executes bash commands.
type BashSkill struct{}

func (BashSkill) Name() string {
	return "bash"
}

func (BashSkill) Description() string {
	return "Execute a bash command and return stdout/stderr"
}

func (BashSkill) Execute(ctx context.Context, args []byte) (string, error) {
	var commandObj struct {
		Command string `json:"command"`
	}
	var cmdStr string
	if err := json.Unmarshal(args, &commandObj); err == nil && commandObj.Command != "" {
		cmdStr = commandObj.Command
	} else {
		cmdStr = string(args)
	}

	cmdStr = strings.TrimSpace(cmdStr)
	if cmdStr == "" {
		return "", fmt.Errorf("empty command")
	}

	// Try /bin/bash first, then fallback to /bin/sh, then fallback to PATH lookup "bash"
	shellPath := "/bin/bash"
	if _, err := exec.LookPath(shellPath); err != nil {
		shellPath = "/bin/sh"
		if _, err := exec.LookPath(shellPath); err != nil {
			shellPath = "bash"
		}
	}

	cmd := exec.CommandContext(ctx, shellPath, "-c", cmdStr)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	output := stdout.String()
	if stderr.Len() > 0 {
		if output != "" {
			output += "\n"
		}
		output += "Stderr: " + stderr.String()
	}

	return output, err
}
