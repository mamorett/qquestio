package main

import (
	"context"
	"fmt"
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
	return SkillRegistry{skills: make(map[string]Skill)}
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

// BashSkill executes bash commands.
type BashSkill struct{}

func (BashSkill) Name() string {
	return "bash"
}

func (BashSkill) Description() string {
	return "Execute a bash command and return stdout/stderr"
}

func (BashSkill) Execute(ctx context.Context, args []byte) (string, error) {
	// Stub implementation for now
	return "bash execution not implemented in this phase", nil
}
