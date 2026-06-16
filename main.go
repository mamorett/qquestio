package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	// Set up file-based logging to prevent stdout corruption
	if f, err := tea.LogToFile("debug.log", "qquestio"); err == nil {
		defer f.Close()
	}

	cfg, err := LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	m := NewModel(ctx, cfg)
	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Fatal: %v\n", err)
		os.Exit(1)
	}
}
