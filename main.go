package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	// Parse version flags
	versionFlag := flag.Bool("version", false, "Print version information and exit")
	vFlag := flag.Bool("v", false, "Print version information and exit")
	searchCapFlag := flag.Int("search-cap", -1, "Maximum candidate pool for Qdrant search (-1 = no CLI override, 0 = no cap, N = cap to N)")
	flag.Parse()

	if *versionFlag || *vFlag {
		fmt.Printf("QQuestio version v%s\n", Version)
		os.Exit(0)
	}

	// Set up file-based logging to prevent stdout corruption
	if f, err := tea.LogToFile("debug.log", "qquestio"); err == nil {
		defer f.Close()
	}

	cfg, err := LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	// CLI flag takes highest precedence for the initial cap (only if explicitly set)
	if *searchCapFlag >= 0 {
		cfg.SearchCap = *searchCapFlag
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
