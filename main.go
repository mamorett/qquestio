package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	// 1. Manually parse optional session flags: -c [session_id]
	var loadSession bool
	var sessionIDToLoad string

	for i := 1; i < len(os.Args); i++ {
		arg := os.Args[i]
		if arg == "-c" || arg == "--c" {
			loadSession = true
			if i+1 < len(os.Args) && !strings.HasPrefix(os.Args[i+1], "-") {
				sessionIDToLoad = os.Args[i+1]
				os.Args = append(os.Args[:i+1], os.Args[i+2:]...)
			} else {
				sessionIDToLoad = "last"
			}
			os.Args = append(os.Args[:i], os.Args[i+1:]...)
			break
		}
	}

	// Parse version flags
	versionFlag := flag.Bool("version", false, "Print version information and exit")
	vFlag := flag.Bool("v", false, "Print version information and exit")
	searchCapFlag := flag.Int("search-cap", -1, "Maximum candidate pool for Qdrant search (-1 = no CLI override, 0 = no cap, N = cap to N)")
	safeFlag := flag.Bool("safe", false, "Require user confirmation before executing any local skills/tools")
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

	if *safeFlag || os.Getenv("QQUESTIO_SKILLS_REQUIRE_CONFIRM") == "1" {
		cfg.SkillsRequireConfirm = true
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	m := NewModel(ctx, cfg)

	if loadSession {
		if sessionIDToLoad == "last" {
			lastID, err := GetLastSessionID()
			if err != nil {
				fmt.Fprintf(os.Stderr, "No previous session found: %v\n", err)
			} else {
				sessionIDToLoad = lastID
			}
		}
		if sessionIDToLoad != "last" && sessionIDToLoad != "" {
			if err := m.loadSession(sessionIDToLoad); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to load session %q: %v\n", sessionIDToLoad, err)
			} else {
				m.statusMsg = fmt.Sprintf("Loaded session %s", sessionIDToLoad)
			}
		}
	}

	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	finalModel, err := p.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal: %v\n", err)
		os.Exit(1)
	}

	if fm, ok := finalModel.(*Model); ok {
		if !fm.hasUserPrompt() {
			fmt.Printf("\n👋 Thank you for using QQuestio!\n\n")
		} else {
			if err := fm.saveSession(); err != nil {
				fmt.Fprintf(os.Stderr, "\n⚠️  Warning: Failed to save session: %v\n", err)
			} else {
				fmt.Printf("\n👋 Thank you for using QQuestio!\n")
				fmt.Printf("💾 Session successfully saved: %s\n", fm.sessionID)
				fmt.Printf("🔄 To recall this session, run:      ./qquestio -c %s\n", fm.sessionID)
				fmt.Printf("⏮️  To resume the most recent session: ./qquestio -c\n\n")
			}
		}
	}
}
