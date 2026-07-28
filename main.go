package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"qquestio/internal/rag"
)

var (
	cliSearchCap = -1
	cliSafe      = false
	debugFlag    = false
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
	debugFlagPtr := flag.Bool("debug", false, "Enable debug logging to file")
	searchCapFlag := flag.Int("search-cap", -1, "Maximum candidate pool for Qdrant search (-1 = no CLI override, 0 = no cap, N = cap to N)")
	safeFlag := flag.Bool("safe", false, "Require user confirmation before executing any local skills/tools")
	confFlag := flag.String("conf", "", "Configuration profile to load from config file")
	flag.Parse()

	cliSearchCap = *searchCapFlag
	cliSafe = *safeFlag
	debugFlag = *debugFlagPtr

	if *versionFlag || *vFlag {
		fmt.Printf("QQuestio version v%s\n", Version)
		os.Exit(0)
	}

	// Only enable file-based logging when --debug flag or QQUESTIO_DEBUG=1 is set
	// This prevents unbounded log growth and protects user privacy
	debugEnabled := debugFlag || os.Getenv("QQUESTIO_DEBUG") == "1"
	if debugEnabled {
		rag.VerboseLogging = true
		if f, err := tea.LogToFile(getLogFilePath(), "qquestio"); err == nil {
			defer f.Close()
		}
	}

	cfg, err := LoadConfig(*confFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
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

// getLogFilePath returns the path for debug log files.
// Logs are written to the cache directory instead of CWD to prevent
// polluting the working directory and to centralize log management.
func getLogFilePath() string {
	// Use the same cache directory as the RAG package
	if cacheDir := rag.CacheDir(); cacheDir != "" {
		return cacheDir + "/qquestio-debug.log"
	}
	// Fallback to current directory if cache dir is unavailable
	return "debug.log"
}
