package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/atotto/clipboard"
	tea "github.com/charmbracelet/bubbletea"
)

// handleSlashCmd parses and dispatches slash commands.
// Mutations happen inside the returned tea.Cmd closure which captures the *Model pointer.
func (m *Model) handleSlashCmd(raw string) tea.Cmd {
	return func() tea.Msg {
		parts := strings.Fields(raw)
		if len(parts) == 0 {
			return appErrMsg{
				err:    fmt.Errorf("empty command"),
				reason: "Command cannot be empty",
				stage:  "slash",
			}
		}
		cmd := parts[0]
		args := parts[1:]

		switch cmd {
		case "/collection":
			if len(args) != 1 {
				return appErrMsg{
					err:    fmt.Errorf("/collection requires exactly 1 argument"),
					reason: "Usage: /collection <name>",
					stage:  "slash",
				}
			}
			m.collection = args[0]
			return slashResultMsg{feedback: fmt.Sprintf("Collection → %s", args[0])}

		case "/limit":
			if len(args) != 1 {
				return appErrMsg{
					err:    fmt.Errorf("/limit requires exactly 1 argument"),
					reason: "Usage: /limit <1-20>",
					stage:  "slash",
				}
			}
			n, err := strconv.Atoi(args[0])
			if err != nil || n < 1 || n > 20 {
				return appErrMsg{
					err:    fmt.Errorf("/limit must be an integer between 1 and 20"),
					reason: "Limit must be 1-20",
					stage:  "slash",
				}
			}
			m.searchLimit = n
			return slashResultMsg{feedback: fmt.Sprintf("Search limit → %d", n)}

		case "/system":
			if len(args) == 0 {
				return appErrMsg{
					err:    fmt.Errorf("/system requires a prompt message"),
					reason: "Usage: /system <prompt>",
					stage:  "slash",
				}
			}
			m.systemPrompt = strings.Join(args, " ")
			return slashResultMsg{feedback: "System prompt updated"}

		case "/copy":
			if len(args) > 0 && args[0] == "all" {
				return m.copyAllConversationCmd()()
			}
			var lastResponse string
			for i := len(m.history) - 1; i >= 0; i-- {
				if m.history[i].Role == "assistant" {
					lastResponse = m.history[i].Content
					break
				}
			}
			if lastResponse == "" {
				return appErrMsg{
					err:    fmt.Errorf("no response to copy"),
					reason: "No response in history to copy",
					stage:  "slash",
				}
			}
			if err := clipboard.WriteAll(lastResponse); err != nil {
				return appErrMsg{
					err:    err,
					reason: "Failed to write to clipboard",
					stage:  "slash",
				}
			}
			return slashResultMsg{feedback: "Copied last response to clipboard"}

		case "/save", "/write":
			if len(args) == 0 {
				return appErrMsg{
					err:    fmt.Errorf("/save requires at least a filename"),
					reason: "Usage: /save <filename.md> OR /save all <filename.md>",
					stage:  "slash",
				}
			}
			var filename string
			var saveAll bool
			if args[0] == "all" {
				if len(args) < 2 {
					return appErrMsg{
						err:    fmt.Errorf("/save all requires a filename"),
						reason: "Usage: /save all <filename.md>",
						stage:  "slash",
					}
				}
				saveAll = true
				filename = args[1]
			} else {
				filename = args[0]
			}
			if saveAll {
				return m.saveAllConversationCmd(filename)()
			} else {
				return m.saveLastResponseCmd(filename)()
			}

		case "/quit":
			return quitMsg{}

		case "/help":
			helpText := "Available Slash Commands:\n" +
				"  /collection <name>  - Switch the active Qdrant collection\n" +
				"  /limit <1-20>       - Set the number of context documents to retrieve\n" +
				"  /system <prompt>    - Update the custom LLM system prompt\n" +
				"  /copy               - Copy the last assistant response to the clipboard\n" +
				"  /copy all           - Copy the entire conversation transcript to the clipboard\n" +
				"  /save <file.md>     - Write the last response directly to a markdown file\n" +
				"  /save all <file.md> - Write the entire conversation history to a markdown file\n" +
				"  /help               - Show this help information\n" +
				"  /quit               - Exit the application\n\n" +
				"Keyboard Shortcuts:\n" +
				"  Ctrl+C              - Quit the application\n" +
				"  Double Escape       - Stop prompt generation (cancel request)\n" +
				"  Ctrl+Y              - Copy the last assistant response to the clipboard\n" +
				"  Ctrl+R              - Toggle Markdown rendering vs. Raw Source\n" +
				"  Up / Down           - Navigate prompt history"
			return systemLogMsg{
				content:  helpText,
				feedback: "Showing help menu",
			}

		default:
			return appErrMsg{
				err:    fmt.Errorf("unknown command: %s", cmd),
				reason: fmt.Sprintf("Unknown command: %s", cmd),
				stage:  "slash",
			}
		}
	}
}
