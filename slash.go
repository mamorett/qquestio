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
					reason: "Usage: /limit <1-100>",
					stage:  "slash",
				}
			}
			n, err := strconv.Atoi(args[0])
			if err != nil || n < 1 || n > 100 {
				return appErrMsg{
					err:    fmt.Errorf("/limit must be an integer between 1 and 100"),
					reason: "Limit must be 1-100",
					stage:  "slash",
				}
			}
			m.searchLimit = n
			return slashResultMsg{feedback: fmt.Sprintf("Search limit → %d", n)}

		case "/mode":
			if len(args) != 1 {
				return appErrMsg{
					err:    fmt.Errorf("/mode requires exactly 1 argument"),
					reason: "Usage: /mode <strict|hybrid>",
					stage:  "slash",
				}
			}
			newMode := strings.ToLower(args[0])
			if newMode != "strict" && newMode != "hybrid" {
				return appErrMsg{
					err:    fmt.Errorf("/mode must be 'strict' or 'hybrid'"),
					reason: "Mode must be 'strict' or 'hybrid'",
					stage:  "slash",
				}
			}
			m.ragMode = newMode
			return slashResultMsg{feedback: fmt.Sprintf("RAG mode → %s", newMode)}

		case "/filter":
			if len(args) == 0 || args[0] == "clear" {
				m.filterKey = ""
				m.filterValue = ""
				return slashResultMsg{feedback: "Active metadata filter cleared"}
			}
			var key string
			var val string
			if len(args) == 1 {
				key = "*"
				val = args[0]
			} else {
				key = args[0]
				val = strings.Join(args[1:], " ")
			}
			if strings.HasPrefix(val, "\"") && strings.HasSuffix(val, "\"") && len(val) >= 2 {
				val = val[1 : len(val)-1]
			} else if strings.HasPrefix(val, "'") && strings.HasSuffix(val, "'") && len(val) >= 2 {
				val = val[1 : len(val)-1]
			}
			m.filterKey = key
			m.filterValue = val
			if key == "*" {
				return slashResultMsg{feedback: fmt.Sprintf("Filter applied → [any document field] = %s", val)}
			}
			return slashResultMsg{feedback: fmt.Sprintf("Filter applied → %s = %s", key, val)}

		case "/rerank":
			if len(args) != 1 {
				return appErrMsg{
					err:    fmt.Errorf("/rerank requires 'on' or 'off'"),
					reason: "Usage: /rerank <on|off>",
					stage:  "slash",
				}
			}
			val := strings.ToLower(args[0])
			if val == "on" {
				m.disableReranker = false
				return slashResultMsg{feedback: "Reranker enabled"}
			} else if val == "off" {
				m.disableReranker = true
				return slashResultMsg{feedback: "Reranker disabled (bypassed)"}
			} else {
				return appErrMsg{
					err:    fmt.Errorf("/rerank must be 'on' or 'off'"),
					reason: "Usage: /rerank <on|off>",
					stage:  "slash",
				}
			}

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
				"  /limit <1-100>      - Set the number of context documents to retrieve\n" +
				"  /filter [key] <val> - Filter search (e.g. '/filter file_name guide.txt' or '/filter guide.txt')\n" +
				"  /rerank <on|off>    - Enable/disable the reranker step\n" +
				"  /mode <strict|hybrid>- Switch RAG mode (strict closed-book vs hybrid general-knowledge)\n" +
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
