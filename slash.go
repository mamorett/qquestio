package main

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	tea "github.com/charmbracelet/bubbletea"

	"qquestio/internal/rag"
)

// handleSlashCmd parses and dispatches slash commands.
// Mutations happen inside the returned tea.Cmd closure which captures the *Model pointer.
// Note: This pattern of direct Model mutation inside command closures is safe because
// Bubble Tea dispatches and processes all messages serially.
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

		case "/conf":
			if len(args) == 0 {
				names, defaultConf, err := GetAvailableConfigs()
				if err != nil {
					return appErrMsg{
						err:    err,
						reason: "Failed to read configuration profiles",
						stage:  "slash",
					}
				}
				active := m.cfg.ActiveConfigName
				if active == "" {
					active = "(none/default)"
				}
				var details string
				if len(names) > 0 {
					details = fmt.Sprintf("Active configuration: %s\nDefault configuration: %s\nAvailable configurations: %s", active, defaultConf, strings.Join(names, ", "))
				} else {
					details = fmt.Sprintf("Active configuration: %s\nNo configuration profiles defined in configurations block.", active)
				}
				return systemLogMsg{
					content:  details,
					feedback: fmt.Sprintf("Active configuration: %s", active),
				}
			}
			if len(args) > 1 {
				return appErrMsg{
					err:    fmt.Errorf("/conf requires at most 1 argument"),
					reason: "Usage: /conf [config_name]",
					stage:  "slash",
				}
			}

			newCfg, err := LoadConfig(args[0])
			if err != nil {
				return appErrMsg{
					err:    err,
					reason: fmt.Sprintf("Failed to load configuration %q: %v", args[0], err),
					stage:  "slash",
				}
			}

			m.cfg = newCfg
			m.collection = newCfg.DefaultCollection
			m.searchCap = newCfg.SearchCap
			m.rerankerPool = newCfg.RerankerPool
			rag.HTTPTimeout = time.Duration(newCfg.HTTPTimeoutSeconds) * time.Second
			rag.QdrantVectorName = newCfg.QdrantVectorName

			return slashResultMsg{feedback: fmt.Sprintf("Switched to configuration → %s", args[0])}

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
		case "/search":
			if len(args) == 0 {
				return slashResultMsg{feedback: fmt.Sprintf("Search mode → %s", m.searchMode)}
			}
			arg := strings.ToLower(strings.TrimSpace(args[0]))
			switch arg {
			case "auto":
				m.searchMode = "auto"
				return slashResultMsg{feedback: "Search mode → auto (HNSW vector search when cap > 0, exact search when cap = 0)"}
			case "exact":
				m.searchMode = "exact"
				return slashResultMsg{feedback: "Search mode → exact (forces server-side brute-force exact search — maximum precision, slower)"}
			case "local":
				m.searchMode = "local"
				return slashResultMsg{feedback: "Search mode → local (client-side brute-force search using local corpus cache)"}
			default:
				return appErrMsg{
					err:    fmt.Errorf("/search must be 'auto', 'exact', or 'local'"),
					reason: "Usage: /search <auto|exact|local>",
					stage:  "slash",
				}
			}

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

		case "/cap":
			if len(args) == 0 {
				// Show current cap + mode
				modeStr := m.searchMode
				if modeStr == "" {
					modeStr = "auto"
				}
				capStr := "none"
				if m.searchCap > 0 {
					capStr = fmt.Sprintf("%d", m.searchCap)
				}
				return slashResultMsg{feedback: fmt.Sprintf("Search cap → %s | mode → %s", capStr, modeStr)}
			}
			arg := strings.ToLower(strings.TrimSpace(args[0]))
			// Mode commands first (do not change the numeric cap)
			switch arg {
			case "auto":
				m.searchMode = "auto"
				return slashResultMsg{feedback: "Search mode → auto (server-side brute-force when cap=0, HNSW when cap>0)"}
			case "exact":
				m.searchMode = "exact"
				return slashResultMsg{feedback: "Search mode → exact (server-side brute-force via params.exact=true — MAX SPEED, full Qdrant CPU)"}
			case "local":
				m.searchMode = "local"
				return slashResultMsg{feedback: "Search mode → local (client-side brute-force on all local CPU cores — fallback for when Qdrant refuses params.exact=true)"}
			}
			if arg == "off" || arg == "none" || arg == "unlimited" {
				m.searchCap = 0
				return slashResultMsg{feedback: "Search cap cleared (searches full corpus via Qdrant brute-force by default)"}
			}
			n, err := strconv.Atoi(args[0])
			if err != nil || n < 1 {
				return appErrMsg{
					err:    fmt.Errorf("/cap requires a positive integer, 'off', or a mode: 'auto' | 'exact' | 'local'"),
					reason: "Usage: /cap <N> | /cap off | /cap auto | /cap exact | /cap local",
					stage:  "slash",
				}
			}
			m.searchCap = n
			return slashResultMsg{feedback: fmt.Sprintf("Search cap → %d (HNSW candidate pool limited to top-%d before returning %d docs)", n, n, m.searchLimit)}

		case "/cache":
			if len(args) == 0 || args[0] == "status" {
				info, err := rag.CacheInfo(m.collection)
				if err != nil {
					return appErrMsg{err: err, reason: "Failed to read cache info", stage: "slash"}
				}
				dir := rag.CacheDir()
				return systemLogMsg{
					content:  fmt.Sprintf("Cache directory: %s\nCollection: %s\nStatus: %s", dir, m.collection, info),
					feedback: fmt.Sprintf("Cache status for %s", m.collection),
				}
			}
			if args[0] == "refresh" {
				m.cacheForceRefresh = true
				return slashResultMsg{feedback: "Cache refresh armed: next full-corpus query will re-scroll Qdrant"}
			}
			if args[0] == "warmup" {
				return warmupCacheMsg{}
			}
			if args[0] == "clear" {
				if err := rag.DeleteCorpusCache(m.collection); err != nil {
					return appErrMsg{err: err, reason: "Failed to delete cache", stage: "slash"}
				}
				m.cacheInfo = ""
				return slashResultMsg{feedback: fmt.Sprintf("Cache cleared for %s", m.collection)}
			}
			if args[0] == "dir" {
				return slashResultMsg{feedback: fmt.Sprintf("Cache dir: %s", rag.CacheDir())}
			}
			return appErrMsg{
				err:    fmt.Errorf("unknown /cache subcommand: %s", args[0]),
				reason: "Usage: /cache [status|refresh|warmup|clear|dir]",
				stage:  "slash",
			}

		case "/expand":
			if len(args) == 0 {
				// Show current value.
				if m.searchExpand == 0 {
					return slashResultMsg{feedback: "Context expand → off (top-N only, no adjacent chunks)"}
				}
				return slashResultMsg{feedback: fmt.Sprintf("Context expand → ±%d adjacent chunks per match", m.searchExpand)}
			}
			arg := strings.ToLower(strings.TrimSpace(args[0]))
			if arg == "off" || arg == "none" || arg == "0" {
				m.searchExpand = 0
				return slashResultMsg{feedback: "Context expand → off (legacy top-N only)"}
			}
			n, err := strconv.Atoi(args[0])
			if err != nil || n < 0 {
				return appErrMsg{
					err:    fmt.Errorf("/expand requires a non-negative integer or 'off'"),
					reason: "Usage: /expand <N> | /expand off  (0 = off, 1 = ±1, 2 = ±2, ...)",
					stage:  "slash",
				}
			}
			if n > 20 {
				return appErrMsg{
					err:    fmt.Errorf("/expand value too large: %d", n),
					reason: "Expand must be between 0 and 20 (each ±N pull adds chunks and slows the query)",
					stage:  "slash",
				}
			}
			m.searchExpand = n
			if n == 0 {
				return slashResultMsg{feedback: "Context expand → off (legacy top-N only)"}
			}
			return slashResultMsg{feedback: fmt.Sprintf("Context expand → ±%d adjacent chunks per match from the same document", n)}

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

		case "/rerankerpool":
			if len(args) == 0 {
				if m.rerankerPool <= 0 {
					return slashResultMsg{feedback: "Reranker pool → auto (dynamically sized based on number of documents requested)"}
				}
				return slashResultMsg{feedback: fmt.Sprintf("Reranker pool → %d candidates", m.rerankerPool)}
			}
			arg := strings.ToLower(strings.TrimSpace(args[0]))
			if arg == "auto" || arg == "off" || arg == "none" || arg == "0" {
				m.rerankerPool = 0
				return slashResultMsg{feedback: "Reranker pool → auto (dynamically sized based on number of documents requested)"}
			}
			n, err := strconv.Atoi(args[0])
			if err != nil || n < 1 {
				return appErrMsg{
					err:    fmt.Errorf("/rerankerpool requires a positive integer, 'auto', or 'off'"),
					reason: "Usage: /rerankerpool <N> | /rerankerpool auto (0 = auto)",
					stage:  "slash",
				}
			}
			m.rerankerPool = n
			return slashResultMsg{feedback: fmt.Sprintf("Reranker pool → %d primary candidates", n)}

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
			if len(args) > 0 && (args[0] == "ref" || args[0] == "refs") {
				refs := m.getActiveReferences()
				if len(refs) == 0 {
					return appErrMsg{
						err:    fmt.Errorf("no references to copy"),
						reason: "No references in history to copy",
						stage:  "slash",
					}
				}
				refText := formatReferences(refs, 80)
				if err := clipboard.WriteAll(refText); err != nil {
					return appErrMsg{
						err:    err,
						reason: "Failed to write references to clipboard",
						stage:  "slash",
					}
				}
				return slashResultMsg{feedback: "Copied references to clipboard"}
			}

			if m.focusRef {
				refs := m.getActiveReferences()
				if len(refs) == 0 {
					return appErrMsg{
						err:    fmt.Errorf("no references to copy"),
						reason: "No references in history to copy",
						stage:  "slash",
					}
				}
				refText := formatReferences(refs, 80)
				if err := clipboard.WriteAll(refText); err != nil {
					return appErrMsg{
						err:    err,
						reason: "Failed to write references to clipboard",
						stage:  "slash",
					}
				}
				return slashResultMsg{feedback: "Copied references to clipboard"}
			}

			var lastResponse string
			for i := len(m.history) - 1; i >= 0; i-- {
				if m.history[i].Role == "assistant" {
					turn := m.history[i]
					content := turn.Content
					if turn.Reasoning != "" {
						content = "*Thinking:*\n" + turn.Reasoning + "\n\n" + content
					}
					lastResponse = content
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

		case "/compact":
			// Default: keep last 3 Q&A pairs; caller can pass a custom count.
			keepPairs := 3
			if len(args) == 1 {
				if n, err := strconv.Atoi(args[0]); err == nil && n >= 1 {
					keepPairs = n
				}
			}
			before := len(m.history)
			m.compactHistory(keepPairs)
			after := len(m.history)
			removed := before - after
			if removed == 0 {
				return slashResultMsg{feedback: fmt.Sprintf("Nothing to compact (≤%d Q&A pairs in history)", keepPairs)}
			}
			m.history = append(m.history, ConversationTurn{
				Role:    "system",
				Content: fmt.Sprintf("[ Manually compacted: %d entr%s removed, kept last %d Q&A pair(s) ]",
					removed,
					map[bool]string{true: "y", false: "ies"}[removed == 1],
					keepPairs),
			})
			m.updateViewport()
			return slashResultMsg{feedback: fmt.Sprintf("Context compacted: removed %d entries, kept last %d Q&A pair(s)", removed, keepPairs)}

		case "/clear":
			m.history = nil
			m.lastPoints = nil
			m.ragContext = ""
			m.output = ""
			m.history = append(m.history, ConversationTurn{
				Role:    "system",
				Content: "[ Conversation and references cleared ]",
			})
			m.updateViewport()
			return slashResultMsg{feedback: "Cleared context, conversation history, and references."}

		case "/quit":
			return quitMsg{}

		case "/help":
			helpText := "Available Slash Commands:\n" +
				"  /collection <name>  - Switch the active Qdrant collection\n" +
				"  /conf [name]        - View or switch the active configuration profile\n" +
				"  /limit <1-100>      - Set the number of context documents to retrieve\n" +
				"  /expand <N|off>     - ±N adjacent chunks from the same doc per match (0=off, 1=default)\n" +
				"  /cap [N|off]        - Set/clear the candidate pool cap (0/no cap = full corpus)\n" +
				"  /search <auto|exact|local> - Set vector search mode (auto = HNSW, exact = server brute force, local = cache)\n" +
				"  /cache [status|refresh|warmup|clear|dir] - Inspect or control the on-disk corpus cache\n" +
				"  /filter [key] <val> - Filter search (e.g. '/filter file_name guide.txt' or '/filter guide.txt')\n" +
				"  /rerank <on|off>    - Enable/disable the reranker step\n" +
				"  /rerankerpool <N|auto> - Set the candidate pool size for the reranker (0/auto = dynamic)\n" +
				"  /mode <strict|hybrid>- Switch RAG mode (strict closed-book vs hybrid general-knowledge)\n" +
				"  /system <prompt>    - Update the custom LLM system prompt\n" +
				"  /compact [N]        - Compact history, keeping last N Q&A pairs (default 3); auto at 85% ctx\n" +
				"  /clear              - Clear conversation history and references (keeps prompt history)\n" +
				"  /copy               - Copy the last response (or references if ref panel is focused) to the clipboard\n" +
				"  /copy ref           - Copy the last retrieved references to the clipboard\n" +
				"  /copy all           - Copy the entire conversation transcript to the clipboard\n" +
				"  /save <file.md>     - Write the last response directly to a markdown file\n" +
				"  /save all <file.md> - Write the entire conversation history to a markdown file\n" +
				"  /help               - Show this help information\n" +
				"  /quit               - Exit the application\n\n" +
				"Keyboard Shortcuts:\n" +
				"  Ctrl+C              - Ask for exit confirmation (press again or Y to confirm, Esc/N to cancel)\n" +
				"  Double Escape       - Stop prompt generation (cancel request)\n" +
				"  Tab                 - Toggle focus between conversation and references\n" +
				"  Ctrl+Y              - Copy the last response (or references if ref panel is focused) to the clipboard\n" +
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
