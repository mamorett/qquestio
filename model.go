package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"qquestio/internal/rag"
)

type appState int

const (
	stateIdle        appState = iota // Waiting for user input
	stateEmbedding                   // Generating embedding vector
	stateSearching                   // Querying Qdrant
	stateReranking                   // Reranking retrieved documents
	stateStreaming                   // Receiving LLM SSE chunks
	stateError                       // Displaying error, input still active
	stateConfirmQuit                  // Awaiting exit confirmation (first Ctrl+C pressed)
)

type ConversationTurn struct {
	Role                    string            `json:"role"`                      // "user" | "assistant" | "system"
	Content                 string            `json:"content"`                   // The text content
	References              []rag.QdrantPoint `json:"references,omitempty"`      // Retrieved context points (only for assistant responses)
	RenderedContent         string            `json:"rendered_content,omitempty"` // Cached rendered markdown
	RenderedWidth           int               `json:"rendered_width,omitempty"`   // The width at which it was rendered
	RenderedReferences      string            `json:"rendered_references,omitempty"` // Cached rendered references block
	RenderedReferencesWidth int               `json:"rendered_references_width,omitempty"` // The width at which references were rendered
}

type Model struct {
	// --- Config (immutable) ---
	cfg            Config
	sessionID      string // Unique identifier for the active session (timestamp-based)
	loadingMessage string // Random loading phrase selected per query
	leakState      int    // State machine tracker to filter out leaked terminal SGR mouse sequences

	// --- Runtime state (mutable via slash commands) ---
	collection        string // Active Qdrant collection (init: cfg.DefaultCollection)
	searchLimit       int    // Number of Qdrant results (default: 5)
	searchCap         int    // Hard upper bound on candidate pool for Qdrant search (0 = no cap, search full corpus)
	rerankerPool      int    // Primary candidate pool size for reranker (0 = auto)
	searchExpand      int    // ±N adjacent chunks to expand each top match from the same document (0 = disabled, 1 = default)
	searchMode        string // "auto" (default), "exact" (force server-side), or "local" (client-side brute-force using all CPU cores)
	systemPrompt      string // Custom system prompt (default: built-in RAG prompt)
	ragMode           string // RAG mode: "strict" or "hybrid"
	filterKey         string // Active filter metadata key
	filterValue       string // Active filter metadata value
	disableReranker   bool   // Toggle to bypass reranker step
	cacheForceRefresh bool   // When true, the next full-corpus search re-scrolls Qdrant (set by /cache refresh, auto-cleared after one use)
	cacheInfo         string // Last known cache summary for header display (e.g. "✓ 1.2M pts, 2m ago")

	// --- FSM ---
	state appState

	// --- UI components ---
	textInput   textinput.Model
	viewport    viewport.Model
	refViewport viewport.Model
	focusRef    bool // True if focused on the references panel
	spinner     spinner.Model
	statusMsg   string // Displayed in header bar

	// --- Conversation ---
	history       []ConversationTurn // Full conversation history including system info
	output        string             // Accumulated LLM response text for current turn
	showRawSource bool               // Toggle between glamour-rendered markdown and raw markdown source

	// --- Pipeline transient ---
	lastQuery     string            // The user query that started the pipeline
	exactPhrase   string            // Parsed exact phrase (if any) to bypass embedding and force string match
	ragContext    string            // Retrieved text from Qdrant (current turn)
	lastPoints    []rag.QdrantPoint // Retrieved points from Qdrant (current turn)
	cancelRequest context.CancelFunc
	streamReader  *rag.SSEReader
	escCount      int  // consecutive Esc presses
	stoppedByUser bool // explicit user abort flag

	// --- Qdrant Collection Stats ---
	qdrantPoints  int
	qdrantVectors int
	qdrantStatus  string
	qdrantInfoErr error

	// --- Prompt History ---
	inputHistory []string
	historyIndex int
	tempInput    string

	// --- Skills ---
	skills SkillRegistry

	// --- Dimensions ---
	width  int
	height int

	// --- Root context ---
	ctx context.Context
}

// estimateContextTokens returns a rough token count of what is ACTUALLY sent
// to the LLM: turn content (Q&A text) + the current query's retrieved chunks.
// Does NOT count stored References from past turns because those are no longer
// injected into the LLM prompt (they live only in the references panel).
func (m *Model) estimateContextTokens() int {
	total := 0
	for _, turn := range m.history {
		total += len(turn.Content) / 4
	}
	// Count the current query's retrieved context (not yet in history)
	for _, pt := range m.lastPoints {
		total += len(pt.ExtractPrimaryText()) / 4
	}
	return total
}

// compactHistory summarizes the oldest conversation turns beyond the keepPairs
// threshold, replacing them with a single system-turn summary. This prevents
// the conversation from growing unboundedly and keeps LLM context usage in check.
// keepPairs is the number of recent Q&A pairs (user+assistant) to keep intact.
func (m *Model) compactHistory(keepPairs int) {
	if keepPairs < 1 {
		keepPairs = 1
	}
	// Find the index of each user turn (each represents a Q&A pair boundary).
	var userIdx []int
	for i, t := range m.history {
		if t.Role == "user" {
			userIdx = append(userIdx, i)
		}
	}
	if len(userIdx) <= keepPairs {
		return // nothing old enough to compact
	}
	// cutoff is the index of the first turn we keep.
	cutoff := userIdx[len(userIdx)-keepPairs]

	// Build a brief plain-text summary of the compacted turns.
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("[ Context compacted — %d older Q&A pair(s) summarized ]\n\n", len(userIdx)-keepPairs))
	for i := 0; i < cutoff; i++ {
		t := m.history[i]
		switch t.Role {
		case "user":
			sb.WriteString("Q: " + t.Content + "\n")
		case "assistant":
			content := t.Content
			if len(content) > 300 {
				content = content[:300] + "…"
			}
			sb.WriteString("A: " + content + "\n\n")
		}
	}
	summary := ConversationTurn{
		Role:    "system",
		Content: strings.TrimSpace(sb.String()),
	}
	m.history = append([]ConversationTurn{summary}, m.history[cutoff:]...)
}

// maybeAutoCompact fires context compaction when the estimated token count
// exceeds 85% of the configured ContextLimit. No-op when ContextLimit == 0.
func (m *Model) maybeAutoCompact() {
	if m.cfg.ContextLimit <= 0 {
		return
	}
	threshold := int(float64(m.cfg.ContextLimit) * 0.85)
	if m.estimateContextTokens() <= threshold {
		return
	}
	before := len(m.history)
	m.compactHistory(3)
	after := len(m.history)
	m.history = append(m.history, ConversationTurn{
		Role: "system",
		Content: fmt.Sprintf(
			"[ Auto-compacted: %d entr%s removed — context reached ≥85%% of %s token limit ]",
			before-after,
			map[bool]string{true: "y", false: "ies"}[before-after == 1],
			formatNumber(m.cfg.ContextLimit),
		),
	})
	m.statusMsg = fmt.Sprintf("Context auto-compacted (≥85%% of %s token limit)", formatNumber(m.cfg.ContextLimit))
}

func NewModel(ctx context.Context, cfg Config) *Model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(nord8)

	ti := textinput.New()
	ti.Placeholder = "Type your query here... (Ctrl+Y to copy last response, /help for commands)"
	ti.Focus()
	ti.Prompt = " ❯ "
	ti.PromptStyle = lipgloss.NewStyle().Foreground(nord8).Bold(true)
	ti.TextStyle = lipgloss.NewStyle().Foreground(nord6)

	vp := viewport.New(0, 0)
	vp.Style = lipgloss.NewStyle().Background(nord0).Foreground(nord4)

	refVp := viewport.New(0, 0)
	refVp.Style = lipgloss.NewStyle().Background(nord0).Foreground(nord4)

	sessionID := time.Now().Format("20060102-150405")

	return &Model{
		cfg:             cfg,
		sessionID:       sessionID,
		loadingMessage:  "Thinking...",
		collection:      cfg.DefaultCollection,
		searchLimit:     10,
		searchCap:       cfg.SearchCap,
		rerankerPool:    cfg.RerankerPool,
		searchExpand:    1, // ±1 adjacent chunk from the same document by default
		searchMode:      "auto",
		state:           stateIdle,
		textInput:       ti,
		viewport:        vp,
		refViewport:     refVp,
		focusRef:        false,
		spinner:         s,
		statusMsg:       "Ready",
		skills:          NewSkillRegistry(),
		ctx:             ctx,
		ragMode:         "strict",
		filterKey:       "",
		filterValue:     "",
		disableReranker: false,
	}
}

func (m *Model) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink,
		m.spinner.Tick,
		m.fetchQdrantInfoCmd(),
		m.preloadCacheInfoCmd(),
	)
}

// extractExactPhrase strips surrounding standard or smart quotes from a query.
func extractExactPhrase(raw string) string {
	s := strings.TrimSpace(raw)
	quotes := []string{"\"", "“", "”", "„", "«", "»"}

	hasPrefix := false
	for _, q := range quotes {
		if strings.HasPrefix(s, q) {
			hasPrefix = true
			break
		}
	}
	hasSuffix := false
	for _, q := range quotes {
		if strings.HasSuffix(s, q) {
			hasSuffix = true
			break
		}
	}

	if hasPrefix && hasSuffix {
		for _, q := range quotes {
			s = strings.TrimPrefix(s, q)
			s = strings.TrimSuffix(s, q)
		}
		s = strings.TrimSpace(s)
		if len(s) > 0 {
			return s
		}
	}
	return ""
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// Filter out leaked CSI/SGR mouse escape sequences that some terminals
	// emit as individual KeyMsg events instead of as tea.MouseMsg.
	//
	// Any CSI sequence starts with ESC [ (leakState 1→2) and continues with
	// parameter bytes in the range 0x30-0x3F (digits 0-9, ;, <, =, >, ?)
	// followed by a single letter terminator. Mouse sequences in both the
	// legacy X10 format (ESC [ A ; B ; C M) and the SGR format
	// (ESC [ < A ; B ; C M / m) are handled by the same generic rule:
	// once we see ESC [, swallow everything up to and including the first
	// letter we encounter.
	if keyMsg, ok := msg.(tea.KeyMsg); ok {
		if keyMsg.Type == tea.KeyEscape {
			m.leakState = 1
			// Don't return here — ESC is also used for double-Esc cancel.
		} else if m.leakState == 1 && len(keyMsg.Runes) > 0 && keyMsg.Runes[0] == '[' {
			// ESC [ seen → enter CSI sequence
			m.leakState = 2
			return m, nil
		} else if m.leakState >= 2 && len(keyMsg.Runes) > 0 {
			r := keyMsg.Runes[0]
			// CSI parameter bytes: 0x30-0x3F  (0-9, ;, <, =, >, ?)
			if r >= 0x30 && r <= 0x3F {
				return m, nil // consume parameter byte
			}
			// Any letter terminates the sequence (M, m, A, B, H, …)
			m.leakState = 0
			return m, nil // consume terminator
		} else {
			m.leakState = 0
		}
	} else if _, isMouse := msg.(tea.MouseMsg); isMouse {
		m.leakState = 0
	}

	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case tea.KeyMsg:
		if msg.Type != tea.KeyEscape {
			m.escCount = 0
		}

		// --- Quit confirmation dialog: first Ctrl+C arms it, second confirms ---
		if m.state == stateConfirmQuit {
			switch msg.Type {
			case tea.KeyCtrlC:
				if m.cancelRequest != nil {
					m.cancelRequest()
				}
				return m, tea.Quit
			case tea.KeyEscape:
				m.state = stateIdle
				m.statusMsg = "Quit cancelled"
				m.updateViewport()
				return m, nil
			case tea.KeyRunes:
				switch strings.ToLower(string(msg.Runes)) {
				case "y":
					if m.cancelRequest != nil {
						m.cancelRequest()
					}
					return m, tea.Quit
				case "n":
					m.state = stateIdle
					m.statusMsg = "Quit cancelled"
					m.updateViewport()
					return m, nil
				}
			}
			// Swallow all other keys while the dialog is visible.
			return m, nil
		}
		switch msg.Type {
		case tea.KeyEscape:
			if m.state == stateEmbedding || m.state == stateSearching || m.state == stateReranking || m.state == stateStreaming {
				m.escCount++
				if m.escCount == 2 {
					m.escCount = 0
					m.stoppedByUser = true
					if m.cancelRequest != nil {
						m.cancelRequest()
					}
					m.state = stateIdle
					m.statusMsg = "Generation stopped"
					if m.streamReader != nil {
						_ = m.streamReader.Close()
						m.streamReader = nil
					}
					m.lastQuery = ""
					m.output = ""
					m.updateViewport()
				}
				return m, nil
			}
		case tea.KeyCtrlC:
			// First Ctrl+C arms the confirmation dialog instead of quitting immediately.
			m.state = stateConfirmQuit
			m.updateViewport()
			return m, nil
		case tea.KeyCtrlY:
			cmd := m.copyLastResponseCmd()
			if cmd != nil {
				cmds = append(cmds, cmd)
			}
		case tea.KeyCtrlR:
			m.showRawSource = !m.showRawSource
			m.updateViewport()
			var mode string
			if m.showRawSource {
				mode = "Raw Source"
			} else {
				mode = "Rendered Markdown"
			}
			m.statusMsg = fmt.Sprintf("View mode → %s", mode)
		case tea.KeyTab:
			m.focusRef = !m.focusRef
			m.updateViewport()
		case tea.KeyCtrlUp:
			if m.focusRef {
				m.refViewport.LineUp(1)
			} else {
				m.viewport.LineUp(1)
			}
		case tea.KeyCtrlDown:
			if m.focusRef {
				m.refViewport.LineDown(1)
			} else {
				m.viewport.LineDown(1)
			}
		case tea.KeyPgUp:
			if m.focusRef {
				m.refViewport.HalfPageUp()
			} else {
				m.viewport.HalfPageUp()
			}
		case tea.KeyPgDown:
			if m.focusRef {
				m.refViewport.HalfPageDown()
			} else {
				m.viewport.HalfPageDown()
			}
		case tea.KeyUp:
			if len(m.inputHistory) > 0 {
				if m.historyIndex == len(m.inputHistory) {
					m.tempInput = m.textInput.Value()
				}
				if m.historyIndex > 0 {
					m.historyIndex--
					m.textInput.SetValue(m.inputHistory[m.historyIndex])
					m.textInput.SetCursor(len(m.textInput.Value()))
				}
			}
		case tea.KeyDown:
			if len(m.inputHistory) > 0 {
				if m.historyIndex < len(m.inputHistory)-1 {
					m.historyIndex++
					m.textInput.SetValue(m.inputHistory[m.historyIndex])
					m.textInput.SetCursor(len(m.textInput.Value()))
				} else if m.historyIndex == len(m.inputHistory)-1 {
					m.historyIndex++
					m.textInput.SetValue(m.tempInput)
					m.textInput.SetCursor(len(m.textInput.Value()))
				}
			}
		case tea.KeyEnter:
			if m.state == stateIdle || m.state == stateError {
				raw := strings.TrimSpace(m.textInput.Value())
				m.textInput.Reset()
				if m.state == stateError {
					m.state = stateIdle
					m.statusMsg = "Ready"
				}
				if raw == "" {
					return m, nil
				}
				// Check slash command
				if strings.HasPrefix(raw, "/") {
					return m, m.handleSlashCmd(raw)
				}
				// Save to prompt history
				m.inputHistory = append(m.inputHistory, raw)
				m.historyIndex = len(m.inputHistory)
				m.tempInput = ""

				// Start RAG pipeline
				rawClean := strings.TrimSpace(raw)
				m.lastQuery = rawClean
				m.exactPhrase = ""
				isQuote := (strings.HasPrefix(rawClean, "\"") && strings.HasSuffix(rawClean, "\"")) ||
					(strings.HasPrefix(rawClean, "“") && strings.HasSuffix(rawClean, "”")) ||
					(strings.HasPrefix(rawClean, "'") && strings.HasSuffix(rawClean, "'")) ||
					(strings.HasPrefix(rawClean, "«") && strings.HasSuffix(rawClean, "»"))
					
				if isQuote {
					runes := []rune(rawClean)
					if len(runes) > 2 {
						m.exactPhrase = string(runes[1 : len(runes)-1])
						m.lastQuery = m.exactPhrase
					}
				}
				m.output = ""
				m.lastPoints = nil
				m.ragContext = ""
				m.selectRandomLoadingMessage()
				m.state = stateEmbedding
				m.statusMsg = "Generating embedding..."
				m.updateViewport()
				cmds = append(cmds, m.generateEmbeddingCmd(rawClean))
			}
		}

	// --- Pipeline chain ---
	case embeddingMsg:
		m.state = stateSearching
		m.statusMsg = "Searching knowledge base..."
		cmds = append(cmds, m.searchQdrantCmd(msg.vector))

	case searchResultMsg:
		m.refViewport.GotoTop()
		m.maybeAutoCompact()
		if m.cfg.RerankerURL != "" && !m.disableReranker && m.exactPhrase == "" {
			m.state = stateReranking
			m.statusMsg = "Reranking retrieved documents..."
			m.updateViewport()
			cmds = append(cmds, m.rerankPointsCmd(msg))
		} else {
			m.ragContext = msg.context
			m.lastPoints = msg.points
			m.state = stateStreaming
			docCount := len(msg.points)
			m.statusMsg = fmt.Sprintf("Generating response... (%d docs retrieved)", docCount)
			cmds = append(cmds, m.startLLMStreamCmd())
		}

	case rerankResultMsg:
		m.refViewport.GotoTop()
		m.maybeAutoCompact()
		m.ragContext = msg.context
		m.lastPoints = msg.points
		m.state = stateStreaming
		docCount := len(msg.points)
		m.statusMsg = fmt.Sprintf("Generating response... (%d docs retrieved)", docCount)
		cmds = append(cmds, m.startLLMStreamCmd())

	case streamChunkMsg:
		if msg.done {
			// Check if output contains a tool call
			if name, args, ok := ParseCall(m.output); ok {
				m.statusMsg = fmt.Sprintf("Executing skill '%s'...", name)
				cleanedOutput := cleanLLMOutput(m.output)
				if cleanedOutput == "" {
					cleanedOutput = m.output
				}
				m.history = append(m.history,
					ConversationTurn{Role: "user", Content: m.lastQuery},
					ConversationTurn{Role: "assistant", Content: cleanedOutput, References: m.lastPoints},
				)
				m.updateViewport()
				cmds = append(cmds, m.executeSkillCmd(name, args))
			} else {
				// Stream complete
				m.state = stateIdle
				m.statusMsg = "Ready"
				m.history = append(m.history,
					ConversationTurn{Role: "user", Content: m.lastQuery},
					ConversationTurn{Role: "assistant", Content: cleanLLMOutput(m.output), References: m.lastPoints},
				)
				m.lastQuery = ""
				m.output = ""
				m.updateViewport()
				if m.streamReader != nil {
					_ = m.streamReader.Close()
					m.streamReader = nil
				}
			}
		} else {
			m.output += msg.content
			m.updateViewport()
			cmds = append(cmds, m.receiveStreamChunkCmd())
		}

	case skillResultMsg:
		if msg.err != nil {
			m.state = stateError
			m.statusMsg = fmt.Sprintf("Skill error: %v", msg.err)
			m.lastQuery = ""
			m.output = ""
			if m.streamReader != nil {
				_ = m.streamReader.Close()
				m.streamReader = nil
			}
			break
		}
		// Skill succeeded! Transition back to streaming to complete the answer.
		toolResponse := fmt.Sprintf("Tool %s executed. Result:\n%s", msg.name, msg.output)
		m.lastQuery = toolResponse
		m.output = ""
		m.state = stateStreaming
		m.statusMsg = fmt.Sprintf("Generating response (resumed after '%s')...", msg.name)
		cmds = append(cmds, m.startLLMStreamCmd())

	case appErrMsg:
		if m.stoppedByUser {
			m.stoppedByUser = false
			return m, nil
		}
		m.state = stateError
		m.statusMsg = fmt.Sprintf("Error [%s]: %s (%v)", msg.stage, msg.reason, msg.err)
		if msg.stage == "slash" {
			m.history = append(m.history, ConversationTurn{
				Role:    "system",
				Content: fmt.Sprintf("Command Error: %s", msg.reason),
			})
		}
		m.updateViewport()
		if m.streamReader != nil {
			_ = m.streamReader.Close()
			m.streamReader = nil
		}

	case qdrantInfoMsg:
		if msg.err != nil {
			m.qdrantInfoErr = msg.err
			m.qdrantStatus = ""
			m.qdrantPoints = 0
			m.qdrantVectors = 0
		} else {
			m.qdrantInfoErr = nil
			m.qdrantStatus = msg.status
			m.qdrantPoints = msg.pointsCount
			m.qdrantVectors = msg.vectorsCount
		}

	case systemLogMsg:
		m.history = append(m.history, ConversationTurn{Role: "system", Content: msg.content})
		m.statusMsg = msg.feedback
		m.updateViewport()

	case slashResultMsg:
		m.statusMsg = msg.feedback
		m.history = append(m.history, ConversationTurn{
			Role:    "system",
			Content: fmt.Sprintf("Command executed: %s", msg.feedback),
		})
		m.updateViewport()
		cmds = append(cmds, m.fetchQdrantInfoCmd())

	case quitMsg:
		if m.cancelRequest != nil {
			m.cancelRequest()
		}
		return m, tea.Quit

	case spinner.TickMsg:
		var spinnerCmd tea.Cmd
		m.spinner, spinnerCmd = m.spinner.Update(msg)
		cmds = append(cmds, spinnerCmd)
		if m.state == stateEmbedding || m.state == stateSearching || m.state == stateReranking || m.state == stateStreaming {
			m.updateViewport()
		}

	case searchProgressMsg:
		// Transient status update from the full-corpus search loop.
		// Just refresh the header; do not touch FSM state or history.
		m.statusMsg = msg.status
		m.updateViewport()

	case cachePreloadMsg:
		if msg.found {
			m.cacheInfo = msg.info
		}

	case warmupCacheMsg:
		// Triggered by /cache warmup. Run the scroll-based cache population
		// in a background command. The FSM stays in stateSearching so the
		// user sees progress and can Esc-Esc out.
		m.state = stateSearching
		m.statusMsg = "Warming up cache (streaming corpus from Qdrant)..."
		m.updateViewport()
		cmds = append(cmds, m.warmupCacheCmd())

	case tea.MouseMsg:
		headerH := 4
		if m.cfg.RerankerURL != "" {
			headerH = 5
		}
		footerH := 3
		if msg.Y >= headerH && msg.Y < m.height-footerH {
			refWidth := m.width / 3
			if refWidth < 20 {
				refWidth = 20
			}
			if refWidth > m.width/2 {
				refWidth = m.width/2
			}
			mainWidth := m.width - refWidth

			if msg.X < mainWidth {
				m.focusRef = false
			} else {
				m.focusRef = true
			}
			m.updateViewport()
		}

	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		headerH := 4
		if m.cfg.RerankerURL != "" {
			headerH = 5
		}
		footerH := 3

		refWidth := m.width / 3
		if refWidth < 20 {
			refWidth = 20
		}
		if refWidth > m.width/2 {
			refWidth = m.width/2
		}
		mainWidth := m.width - refWidth

		bodyHeight := m.height - headerH - footerH
		viewportHeight := bodyHeight - 2
		if viewportHeight < 1 {
			viewportHeight = 1
		}

		m.viewport.Width = mainWidth - 2
		m.viewport.Height = viewportHeight

		m.refViewport.Width = refWidth - 2
		m.refViewport.Height = viewportHeight

		m.textInput.Width = m.width - 4
		m.updateViewport()
	}

	// --- Sub-model updates (always, for cursor blink + spinner) ---
	var spinnerCmd tea.Cmd
	if _, ok := msg.(spinner.TickMsg); !ok {
		m.spinner, spinnerCmd = m.spinner.Update(msg)
		cmds = append(cmds, spinnerCmd)
	}
	// Only forward keyboard messages to the text input.
	// Forwarding MouseMsg causes mouse coordinates to be inserted as
	// printable text when the user scrolls (the terminal's mouse reporting
	// escape sequences leak into the input buffer).
	if _, isMouse := msg.(tea.MouseMsg); !isMouse {
		if _, isWindowSize := msg.(tea.WindowSizeMsg); !isWindowSize {
			var tiCmd tea.Cmd
			m.textInput, tiCmd = m.textInput.Update(msg)
			cmds = append(cmds, tiCmd)
		}
	}

	if m.state == stateStreaming || m.state == stateIdle || m.state == stateError {
		var shouldForward = true
		if keyMsg, ok := msg.(tea.KeyMsg); ok {
			if keyMsg.Type == tea.KeyUp || keyMsg.Type == tea.KeyDown {
				shouldForward = false
			}
		}
		if shouldForward {
			var vpCmd tea.Cmd
			if m.focusRef {
				m.refViewport, vpCmd = m.refViewport.Update(msg)
			} else {
				m.viewport, vpCmd = m.viewport.Update(msg)
			}
			cmds = append(cmds, vpCmd)
		}
	}

	// Aggressive cleanup: if any terminal mouse escape sequences leaked
	// through the regular event loop (which happens sometimes when scrolling
	// fast or if the terminal uses a slightly different SGR sequence), strip
	// them from the text input buffer before they render.
	// We look for patterns like `[<65;85;14M` or `[<64;85;14m` which are SGR mouse codes.
	leakRegex := regexp.MustCompile(`\[<\d+;\d+;\d+[mM]`)
	if m.tempInput != "" {
		m.tempInput = leakRegex.ReplaceAllString(m.tempInput, "")
	}
	if m.textInput.Value() != "" {
		cleaned := leakRegex.ReplaceAllString(m.textInput.Value(), "")
		if cleaned != m.textInput.Value() {
			m.textInput.SetValue(cleaned)
			m.textInput.SetCursor(len(cleaned))
		}
	}

	return m, tea.Batch(cmds...)
}

func (m *Model) View() string {
	header := m.renderHeader()

	styles := DefaultStyles()
	var mainView string
	var refView string

	if m.focusRef {
		mainView = styles.MainViewportUnfocused.Render(m.viewport.View())
		refView = styles.RefViewportFocused.Render(m.refViewport.View())
	} else {
		mainView = styles.MainViewportFocused.Render(m.viewport.View())
		refView = styles.RefViewportUnfocused.Render(m.refViewport.View())
	}

	body := lipgloss.JoinHorizontal(lipgloss.Top, mainView, refView)
	footer := m.renderFooter()
	return lipgloss.JoinVertical(lipgloss.Left, header, body, footer)
}

func (m *Model) renderHeader() string {
	styles := DefaultStyles()

	// Status text based on current state
	var statusText string
	var statusStyle lipgloss.Style
	switch m.state {
	case stateIdle:
		statusText = "IDLE"
		statusStyle = styles.HeaderStatus.Foreground(nord14)
	case stateEmbedding:
		statusText = "EMBEDDING"
		statusStyle = styles.HeaderStatus.Foreground(nord13)
	case stateSearching:
		statusText = "SEARCHING"
		statusStyle = styles.HeaderStatus.Foreground(nord13)
	case stateReranking:
		statusText = "RERANKING"
		statusStyle = styles.HeaderStatus.Foreground(nord13)
	case stateStreaming:
		statusText = "STREAMING"
		statusStyle = styles.HeaderStatus.Foreground(nord13)
	case stateError:
		statusText = "ERROR"
		statusStyle = styles.HeaderStatus.Foreground(nord11)
	}

	var indicator string
	if m.state == stateEmbedding || m.state == stateSearching || m.state == stateReranking || m.state == stateStreaming {
		indicator = m.spinner.View() + " "
	}

	// Line 1: Program Name and Status
	statusStr := fmt.Sprintf("[%s] %s", statusText, m.statusMsg)
	modeColor := nord14 // green for strict
	if m.ragMode == "hybrid" {
		modeColor = nord13 // yellow/orange for hybrid
	}
	modeText := lipgloss.NewStyle().Foreground(modeColor).Render(fmt.Sprintf("(%s)", m.ragMode))
	leftText := fmt.Sprintf(" ◉ QQuestio v%s %s  %s%s", Version, modeText, indicator, statusStyle.Render(statusStr))
	line1Left := styles.Header.Render(leftText)

	tagRender := styles.CollectionTag.Render(m.collection)

	// Calculate space to align right for Line 1
	spaceWidth := m.width - lipgloss.Width(line1Left) - lipgloss.Width(tagRender)
	if spaceWidth < 0 {
		spaceWidth = 0
	}
	spaces := strings.Repeat(" ", spaceWidth)
	bgStyle := lipgloss.NewStyle().Background(nord1)
	line1 := bgStyle.Render(line1Left + spaces + tagRender)

	// Line 2: Qdrant DB & Collection Info
	labelStyle := lipgloss.NewStyle().Foreground(nord9).Bold(true)
	valueStyle := lipgloss.NewStyle().Foreground(nord4)
	delimStyle := lipgloss.NewStyle().Foreground(nord3)

	var statsStr string
	if m.qdrantStatus != "" {
		statusColor := nord14
		if m.qdrantStatus != "green" {
			statusColor = nord13
		}
		statsStr = fmt.Sprintf("  %s  %s %s  %s %d  %s %d",
			delimStyle.Render("│"),
			labelStyle.Render("Status:"), lipgloss.NewStyle().Foreground(statusColor).Render(m.qdrantStatus),
			labelStyle.Render("Points:"), m.qdrantPoints,
			labelStyle.Render("Vectors:"), m.qdrantVectors,
		)
	} else if m.qdrantInfoErr != nil {
		statsStr = fmt.Sprintf("  %s  %s",
			delimStyle.Render("│"),
			lipgloss.NewStyle().Foreground(nord11).Render("Err: connection failed"),
		)
	} else {
		statsStr = fmt.Sprintf("  %s  %s",
			delimStyle.Render("│"),
			lipgloss.NewStyle().Foreground(nord13).Render("Fetching stats..."),
		)
	}

	modeView := lipgloss.NewStyle().Foreground(modeColor).Bold(true).Render(m.ragMode)

	// Cap display: "none" when unset (0) for full-corpus search, otherwise the integer value
	capText := "none"
	if m.searchCap > 0 {
		capText = fmt.Sprintf("%d", m.searchCap)
	}

	cacheText := "—"
	if m.cacheInfo != "" {
		cacheText = m.cacheInfo
	}

	// Expand display: "off" when 0 (legacy top-N only), otherwise "±N"
	expandText := "off"
	if m.searchExpand > 0 {
		expandText = fmt.Sprintf("±%d", m.searchExpand)
	}

	// Context usage display (when ContextLimit > 0)
	var contextStr string
	if m.cfg.ContextLimit > 0 {
		tokens := m.estimateContextTokens()
		pct := int(float64(tokens) * 100.0 / float64(m.cfg.ContextLimit))
		ctxColor := nord14 // green ≤ 70%
		if pct >= 85 {
			ctxColor = nord11 // red ≥85%
		} else if pct >= 70 {
			ctxColor = nord13 // yellow 70–84%
		}
		contextStr = fmt.Sprintf("  %s  %s %s/%s (%d%%)",
			delimStyle.Render("│"),
			labelStyle.Render("Ctx:"),
			lipgloss.NewStyle().Foreground(ctxColor).Render(formatNumber(tokens)),
			formatNumber(m.cfg.ContextLimit),
			pct,
		)
	}

	qdrantInfo := fmt.Sprintf(" %s %s  %s  %s %s  %s %s %d  %s  %s %s  %s  %s %s  %s  %s %s  %s  %s %s%s%s",
		labelStyle.Render("DB:"), valueStyle.Render(m.cfg.QdrantURL),
		delimStyle.Render("│"),
		labelStyle.Render("Col:"), valueStyle.Render(m.collection),
		delimStyle.Render("│"),
		labelStyle.Render("Limit:"), m.searchLimit,
		delimStyle.Render("│"),
		labelStyle.Render("Expand:"), valueStyle.Render(expandText),
		delimStyle.Render("│"),
		labelStyle.Render("Cap:"), valueStyle.Render(capText),
		delimStyle.Render("│"),
		labelStyle.Render("Cache:"), valueStyle.Render(cacheText),
		delimStyle.Render("│"),
		labelStyle.Render("Mode:"), modeView,
		contextStr,
		statsStr,
	)
	line2Pad := m.width - lipgloss.Width(qdrantInfo)
	if line2Pad > 0 {
		qdrantInfo += strings.Repeat(" ", line2Pad)
	}
	line2 := lipgloss.NewStyle().Background(nord2).Render(qdrantInfo)

	// Line 3: Model and Endpoints Info
	modelInfo := fmt.Sprintf(" %s %s (%s)  %s  %s %s (%s)",
		labelStyle.Render("Embed:"), valueStyle.Render(m.cfg.EmbeddingModel), valueStyle.Render(m.cfg.EmbeddingURL),
		delimStyle.Render("│"),
		labelStyle.Render("LLM:"), valueStyle.Render(m.cfg.OpenAIModel), valueStyle.Render(m.cfg.OpenAIURL),
	)
	line3Pad := m.width - lipgloss.Width(modelInfo)
	if line3Pad > 0 {
		modelInfo += strings.Repeat(" ", line3Pad)
	}
	line3 := lipgloss.NewStyle().Background(nord2).Render(modelInfo)

	// Line 4: Reranker Info (Optional)
	var line4 string
	if m.cfg.RerankerURL != "" {
		rerankerModel := m.cfg.RerankerModel
		if rerankerModel == "" {
			rerankerModel = "generic"
		}
		statusDisp := "enabled"
		statusStyle := lipgloss.NewStyle().Foreground(nord14).Bold(true)
		if m.disableReranker {
			statusDisp = "bypassed"
			statusStyle = lipgloss.NewStyle().Foreground(nord13).Bold(true)
		}
		rerankInfo := fmt.Sprintf(" %s %s (%s) [%s]",
			labelStyle.Render("Rerank:"), valueStyle.Render(rerankerModel), valueStyle.Render(m.cfg.RerankerURL), statusStyle.Render(statusDisp),
		)
		line4Pad := m.width - lipgloss.Width(rerankInfo)
		if line4Pad > 0 {
			rerankInfo += strings.Repeat(" ", line4Pad)
		}
		line4 = lipgloss.NewStyle().Background(nord2).Render(rerankInfo)
	}

	// Border separator
	border := lipgloss.NewStyle().Foreground(nord3).Render(strings.Repeat("─", m.width))

	if line4 != "" {
		return lipgloss.JoinVertical(lipgloss.Left, line1, line2, line3, line4, border)
	}
	return lipgloss.JoinVertical(lipgloss.Left, line1, line2, line3, border)
}

func (m *Model) renderFooter() string {
	// Quit confirmation bar replaces the normal footer.
	if m.state == stateConfirmQuit {
		confirmText := " ⚠  Exit QQuestio?  Press Ctrl+C or Y to confirm  ·  Esc or N to cancel "
		padding := m.width - lipgloss.Width(confirmText)
		if padding > 0 {
			confirmText += strings.Repeat(" ", padding)
		}
		return lipgloss.NewStyle().
			Foreground(nord6).
			Background(nord11).
			Bold(true).
			Render(confirmText)
	}
	styles := DefaultStyles()
	inputView := m.textInput.View()
	footerText := " " + inputView

	// Pad footer to the width of the window
	padding := m.width - lipgloss.Width(footerText)
	if padding > 0 {
		footerText += strings.Repeat(" ", padding)
	}
	return styles.Footer.Render(footerText)
}

// getRenderedTurn renders (or retrieves from cache) the markdown content of an assistant turn.
func (m *Model) getRenderedTurn(turn *ConversationTurn) string {
	if turn.Role != "assistant" {
		return turn.Content
	}
	targetWidth := m.viewport.Width
	if targetWidth < 20 {
		targetWidth = 20
	}
	if turn.RenderedContent == "" || turn.RenderedWidth != targetWidth {
		turn.RenderedContent = renderMarkdown(turn.Content, targetWidth)
		turn.RenderedWidth = targetWidth
	}
	return turn.RenderedContent
}

// getRenderedReferences renders (or retrieves from cache) the references block of a turn.
func (m *Model) getRenderedReferences(turn *ConversationTurn) string {
	if len(turn.References) == 0 {
		return ""
	}
	targetWidth := m.refViewport.Width
	if targetWidth < 20 {
		targetWidth = 20
	}
	if turn.RenderedReferences == "" || turn.RenderedReferencesWidth != targetWidth {
		turn.RenderedReferences = formatReferences(turn.References, targetWidth)
		turn.RenderedReferencesWidth = targetWidth
	}
	return turn.RenderedReferences
}

// getActiveReferences retrieves the active references to be displayed in the right panel.
func (m *Model) getActiveReferences() []rag.QdrantPoint {
	if m.state == stateEmbedding || m.state == stateSearching || m.state == stateReranking || m.state == stateStreaming {
		return m.lastPoints
	}
	for i := len(m.history) - 1; i >= 0; i-- {
		if m.history[i].Role == "assistant" && len(m.history[i].References) > 0 {
			return m.history[i].References
		}
	}
	return nil
}

// updateRefViewport constructs and renders the references in the right viewport.
func (m *Model) updateRefViewport() {
	if m.state == stateEmbedding || m.state == stateSearching || m.state == stateReranking {
		m.refViewport.SetContent("\n  Retrieving references...")
		return
	}

	refs := m.getActiveReferences()
	if len(refs) == 0 {
		m.refViewport.SetContent("\n  No references loaded.\n  Run a query to retrieve context.")
		return
	}
	
	// Format references wrapping them to the viewport's width.
	rendered := formatReferences(refs, m.refViewport.Width)
	m.refViewport.SetContent(rendered)
}

// updateViewport constructs and renders the conversation history in the viewport.
func (m *Model) updateViewport() {
	var sb strings.Builder

	// Render past conversation turns
	for i := range m.history {
		turn := &m.history[i]
		if turn.Role == "user" {
			sb.WriteString(lipgloss.NewStyle().Foreground(nord8).Bold(true).Render("❯ You: ") + turn.Content + "\n\n")
		} else if turn.Role == "assistant" {
			if m.showRawSource {
				sb.WriteString(turn.Content + "\n\n")
			} else {
				sb.WriteString(m.getRenderedTurn(turn) + "\n\n")
			}
			// Add a horizontal rule separating turns
			dividerWidth := m.viewport.Width
			if dividerWidth < 10 {
				dividerWidth = 10
			}
			sb.WriteString(lipgloss.NewStyle().Foreground(nord3).Render(strings.Repeat("─", dividerWidth)) + "\n\n")
		} else if turn.Role == "system" {
			sb.WriteString(lipgloss.NewStyle().Foreground(nord13).Italic(true).Render("ℹ "+turn.Content) + "\n\n")
			dividerWidth := m.viewport.Width
			if dividerWidth < 10 {
				dividerWidth = 10
			}
			sb.WriteString(lipgloss.NewStyle().Foreground(nord3).Render(strings.Repeat("─", dividerWidth)) + "\n\n")
		}
	}

	// Render currently processing user query
	if m.lastQuery != "" {
		sb.WriteString(lipgloss.NewStyle().Foreground(nord8).Bold(true).Render("❯ You: ") + m.lastQuery + "\n\n")
	}

	// Render currently processing stages (embedding, searching, reranking)
	if m.state == stateEmbedding {
		var cb strings.Builder
		cb.WriteString(lipgloss.NewStyle().Foreground(nord13).Render(fmt.Sprintf("  %s Generating embedding vector...", m.spinner.View())) + "\n")
		sb.WriteString(cb.String() + "\n")
	} else if m.state == stateSearching {
		var cb strings.Builder
		cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render("  ✔ Generated embedding vector") + "\n")
		cb.WriteString(lipgloss.NewStyle().Foreground(nord13).Render(fmt.Sprintf("  %s Searching knowledge base in Qdrant...", m.spinner.View())) + "\n")
		sb.WriteString(cb.String() + "\n")
	} else if m.state == stateReranking {
		var cb strings.Builder
		cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render("  ✔ Generated embedding vector") + "\n")
		cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render("  ✔ Retrieved candidates from Qdrant") + "\n")
		cb.WriteString(lipgloss.NewStyle().Foreground(nord13).Render(fmt.Sprintf("  %s Reranking retrieved documents...", m.spinner.View())) + "\n")
		sb.WriteString(cb.String() + "\n")
	}

	// Render currently streaming assistant reply
	if m.state == stateStreaming {
		if m.output != "" {
			var streamingText string
			if m.showRawSource {
				streamingText = cleanLLMOutput(m.output)
			} else {
				streamingText = renderMarkdown(cleanLLMOutput(m.output), m.viewport.Width)
			}
			sb.WriteString(streamingText + "\n\n" + m.spinner.View() + lipgloss.NewStyle().Foreground(nord13).Render(" "+m.loadingMessage) + "\n")
		} else {
			var cb strings.Builder
			cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render("  ✔ Generated embedding vector") + "\n")
			if m.cfg.RerankerURL != "" {
				cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render("  ✔ Retrieved candidates from Qdrant") + "\n")
				cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render(fmt.Sprintf("  ✔ Reranked and selected %d documents", len(m.lastPoints))) + "\n")
			} else {
				cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render(fmt.Sprintf("  ✔ Retrieved %d context documents from Qdrant", len(m.lastPoints))) + "\n")
			}
			cb.WriteString(lipgloss.NewStyle().Foreground(nord13).Render(fmt.Sprintf("  %s %s", m.spinner.View(), m.loadingMessage)) + "\n")
			sb.WriteString(cb.String() + "\n")
		}
	}

	wasAtBottom := m.viewport.AtBottom()
	m.viewport.SetContent(sb.String())
	if wasAtBottom || (m.state == stateStreaming && len(m.output) < 20) {
		m.viewport.GotoBottom()
	}

	// Always sync references panel
	m.updateRefViewport()
}

// cleanLLMOutput strips unwanted LLM template wrapper tokens from the output display.
func cleanLLMOutput(text string) string {
	prefixes := []string{
		"RT_TEXT|>",
		"<|START_OF_TURN_TOKEN|>",
		"<|CHAT_BOT|>",
		"<|im_start|>",
		"<|im_end|>",
		"<|assistant|>",
		"<|START_TEXT|>",
		"START_TEXT",
		"START TEXT",
		"START TEXT:",
		"START TEXT\n",
		"<START TEXT>",
		"[START TEXT]",
	}
	suffixes := []string{
		"<|END_TEXT|>",
		"END_TEXT",
		"END TEXT",
		"END TEXT.",
		"\nEND TEXT",
		"<END TEXT>",
		"[END TEXT]",
	}
	cleaned := text
	for {
		changed := false
		trimmed := strings.TrimSpace(cleaned)
		for _, prefix := range prefixes {
			if strings.HasPrefix(strings.ToUpper(trimmed), strings.ToUpper(prefix)) {
				cleaned = trimmed[len(prefix):]
				changed = true
				break
			}
		}
		trimmed = strings.TrimSpace(cleaned)
		for _, suffix := range suffixes {
			if strings.HasSuffix(strings.ToUpper(trimmed), strings.ToUpper(suffix)) {
				cleaned = trimmed[:len(trimmed)-len(suffix)]
				changed = true
				break
			}
		}
		if !changed {
			break
		}
	}
	return strings.TrimSpace(cleaned)
}

// copyLastResponseCmd finds and copies the last assistant response (or references, depending on focus) to system clipboard.
func (m *Model) copyLastResponseCmd() tea.Cmd {
	if m.focusRef {
		refs := m.getActiveReferences()
		if len(refs) == 0 {
			return func() tea.Msg {
				return slashResultMsg{feedback: "No references to copy yet"}
			}
		}
		refText := formatReferences(refs, 80)
		return func() tea.Msg {
			if err := clipboard.WriteAll(refText); err != nil {
				return appErrMsg{err: err, reason: "Failed to copy references to clipboard", stage: "slash"}
			}
			return slashResultMsg{feedback: "Copied references to clipboard"}
		}
	}

	var lastResponse string
	for i := len(m.history) - 1; i >= 0; i-- {
		if m.history[i].Role == "assistant" {
			lastResponse = m.history[i].Content
			break
		}
	}
	if lastResponse == "" {
		return func() tea.Msg {
			return slashResultMsg{feedback: "No response to copy yet"}
		}
	}
	return func() tea.Msg {
		if err := clipboard.WriteAll(lastResponse); err != nil {
			return appErrMsg{err: err, reason: "Failed to copy to clipboard", stage: "slash"}
		}
		return slashResultMsg{feedback: "Copied last response to clipboard"}
	}
}

// copyAllConversationCmd formats and copies the entire conversation log to the clipboard.
func (m *Model) copyAllConversationCmd() tea.Cmd {
	if len(m.history) == 0 {
		return func() tea.Msg {
			return slashResultMsg{feedback: "No conversation history to copy"}
		}
	}
	var sb strings.Builder
	sb.WriteString("=== Conversation Transcript ===\n\n")
	for _, turn := range m.history {
		if turn.Role == "user" {
			sb.WriteString("❯ You: " + turn.Content + "\n\n")
		} else if turn.Role == "assistant" {
			sb.WriteString("Assistant:\n" + turn.Content + "\n\n")
			if len(turn.References) > 0 {
				sb.WriteString("References:\n")
				for i, pt := range turn.References {
					pointIDStr := fmt.Sprintf("%v", pt.ID)
					textStr := pt.ExtractText()

					docName := extractDocumentName(pt.Payload)
					if docName == "" {
						docName = fmt.Sprintf("ID %s", pointIDStr)
					}
					sb.WriteString(fmt.Sprintf("  [%d] Document: %s (Score: %.4f | ID: %s)\n", i+1, docName, pt.Score, pointIDStr))
					if textStr != "" {
						lines := strings.Split(strings.TrimSpace(textStr), "\n")
						for _, line := range lines {
							sb.WriteString("      " + line + "\n")
						}
					}
					if i < len(turn.References)-1 {
						sb.WriteString("    " + strings.Repeat("-", 60) + "\n")
					}
				}
				sb.WriteString("\n")
			}
			sb.WriteString(strings.Repeat("=", 60) + "\n\n")
		} else if turn.Role == "system" {
			sb.WriteString("System: " + turn.Content + "\n\n")
			sb.WriteString(strings.Repeat("=", 60) + "\n\n")
		}
	}
	return func() tea.Msg {
		if err := clipboard.WriteAll(sb.String()); err != nil {
			return appErrMsg{err: err, reason: "Failed to copy all to clipboard", stage: "slash"}
		}
		return slashResultMsg{feedback: "Copied all conversation to clipboard"}
	}
}

// saveLastResponseCmd writes the last assistant response to a local file.
func (m *Model) saveLastResponseCmd(filename string) tea.Cmd {
	var lastResponse string
	for i := len(m.history) - 1; i >= 0; i-- {
		if m.history[i].Role == "assistant" {
			lastResponse = m.history[i].Content
			break
		}
	}
	if lastResponse == "" {
		return func() tea.Msg {
			return slashResultMsg{feedback: "No response to save yet"}
		}
	}
	return func() tea.Msg {
		err := os.WriteFile(filename, []byte(lastResponse), 0644)
		if err != nil {
			return appErrMsg{err: err, reason: fmt.Sprintf("Failed to write to file: %s", filename), stage: "slash"}
		}
		return slashResultMsg{feedback: fmt.Sprintf("Saved last response to %s", filename)}
	}
}

// saveAllConversationCmd formats and writes the entire conversation log to a local file.
func (m *Model) saveAllConversationCmd(filename string) tea.Cmd {
	if len(m.history) == 0 {
		return func() tea.Msg {
			return slashResultMsg{feedback: "No conversation history to save"}
		}
	}
	var sb strings.Builder
	sb.WriteString("# Conversation Transcript\n\n")
	for _, turn := range m.history {
		if turn.Role == "user" {
			sb.WriteString("## ❯ You\n\n" + turn.Content + "\n\n")
		} else if turn.Role == "assistant" {
			sb.WriteString("## 🤖 Assistant\n\n" + turn.Content + "\n\n")
			if len(turn.References) > 0 {
				sb.WriteString("### References\n\n")
				for i, pt := range turn.References {
					pointIDStr := fmt.Sprintf("%v", pt.ID)
					docName := extractDocumentName(pt.Payload)
					if docName == "" {
						docName = fmt.Sprintf("ID %s", pointIDStr)
					}
					textStr := pt.ExtractText()
					sb.WriteString(fmt.Sprintf("* **[%d]** `%s` (Score: `%.4f` | ID: `%s`)\n", i+1, docName, pt.Score, pointIDStr))
					if textStr != "" {
						sb.WriteString("  ```text\n")
						lines := strings.Split(strings.TrimSpace(textStr), "\n")
						for _, line := range lines {
							sb.WriteString("  " + line + "\n")
						}
						sb.WriteString("  ```\n\n")
					}
				}
			}
			sb.WriteString("---\n\n")
		} else if turn.Role == "system" {
			sb.WriteString("> *System Info: " + turn.Content + "*\n\n")
			sb.WriteString("---\n\n")
		}
	}

	return func() tea.Msg {
		err := os.WriteFile(filename, []byte(sb.String()), 0644)
		if err != nil {
			return appErrMsg{err: err, reason: fmt.Sprintf("Failed to write to file: %s", filename), stage: "slash"}
		}
		return slashResultMsg{feedback: fmt.Sprintf("Saved entire transcript to %s", filename)}
	}
}

// formatReferences renders retrieved Qdrant points as structured references grouped
// by document. After query-time context expansion, points are already in document
// order with adjacent chunks present, so we group by file_name (or the canonical
// document field) and emit one labeled block per source document:
//
//	--- Document A · chunks 4-6 of 18 · score 0.94 ---
//	...text...
//	--- Document B · chunks 1-3 of 12 · score 0.87 ---
//	...text...
//
// If the payload lacks any recognizable document identifier we fall back to the
// flat per-point rendering so the panel never breaks.
func formatReferences(points []rag.QdrantPoint, width int) string {
	if len(points) == 0 {
		return ""
	}
	if width < 20 {
		width = 20
	}

	// Group points by document identifier. We delegate to the same recursive
	// extractDocumentName helper that buildPromptMessages uses, so the
	// reference panel and the LLM prompt see the SAME document titles —
	// this is the fix for the "lost doc title" regression.
	groups := make([][]rag.QdrantPoint, 0)
	groupKeys := make([]string, 0)
	keyToIdx := make(map[string]int)
	noDocIdx := -1

	for _, pt := range points {
		key := extractDocumentName(pt.Payload)
		if key == "" {
			if noDocIdx < 0 {
				noDocIdx = len(groups)
				groups = append(groups, nil)
				groupKeys = append(groupKeys, "")
			}
			groups[noDocIdx] = append(groups[noDocIdx], pt)
			continue
		}
		if idx, ok := keyToIdx[key]; ok {
			groups[idx] = append(groups[idx], pt)
			continue
		}
		keyToIdx[key] = len(groups)
		groups = append(groups, []rag.QdrantPoint{pt})
		groupKeys = append(groupKeys, key)
	}

	var sb strings.Builder
	titleStyle := lipgloss.NewStyle().Foreground(nord9).Bold(true)
	metaStyle := lipgloss.NewStyle().Foreground(nord4)
	idStyle := lipgloss.NewStyle().Foreground(nord7)
	scoreStyle := lipgloss.NewStyle().Foreground(nord14)
	borderStyle := lipgloss.NewStyle().Foreground(nord3)
	chunkStyle := lipgloss.NewStyle().Foreground(nord7).Italic(true)

	sb.WriteString(titleStyle.Render("📚 References / Retrieved Context Chunks:") + "\n")

	// Helper: pretty-print a chunk's text preview.
	emitChunkText := func(text string) {
		if text == "" {
			sb.WriteString(lipgloss.NewStyle().Foreground(nord11).Italic(true).Render("      (No text payload content found in this point)") + "\n")
			return
		}
		wrapWidth := width - 8
		if wrapWidth < 20 {
			wrapWidth = 20
		}
		textStyle := lipgloss.NewStyle().Width(wrapWidth)
		wrapped := textStyle.Render(strings.TrimSpace(text))
		lines := strings.Split(wrapped, "\n")
		for _, line := range lines {
			sb.WriteString("      " + line + "\n")
		}
	}

	// Helper: extract the chunk_index for a point (or -1 if missing).
	extractIdx := func(p rag.QdrantPoint) int {
		if p.Payload == nil {
			return -1
		}
		for _, k := range []string{"chunk_index", "chunkIndex", "position", "seq", "index", "ord"} {
			if v, ok := p.Payload[k]; ok {
				switch n := v.(type) {
				case float64:
					return int(n)
				case int:
					return n
				case int64:
					return int(n)
				case string:
					var i int
					if _, err := fmt.Sscanf(n, "%d", &i); err == nil {
						return i
					}
				}
			}
		}
		return -1
	}

	// Emit each group as a labeled document block.
	for gi, pts := range groups {
		if len(pts) == 0 {
			continue
		}
		// Sort by chunk_index ascending so adjacent chunks read top-to-bottom.
		sort.SliceStable(pts, func(i, j int) bool {
			li, lj := extractIdx(pts[i]), extractIdx(pts[j])
			if li < 0 && lj < 0 {
				return false
			}
			if li < 0 {
				return false
			}
			if lj < 0 {
				return true
			}
			return li < lj
		})

		docName := groupKeys[gi]
		if docName == "" {
			// Fallback for points without a recognizable doc id: emit per-point
			// as before so the user still sees something useful.
			for i, pt := range pts {
				pointIDStr := fmt.Sprintf("%v", pt.ID)
				sb.WriteString(fmt.Sprintf("  %s Document: ID %s (Score: %s)\n",
					idStyle.Render(fmt.Sprintf("[%d]", i+1)),
					pointIDStr,
					scoreStyle.Render(fmt.Sprintf("%.4f", pt.Score)),
				))
				emitChunkText(pt.ExtractText())
				if i < len(pts)-1 || gi < len(groups)-1 {
					sb.WriteString(borderStyle.Render("    "+strings.Repeat("┄", width-8)) + "\n")
				}
			}
			continue
		}

		// Compute the chunk range covered by this group and the top score from its primary chunks.
		lo, hi := -1, -1
		var topScore float32
		hasPrimaryScore := false
		for _, p := range pts {
			idx := extractIdx(p)
			if idx >= 0 {
				if lo < 0 || idx < lo {
					lo = idx
				}
				if idx > hi {
					hi = idx
				}
			}
			if p.IsPrimary {
				if !hasPrimaryScore || p.Score > topScore {
					topScore = p.Score
					hasPrimaryScore = true
				}
			}
		}
		if !hasPrimaryScore {
			topScore = pts[0].Score
		}
		chunkRangeStr := ""
		if lo >= 0 && hi >= 0 {
			if lo == hi {
				chunkRangeStr = fmt.Sprintf("chunks %d of %d", lo, hi)
			} else {
				chunkRangeStr = fmt.Sprintf("chunks %d-%d of %d", lo, hi, hi)
			}
		} else {
			chunkRangeStr = fmt.Sprintf("%d chunks", len(pts))
		}

		// First line: Document name. We wrap it completely to fit in width-16.
		// Indent any wrapped lines by 5 spaces.
		wrapWidth := width - 16
		if wrapWidth < 12 {
			wrapWidth = 12
		}
		docStyle := lipgloss.NewStyle().Width(wrapWidth)
		wrappedDoc := docStyle.Render(docName)
		docLines := strings.Split(wrappedDoc, "\n")
		var docDisplay strings.Builder
		for idx, line := range docLines {
			if idx == 0 {
				docDisplay.WriteString(line)
			} else {
				docDisplay.WriteString("\n     " + line)
			}
		}

		// Second line: Range and score.
		// e.g. "Score: 0.8500 · chunks 262-262"
		scoreStr := fmt.Sprintf("%.4f", topScore)
		var metaDisplay string
		if width >= 40 {
			metaDisplay = fmt.Sprintf("     Score: %s · %s", scoreStyle.Render(scoreStr), chunkStyle.Render(chunkRangeStr))
		} else if width >= 30 {
			metaDisplay = fmt.Sprintf("     S: %s · %s", scoreStyle.Render(scoreStr), chunkStyle.Render(chunkRangeStr))
		} else {
			metaDisplay = fmt.Sprintf("     %s · %s", scoreStyle.Render(scoreStr), chunkStyle.Render(chunkRangeStr))
		}

		sb.WriteString(fmt.Sprintf("  %s %s\n%s\n",
			borderStyle.Render("──"),
			metaStyle.Render("Document: "+docDisplay.String()),
			metaDisplay,
		))

		for _, p := range pts {
			idx := extractIdx(p)
			idxTag := ""
			if idx >= 0 {
				if p.OriginalScore != 0 {
					idxTag = fmt.Sprintf(" (chunk %d, score %.4f, db cosine %.4f)", idx, p.Score, p.OriginalScore)
				} else {
					idxTag = fmt.Sprintf(" (chunk %d, score %.4f)", idx, p.Score)
				}
			} else {
				if p.OriginalScore != 0 {
					idxTag = fmt.Sprintf(" (score %.4f, db cosine %.4f)", p.Score, p.OriginalScore)
				} else {
					idxTag = fmt.Sprintf(" (score %.4f)", p.Score)
				}
			}
			sb.WriteString(metaStyle.Render("    • chunk") + idxTag + "\n")
			emitChunkText(p.ExtractText())
		}

		if gi < len(groups)-1 {
			sb.WriteString(borderStyle.Render("    "+strings.Repeat("┄", width-8)) + "\n")
		}
	}

	return sb.String()
}

// extractDocumentName recursively searches a Qdrant point's payload for a human-readable document name,
// prioritizing real filenames/titles over hex hashes and ignoring ID fields.
func extractDocumentName(payload map[string]interface{}) string {
	if payload == nil {
		return ""
	}

	targetKeys := []string{
		"file_name", "filename", "fileName", "document_name", "doc_name",
		"document", "doc", "source_file", "sourceFile", "title", "source", "name", "path", "url",
	}

	var hashCandidate string

	var search func(m map[string]interface{}) string
	search = func(m map[string]interface{}) string {
		// 1. Check exact key matches first
		for _, key := range targetKeys {
			if val, ok := m[key]; ok {
				if s, ok := val.(string); ok && s != "" {
					if isHexHash(s) {
						if hashCandidate == "" {
							hashCandidate = s
						}
					} else {
						return s
					}
				}
			}
		}

		// 2. Generic check for keys containing name indicators, excluding "id" / "score"
		for k, val := range m {
			kl := strings.ToLower(k)
			if strings.Contains(kl, "id") || strings.Contains(kl, "score") {
				continue
			}
			if strings.Contains(kl, "file") || strings.Contains(kl, "name") || strings.Contains(kl, "title") || strings.Contains(kl, "source") || strings.Contains(kl, "path") || strings.Contains(kl, "url") || strings.Contains(kl, "doc") {
				if s, ok := val.(string); ok && s != "" {
					if isHexHash(s) {
						if hashCandidate == "" {
							hashCandidate = s
						}
					} else {
						return s
					}
				}
			}
		}

		// 3. Search nested maps recursively
		for _, val := range m {
			if nestedMap, ok := val.(map[string]interface{}); ok {
				if s := search(nestedMap); s != "" {
					return s
				}
			}
		}
		return ""
	}

	if name := search(payload); name != "" {
		return name
	}

	return hashCandidate
}

// truncateMiddle truncates a string in the middle to fit within maxLen.
// It is particularly useful for keeping the prefix and suffix (like a file extension) intact.
func truncateMiddle(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen < 10 {
		if maxLen > 3 {
			return s[:maxLen-3] + "..."
		}
		return s[:maxLen]
	}
	half := (maxLen - 3) / 2
	return s[:half] + "..." + s[len(s)-half:]
}

// isHexHash returns true if a string is a standard MD5, SHA-1, or SHA-256 hex hash.
func isHexHash(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) != 32 && len(s) != 40 && len(s) != 64 {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F') || r == '-' || r == '_') {
			return false
		}
	}
	return true
}

// formatNumber formats an integer with thousands separators (e.g. 1234567 -> "1,234,567").
// Used for human-readable progress and stats messages in the header / status bar.
func formatNumber(n int) string {
	if n < 0 {
		return "-" + formatNumber(-n)
	}
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	// Insert commas from the right.
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
		if len(s) > pre {
			b.WriteString(",")
		}
	}
	for i := pre; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteString(",")
		}
	}
	return b.String()
}

// renderMarkdown converts raw markdown into styled terminal output using Glamour.
func renderMarkdown(text string, width int) string {
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle("dark"),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		return text
	}
	out, err := r.Render(text)
	if err != nil {
		return text
	}
	return strings.TrimSpace(out)
}

type Session struct {
	ID           string             `json:"id"`
	Collection   string             `json:"collection,omitempty"`
	SearchLimit  int                `json:"search_limit,omitempty"`
	SearchCap    int                `json:"search_cap,omitempty"`
	RerankerPool int                `json:"reranker_pool,omitempty"`
	SearchExpand int                `json:"search_expand,omitempty"`
	SearchMode   string             `json:"search_mode,omitempty"`
	SystemPrompt string             `json:"system_prompt,omitempty"`
	RAGMode      string             `json:"rag_mode,omitempty"`
	FilterKey    string             `json:"filter_key,omitempty"`
	FilterValue  string             `json:"filter_value,omitempty"`
	History      []ConversationTurn `json:"history"`
}

func GetSessionsDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, "config", "qquestio", "sessions")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

func GetLastSessionID() (string, error) {
	dir, err := GetSessionsDir()
	if err != nil {
		return "", err
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var sessionFiles []string
	for _, f := range files {
		if !f.IsDir() && strings.HasSuffix(f.Name(), ".json") {
			sessionFiles = append(sessionFiles, strings.TrimSuffix(f.Name(), ".json"))
		}
	}
	if len(sessionFiles) == 0 {
		return "", fmt.Errorf("no sessions found")
	}
	sort.Strings(sessionFiles)
	return sessionFiles[len(sessionFiles)-1], nil
}

func (m *Model) hasUserPrompt() bool {
	for _, turn := range m.history {
		if turn.Role == "user" {
			return true
		}
	}
	return false
}

func (m *Model) saveSession() error {
	if !m.hasUserPrompt() {
		return nil
	}
	dir, err := GetSessionsDir()
	if err != nil {
		return err
	}
	filePath := filepath.Join(dir, m.sessionID+".json")

	sess := Session{
		ID:           m.sessionID,
		Collection:   m.collection,
		SearchLimit:  m.searchLimit,
		SearchCap:    m.searchCap,
		RerankerPool: m.rerankerPool,
		SearchExpand: m.searchExpand,
		SearchMode:   m.searchMode,
		SystemPrompt: m.systemPrompt,
		RAGMode:      m.ragMode,
		FilterKey:    m.filterKey,
		FilterValue:  m.filterValue,
		History:      m.history,
	}

	data, err := json.MarshalIndent(sess, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filePath, data, 0644)
}

func (m *Model) loadSession(sessionID string) error {
	dir, err := GetSessionsDir()
	if err != nil {
		return err
	}
	filePath := filepath.Join(dir, sessionID+".json")

	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}

	var sess Session
	if err := json.Unmarshal(data, &sess); err != nil {
		return err
	}

	m.sessionID = sess.ID
	if sess.Collection != "" {
		m.collection = sess.Collection
	}
	if sess.SearchLimit > 0 {
		m.searchLimit = sess.SearchLimit
	}
	m.searchCap = sess.SearchCap
	m.rerankerPool = sess.RerankerPool
	m.searchExpand = sess.SearchExpand
	if sess.SearchMode != "" {
		m.searchMode = sess.SearchMode
	}
	if sess.SystemPrompt != "" {
		m.systemPrompt = sess.SystemPrompt
	}
	if sess.RAGMode != "" {
		m.ragMode = sess.RAGMode
	}
	m.filterKey = sess.FilterKey
	m.filterValue = sess.FilterValue
	m.history = sess.History

	// Rebuild input history from user turns
	m.inputHistory = nil
	for _, turn := range m.history {
		if turn.Role == "user" {
			m.inputHistory = append(m.inputHistory, turn.Content)
		}
	}
	m.historyIndex = len(m.inputHistory)

	// Force refresh viewport content and scroll to bottom
	m.updateViewport()
	m.viewport.GotoBottom()
	return nil
}

func (m *Model) selectRandomLoadingMessage() {
	sentences := []string{
		"Consulting the oracle...",
		"Distilling semantic essence...",
		"Pondering the query...",
		"Synthesizing knowledge...",
		"Weaving words together...",
		"Querying the matrix...",
		"Extracting insights...",
		"Formulating explanation...",
		"Decoding latent space...",
		"Assembling response pieces...",
		"Tuning hyperparameters...",
		"Mining vector indexes...",
		"Aggregating context fragments...",
	}
	m.loadingMessage = sentences[time.Now().UnixNano()%int64(len(sentences))]
}


