package main

import (
	"context"
	"fmt"
	"os"
	"strings"

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
	stateIdle      appState = iota // Waiting for user input
	stateEmbedding                 // Generating embedding vector
	stateSearching                 // Querying Qdrant
	stateReranking                 // Reranking retrieved documents
	stateStreaming                 // Receiving LLM SSE chunks
	stateError                     // Displaying error, input still active
)

type ConversationTurn struct {
	Role                    string            // "user" | "assistant" | "system"
	Content                 string            // The text content
	References              []rag.QdrantPoint // Retrieved context points (only for assistant responses)
	RenderedContent         string            // Cached rendered markdown
	RenderedWidth           int               // The width at which it was rendered
	RenderedReferences      string            // Cached rendered references block
	RenderedReferencesWidth int               // The width at which references were rendered
}

type Model struct {
	// --- Config (immutable) ---
	cfg Config

	// --- Runtime state (mutable via slash commands) ---
	collection        string // Active Qdrant collection (init: cfg.DefaultCollection)
	searchLimit       int    // Number of Qdrant results (default: 5)
	searchCap         int    // Hard upper bound on candidate pool for Qdrant search (0 = no cap, search full corpus)
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
	textInput textinput.Model
	viewport  viewport.Model
	spinner   spinner.Model
	statusMsg string // Displayed in header bar

	// --- Conversation ---
	history       []ConversationTurn // Full conversation history including system info
	output        string             // Accumulated LLM response text for current turn
	showRawSource bool               // Toggle between glamour-rendered markdown and raw markdown source

	// --- Pipeline transient ---
	lastQuery     string            // The user query that started the pipeline
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

	return &Model{
		cfg:             cfg,
		collection:      cfg.DefaultCollection,
		searchLimit:     10,
		searchCap:       cfg.SearchCap,
		state:           stateIdle,
		textInput:       ti,
		viewport:        vp,
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
	)
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case tea.KeyMsg:
		if msg.Type != tea.KeyEscape {
			m.escCount = 0
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
			if m.cancelRequest != nil {
				m.cancelRequest()
			}
			return m, tea.Quit
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
		case tea.KeyCtrlUp:
			m.viewport.LineUp(1)
		case tea.KeyCtrlDown:
			m.viewport.LineDown(1)
		case tea.KeyPgUp:
			m.viewport.HalfPageUp()
		case tea.KeyPgDown:
			m.viewport.HalfPageDown()
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
				m.lastQuery = raw
				m.output = ""
				m.state = stateEmbedding
				m.statusMsg = "Generating embedding..."
				m.updateViewport()
				cmds = append(cmds, m.generateEmbeddingCmd(raw))
			}
		}

	// --- Pipeline chain ---
	case embeddingMsg:
		m.state = stateSearching
		m.statusMsg = "Searching knowledge base..."
		cmds = append(cmds, m.searchQdrantCmd(msg.vector))

	case searchResultMsg:
		if m.cfg.RerankerURL != "" && !m.disableReranker {
			m.state = stateReranking
			m.statusMsg = "Reranking retrieved documents..."
			m.updateViewport()
			cmds = append(cmds, m.rerankPointsCmd(msg.points))
		} else {
			m.ragContext = msg.context
			m.lastPoints = msg.points
			m.state = stateStreaming
			docCount := len(msg.points)
			m.statusMsg = fmt.Sprintf("Generating response... (%d docs retrieved)", docCount)
			cmds = append(cmds, m.startLLMStreamCmd())
		}

	case rerankResultMsg:
		m.ragContext = msg.context
		m.lastPoints = msg.points
		m.state = stateStreaming
		docCount := len(msg.points)
		m.statusMsg = fmt.Sprintf("Generating response... (%d docs retrieved)", docCount)
		cmds = append(cmds, m.startLLMStreamCmd())

	case streamChunkMsg:
		if msg.done {
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
		} else {
			m.output += msg.content
			m.updateViewport()
			cmds = append(cmds, m.receiveStreamChunkCmd())
		}

	case appErrMsg:
		if m.stoppedByUser {
			m.stoppedByUser = false
			return m, nil
		}
		m.state = stateError
		m.statusMsg = fmt.Sprintf("Error [%s]: %s (%v)", msg.stage, msg.reason, msg.err)
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

	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		headerH := 4
		if m.cfg.RerankerURL != "" {
			headerH = 5
		}
		footerH := 3
		m.viewport.Width = m.width
		m.viewport.Height = m.height - headerH - footerH
		m.textInput.Width = m.width - 4
	}

	// --- Sub-model updates (always, for cursor blink + spinner) ---
	var spinnerCmd tea.Cmd
	if _, ok := msg.(spinner.TickMsg); !ok {
		m.spinner, spinnerCmd = m.spinner.Update(msg)
		cmds = append(cmds, spinnerCmd)
	}
	var tiCmd tea.Cmd
	m.textInput, tiCmd = m.textInput.Update(msg)
	cmds = append(cmds, tiCmd)

	if m.state == stateStreaming || m.state == stateIdle || m.state == stateError {
		var shouldForward = true
		if keyMsg, ok := msg.(tea.KeyMsg); ok {
			if keyMsg.Type == tea.KeyUp || keyMsg.Type == tea.KeyDown {
				shouldForward = false
			}
		}
		if shouldForward {
			var vpCmd tea.Cmd
			m.viewport, vpCmd = m.viewport.Update(msg)
			cmds = append(cmds, vpCmd)
		}
	}

	return m, tea.Batch(cmds...)
}

func (m *Model) View() string {
	header := m.renderHeader()
	body := m.viewport.View()
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

	qdrantInfo := fmt.Sprintf(" %s %s  %s  %s %s  %s %s %d  %s  %s %s  %s  %s %s  %s  %s %s%s",
		labelStyle.Render("DB:"), valueStyle.Render(m.cfg.QdrantURL),
		delimStyle.Render("│"),
		labelStyle.Render("Col:"), valueStyle.Render(m.collection),
		delimStyle.Render("│"),
		labelStyle.Render("Limit:"), m.searchLimit,
		delimStyle.Render("│"),
		labelStyle.Render("Cap:"), valueStyle.Render(capText),
		delimStyle.Render("│"),
		labelStyle.Render("Cache:"), valueStyle.Render(cacheText),
		delimStyle.Render("│"),
		labelStyle.Render("Mode:"), modeView,
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
	targetWidth := m.width - 4
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
	targetWidth := m.width - 4
	if targetWidth < 20 {
		targetWidth = 20
	}
	if turn.RenderedReferences == "" || turn.RenderedReferencesWidth != targetWidth {
		turn.RenderedReferences = formatReferences(turn.References, targetWidth)
		turn.RenderedReferencesWidth = targetWidth
	}
	return turn.RenderedReferences
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
			if len(turn.References) > 0 {
				sb.WriteString(m.getRenderedReferences(turn) + "\n\n")
			}
			// Add a horizontal rule separating turns
			sb.WriteString(lipgloss.NewStyle().Foreground(nord3).Render(strings.Repeat("─", m.width-4)) + "\n\n")
		} else if turn.Role == "system" {
			sb.WriteString(lipgloss.NewStyle().Foreground(nord13).Italic(true).Render("ℹ "+turn.Content) + "\n\n")
			sb.WriteString(lipgloss.NewStyle().Foreground(nord3).Render(strings.Repeat("─", m.width-4)) + "\n\n")
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
				streamingText = renderMarkdown(cleanLLMOutput(m.output), m.width-4)
			}
			sb.WriteString(streamingText + "\n\n" + m.spinner.View() + lipgloss.NewStyle().Foreground(nord13).Render(" Cooking response from OpenAI...") + "\n")
		} else {
			var cb strings.Builder
			cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render("  ✔ Generated embedding vector") + "\n")
			if m.cfg.RerankerURL != "" {
				cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render("  ✔ Retrieved candidates from Qdrant") + "\n")
				cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render(fmt.Sprintf("  ✔ Reranked and selected %d documents", len(m.lastPoints))) + "\n")
			} else {
				cb.WriteString(lipgloss.NewStyle().Foreground(nord14).Render(fmt.Sprintf("  ✔ Retrieved %d context documents from Qdrant", len(m.lastPoints))) + "\n")
			}
			cb.WriteString(lipgloss.NewStyle().Foreground(nord13).Render(fmt.Sprintf("  %s Cooking response from OpenAI...", m.spinner.View())) + "\n")
			sb.WriteString(cb.String() + "\n")
		}
	}

	wasAtBottom := m.viewport.AtBottom()
	m.viewport.SetContent(sb.String())
	if wasAtBottom || (m.state == stateStreaming && len(m.output) < 20) {
		m.viewport.GotoBottom()
	}
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

// copyLastResponseCmd finds and copies the last assistant response to system clipboard.
func (m *Model) copyLastResponseCmd() tea.Cmd {
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

// formatReferences renders retrieved Qdrant points as structured references with scores and text previews.
func formatReferences(points []rag.QdrantPoint, width int) string {
	if len(points) == 0 {
		return ""
	}
	var sb strings.Builder
	titleStyle := lipgloss.NewStyle().Foreground(nord9).Bold(true)
	metaStyle := lipgloss.NewStyle().Foreground(nord4)
	idStyle := lipgloss.NewStyle().Foreground(nord7)
	scoreStyle := lipgloss.NewStyle().Foreground(nord14)
	borderStyle := lipgloss.NewStyle().Foreground(nord3)

	sb.WriteString(titleStyle.Render("📚 References / Retrieved Context Chunks:") + "\n")
	for i, pt := range points {
		pointIDStr := fmt.Sprintf("%v", pt.ID)
		textStr := pt.ExtractText()

		source := extractDocumentName(pt.Payload)

		var docIdentifier string
		if source != "" {
			docIdentifier = fmt.Sprintf("Document: %s", source)
		} else {
			docIdentifier = fmt.Sprintf("Document: ID %s", pointIDStr)
		}

		idMetaStr := ""
		if source != "" {
			idMetaStr = fmt.Sprintf(" | ID: %s", pointIDStr)
		}

		sb.WriteString(fmt.Sprintf("  %s %s (Score: %s%s)\n",
			idStyle.Render(fmt.Sprintf("[%d]", i+1)),
			metaStyle.Render(docIdentifier),
			scoreStyle.Render(fmt.Sprintf("%.4f", pt.Score)),
			metaStyle.Render(idMetaStr),
		))

		if textStr != "" {
			wrapWidth := width - 6
			if wrapWidth < 20 {
				wrapWidth = 20
			}
			textStyle := lipgloss.NewStyle().Width(wrapWidth)
			wrapped := textStyle.Render(strings.TrimSpace(textStr))
			lines := strings.Split(wrapped, "\n")
			for _, line := range lines {
				sb.WriteString("      " + line + "\n")
			}
		} else {
			sb.WriteString(lipgloss.NewStyle().Foreground(nord11).Italic(true).Render("      (No text payload content found in this point)") + "\n")
		}
		if i < len(points)-1 {
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
