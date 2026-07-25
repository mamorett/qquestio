package main

import "qquestio/internal/rag"

// embeddingMsg is Phase 1 (Stage 1 result): embedding vector from llama.cpp.
type embeddingMsg struct {
	vector []float32
}

// searchResultMsg is Phase 2 (Stage 2 result): retrieved context from Qdrant.
//
// In addition to the joined context and the full points list (already
// expanded when applicable), we also carry the primary top-N matches and
// the cached expansion map. This lets the rerank step rerank only the
// primaries (not the already-expanded set) and then re-apply ±expand
// around the reranked top-K via ApplyExpansionToPrimaries, so /expand
// is preserved through the rerank path.
type searchResultMsg struct {
	context       string            // Concatenated text payloads (post-expansion) from Qdrant
	points        []rag.QdrantPoint // The full expanded points list with scores/metadata
	primaryPoints []rag.QdrantPoint // The top-N primary matches (pre-expansion) for rerank
	expansionMap  rag.ExpansionMap  // docID → chunk_index → point, for post-rerank re-expansion
	expand        int               // The expand value used in this search (so rerank can re-apply)
}

// rerankResultMsg is the result of the optional reranking step. It carries
// the reranked + re-expanded set so the LLM sees the full document context
// around the reranked top-K, not just the top-K fragments.
type rerankResultMsg struct {
	context string            // Concatenated text payloads after rerank + re-expansion
	points  []rag.QdrantPoint // Sorted, re-expanded points list
}

// streamChunkMsg is Phase 3 (Stage 3): one chunk from LiteLLM SSE stream.
type streamChunkMsg struct {
	content   string
	reasoning string
	done      bool // true when stream is exhausted
	usage     rag.TokenUsage
	hasUsage  bool
}

// appErrMsg wraps errors that happen during the pipeline stages.
type appErrMsg struct {
	err    error
	reason string // User-facing explanation
	stage  string // "embedding" | "search" | "stream" | "slash"
}

func (e appErrMsg) Error() string {
	if e.err != nil {
		return e.err.Error()
	}
	return e.reason
}

// slashResultMsg represents the feedback/result from running a slash command.
type slashResultMsg struct {
	feedback string // Confirmation text shown in header
}

// qdrantInfoMsg represents the collection statistics fetched from Qdrant.
type qdrantInfoMsg struct {
	pointsCount  int
	vectorsCount int
	status       string
	err          error
}

// systemLogMsg represents system events or information that should be output to the viewport.
type systemLogMsg struct {
	content  string
	feedback string
}

// quitMsg is returned when a /quit command is executed.
type quitMsg struct{}

// searchProgressMsg is a transient status update emitted by the
// long-running full-corpus search. It updates the header status bar
// (e.g. "Streaming corpus: 234,567 / 1,200,000 points...") without
// polluting the conversation history or advancing the FSM.
type searchProgressMsg struct {
	status string
}

// cachePreloadMsg is returned by preloadCacheInfoCmd on startup to update
// the header bar with cached corpus information (if a cache file exists).
type cachePreloadMsg struct {
	found      bool
	info       string
	pointCount int
}

// warmupCacheMsg signals that the user requested a /cache warmup and the
// scroll-based cache population should begin.
type warmupCacheMsg struct{}

// skillResultMsg is the result of executing a local tool / skill.
type skillResultMsg struct {
	name   string
	input  string
	output string
	err    error
}

// llmInfoMsg represents the LLM connection check result.
type llmInfoMsg struct {
	err error
}

// embedderInfoMsg represents the embedder connection check result.
type embedderInfoMsg struct {
	err error
}
