package main

import "qquestio/internal/rag"

// embeddingMsg is Phase 1 (Stage 1 result): embedding vector from llama.cpp.
type embeddingMsg struct {
	vector []float32
}

// searchResultMsg is Phase 2 (Stage 2 result): retrieved context from Qdrant.
type searchResultMsg struct {
	context string            // Concatenated text payloads from Qdrant results
	points  []rag.QdrantPoint // The full points list with scores/metadata
}

// rerankResultMsg is the result of the optional reranking step.
type rerankResultMsg struct {
	context string            // Concatenated text payloads after reranking
	points  []rag.QdrantPoint // Sorted and sliced points list
}

// streamChunkMsg is Phase 3 (Stage 3): one chunk from LiteLLM SSE stream.
type streamChunkMsg struct {
	content string
	done    bool // true when stream is exhausted
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
