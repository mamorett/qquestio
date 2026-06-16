package main

import (
	"context"
	"fmt"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"qquestio/internal/rag"
)

// generateEmbeddingCmd (Stage 1) calls llama.cpp to generate the embedding vector.
func (m *Model) generateEmbeddingCmd(query string) tea.Cmd {
	return func() tea.Msg {
		vector, err := rag.GetEmbedding(m.ctx, m.cfg.EmbeddingURL, m.cfg.EmbeddingAPIKey, m.cfg.EmbeddingModel, query)
		if err != nil {
			return appErrMsg{err: err, reason: "Failed to generate embedding", stage: "embedding"}
		}
		return embeddingMsg{vector: vector}
	}
}

// searchQdrantCmd (Stage 2) performs similarity search against Qdrant collection.
func (m *Model) searchQdrantCmd(vector []float32) tea.Cmd {
	return func() tea.Msg {
		limit := m.searchLimit
		if m.cfg.RerankerURL != "" {
			limit = m.searchLimit * 3
			if limit < 20 {
				limit = 20
			}
		}
		results, points, err := rag.SearchQdrant(
			m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
			m.collection, vector, limit,
		)
		if err != nil {
			return appErrMsg{err: err, reason: "Qdrant search failed", stage: "search"}
		}
		return searchResultMsg{context: results, points: points}
	}
}

// fetchQdrantInfoCmd queries Qdrant for collection details (points, vectors, status).
func (m *Model) fetchQdrantInfoCmd() tea.Cmd {
	return func() tea.Msg {
		points, vectors, status, err := rag.GetCollectionInfo(
			m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey, m.collection,
		)
		return qdrantInfoMsg{
			pointsCount:  points,
			vectorsCount: vectors,
			status:       status,
			err:          err,
		}
	}
}

// startLLMStreamCmd (Stage 3 start) triggers streaming completions from LiteLLM.
func (m *Model) startLLMStreamCmd() tea.Cmd {
	return func() tea.Msg {
		messages := m.buildPromptMessages() // system + history + RAG context + user query
		ctx, cancel := context.WithCancel(m.ctx)
		m.cancelRequest = cancel

		reader, err := rag.StartLiteLLMStream(ctx, m.cfg.OpenAIURL, m.cfg.OpenAIAPIKey, m.cfg.OpenAIModel, messages)
		if err != nil {
			return appErrMsg{err: err, reason: "LLM connection failed", stage: "stream"}
		}
		m.streamReader = reader

		// Immediately attempt first chunk (mirrors mods pattern)
		return m.receiveStreamChunkCmd()()
	}
}

// receiveStreamChunkCmd (SSE Self-Chaining Loop) reads one chunk from the LLM stream.
func (m *Model) receiveStreamChunkCmd() tea.Cmd {
	return func() tea.Msg {
		if m.streamReader == nil {
			return streamChunkMsg{done: true}
		}
		chunk, done, err := m.streamReader.Next()
		if err != nil {
			return appErrMsg{err: err, reason: "Stream read error", stage: "stream"}
		}
		return streamChunkMsg{content: chunk, done: done}
	}
}

// rerankPointsCmd (Optional Stage 2.5) reranks the retrieved Qdrant points using a generic reranker.
func (m *Model) rerankPointsCmd(points []rag.QdrantPoint) tea.Cmd {
	return func() tea.Msg {
		if len(points) == 0 {
			return rerankResultMsg{context: "", points: nil}
		}

		// Extract texts for reranking
		var texts []string
		for _, pt := range points {
			texts = append(texts, pt.ExtractText())
		}

		rerankItems, err := rag.Rerank(
			m.ctx,
			m.cfg.RerankerURL,
			m.cfg.RerankerAPIKey,
			m.cfg.RerankerModel,
			m.lastQuery,
			texts,
		)
		if err != nil {
			return appErrMsg{err: err, reason: fmt.Sprintf("Reranker query failed: %v", err), stage: "rerank"}
		}

		// Map scores back to the original QdrantPoint slice and sort them
		scoreMap := make(map[int]float64)
		for _, item := range rerankItems {
			scoreMap[item.Index] = item.Score
		}

		// Sort the points based on their rerank score descending
		type scoredPoint struct {
			point rag.QdrantPoint
			score float64
		}
		var scoredPoints []scoredPoint
		for i, pt := range points {
			score, ok := scoreMap[i]
			if !ok {
				score = -999.0
			}
			// Update the score displayed in UI
			pt.Score = float32(score)
			scoredPoints = append(scoredPoints, scoredPoint{point: pt, score: score})
		}

		// Sort by score desc
		sort.Slice(scoredPoints, func(i, j int) bool {
			return scoredPoints[i].score > scoredPoints[j].score
		})

		// Slice to the original searchLimit
		var finalPoints []rag.QdrantPoint
		limit := m.searchLimit
		if limit > len(scoredPoints) {
			limit = len(scoredPoints)
		}
		for i := 0; i < limit; i++ {
			finalPoints = append(finalPoints, scoredPoints[i].point)
		}

		// Rebuild the RAG context string using only finalPoints
		var finalTexts []string
		for _, pt := range finalPoints {
			finalTexts = append(finalTexts, pt.ExtractText())
		}
		finalContext := strings.Join(finalTexts, "\n---\n")

		return rerankResultMsg{context: finalContext, points: finalPoints}
	}
}

// buildPromptMessages formats the chat message slice for LiteLLM.
func (m *Model) buildPromptMessages() []rag.ChatMessage {
	msgs := []rag.ChatMessage{}

	// 1. System prompt
	system := m.systemPrompt
	if system == "" {
		system = "You are QQuestio, an advanced, highly articulate enterprise RAG assistant. " +
			"Your goal is to provide comprehensive, detailed, and well-structured answers using the retrieved context documents. " +
			"Analyze the retrieved context chunks carefully, synthesize the information from all sources, and answer the user's question in a creative, professional, and fully expanded manner. " +
			"Connect different parts of the context to form a coherent, narrative explanation. " +
			"Do not give brief or one-word answers. Always explain the background and details. " +
			"Use rich Markdown formatting (bullet points, bold text, headers, code blocks where appropriate) to make your output clear and easy to read. " +
			"If the retrieved context does not contain the complete answer, use your extensive general knowledge to enrich the answer, but note which parts came from your knowledge base and which parts were retrieved locally."
	}
	msgs = append(msgs, rag.ChatMessage{Role: "system", Content: system})

	// 2. Conversation history (multi-turn)
	for _, turn := range m.history {
		if turn.Role == "user" || turn.Role == "assistant" {
			msgs = append(msgs, rag.ChatMessage{Role: turn.Role, Content: turn.Content})
		}
	}

	// 3. Current user query with RAG context injected as structured chunks
	var contextBuilder strings.Builder
	if len(m.lastPoints) > 0 {
		contextBuilder.WriteString("Retrieved Context Chunks from Knowledge Base:\n")
		for i, pt := range m.lastPoints {
			var source string
			var textStr string
			pointIDStr := fmt.Sprintf("%v", pt.ID)
			if pt.Payload != nil {
				// Prioritized key list for human-readable document name
				for _, key := range []string{"file_name", "filename", "fileName", "document_name", "doc_name", "title", "source", "name"} {
					if val, ok := pt.Payload[key]; ok {
						if s, ok := val.(string); ok && s != "" {
							source = s
							break
						}
					}
				}
				// Generic fallback, excluding key names containing "id" or "score" to prevent selecting point IDs
				if source == "" {
					for k, val := range pt.Payload {
						kl := strings.ToLower(k)
						if strings.Contains(kl, "id") || strings.Contains(kl, "score") {
							continue
						}
						if strings.Contains(kl, "file") || strings.Contains(kl, "name") || strings.Contains(kl, "title") || strings.Contains(kl, "source") || strings.Contains(kl, "path") || strings.Contains(kl, "url") {
							if s, ok := val.(string); ok && s != "" {
								source = s
								break
							}
						}
					}
				}

				for _, key := range []string{"text", "content", "document", "page_content", "description", "body"} {
					if val, ok := pt.Payload[key]; ok {
						if s, ok := val.(string); ok && s != "" {
							textStr = s
							break
						}
					}
				}
				if textStr == "" {
					var parts []string
					for k, v := range pt.Payload {
						isMeta := false
						for _, mk := range []string{"file", "filename", "file_name", "source", "title", "url", "path", "id", "score", "page", "author", "date", "created_at"} {
							if k == mk {
								isMeta = true
								break
							}
						}
						if !isMeta {
							if s, ok := v.(string); ok && len(s) > 5 {
								parts = append(parts, s)
							}
						}
					}
					if len(parts) > 0 {
						textStr = strings.Join(parts, " ")
					}
				}
			}

			docName := source
			if docName == "" {
				docName = fmt.Sprintf("ID %s", pointIDStr)
			}
			contextBuilder.WriteString(fmt.Sprintf("--- Chunk %d | Document: %s ---\n%s\n", i+1, docName, textStr))
		}
		contextBuilder.WriteString("---\n\n")
	}

	userMsg := fmt.Sprintf("%sQuestion: %s", contextBuilder.String(), m.lastQuery)
	msgs = append(msgs, rag.ChatMessage{Role: "user", Content: userMsg})

	return msgs
}
