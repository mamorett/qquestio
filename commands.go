package main

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

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
//
// Three execution paths are chosen automatically based on m.searchCap and m.searchMode:
//
//   - Cap set (m.searchCap > 0): use Qdrant's HNSW query API with the cap as
//     the candidate pool size. Fast, but bounded by the cap (approximate recall).
//
//   - No cap, mode "local" (opt-in via /cap local): use SearchQdrantFullCorpus,
//     which streams every vector via /points/scroll and computes cosine
//     similarity on the client using all local CPU cores. Slower (network-bound),
//     but works when Qdrant refuses params.exact=true.
//
//   - No cap, default: use Qdrant's native brute-force search via
//     params.exact=true. Qdrant scores every vector server-side using SIMD
//     across all its CPU cores and returns ONLY the top-N results. This is the
//     MAXIMUM-SPEED full-corpus path — sub-second even for million-scale
//     collections, with 100% recall and ~100% Qdrant CPU usage.
//
// In all paths, the user-requested `docs` (m.searchLimit) is the final return count.
func (m *Model) searchQdrantCmd(vector []float32) tea.Cmd {
	return func() tea.Msg {
		docs := m.searchLimit

		// If reranking is enabled, we retrieve a larger candidate pool (e.g. 50)
		// so the reranker has a representative set to choose the top-N from.
		searchDocs := docs
		if m.cfg.RerankerURL != "" && !m.disableReranker {
			searchDocs = 50
			if searchDocs < docs {
				searchDocs = docs
			}
		}

		// HNSW path: user explicitly opted into a cap (approximate, bounded).
		if m.searchCap > 0 {
			candidateLimit := m.searchCap
			if candidateLimit < searchDocs {
				candidateLimit = searchDocs
			}
			results, points, err := rag.SearchQdrant(
				m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
				m.collection, vector,
				candidateLimit, searchDocs,
				m.filterKey, m.filterValue,
			)
			if err != nil {
				return appErrMsg{err: err, reason: "Qdrant search failed", stage: "search"}
			}
			return searchResultMsg{context: results, points: points}
		}

		// Local-fallback path: user explicitly requested client-side brute force
		// via /cap local. Streams every vector and scores locally on all CPU cores.
		if m.searchMode == "local" {
			forceRefresh := m.cacheForceRefresh
			results, points, fromCache, err := rag.SearchQdrantFullCorpus(
				m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
				m.collection, vector,
				searchDocs,
				m.filterKey, m.filterValue,
				m.qdrantPoints,
				forceRefresh,
				nil,
			)
			if err != nil {
				return appErrMsg{err: err, reason: "Full-corpus local search failed", stage: "search"}
			}
			m.cacheForceRefresh = false
			if fromCache {
				if cache, _, cerr := rag.LoadCorpusCache(m.collection); cerr == nil && cache != nil {
					age := time.Since(cache.CachedAt).Truncate(time.Second)
					m.cacheInfo = fmt.Sprintf("✓ %s pts (%s old)", formatNumber(cache.PointCount), age)
				}
			}
			return searchResultMsg{context: results, points: points}
		}

		// DEFAULT no-cap path: Qdrant native brute-force (params.exact=true).
		// MAXIMUM-SPEED. Qdrant iterates the entire collection server-side using
		// SIMD across all cores and returns only the top `searchDocs` points —
		// no multi-GB wire transfer, no client-side scoring bottleneck.
		results, points, err := rag.SearchQdrantExact(
			m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
			m.collection, vector,
			searchDocs,
			m.filterKey, m.filterValue,
		)
		if err != nil {
			return appErrMsg{err: err, reason: "Qdrant exact search failed", stage: "search"}
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

// preloadCacheInfoCmd checks for an existing on-disk cache at startup and
// updates the header bar so the user knows whether the cache is warm.
func (m *Model) preloadCacheInfoCmd() tea.Cmd {
	return func() tea.Msg {
		cache, _, err := rag.LoadCorpusCache(m.collection)
		if err != nil || cache == nil {
			return cachePreloadMsg{found: false}
		}
		age := time.Since(cache.CachedAt).Truncate(time.Second)
		info := fmt.Sprintf("✓ %s pts (%s old)", formatNumber(cache.PointCount), age)
		return cachePreloadMsg{found: true, info: info, pointCount: cache.PointCount}
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

// warmupCacheCmd scrolls the entire Qdrant collection and persists it to the
// on-disk cache. This is triggered by `/cache warmup` so subsequent offline
// queries can be served from cache. On completion it transitions back to idle.
func (m *Model) warmupCacheCmd() tea.Cmd {
	return func() tea.Msg {
		// Use a zero-vector to satisfy the function signature; we don't
		// actually need a query for cache warmup. We just scroll + save.
		dummyVector := make([]float32, 0)

		// We need the embedding dimension from the collection to build
		// a proper dummy vector. Fetch one point to discover it.
		_, _, _, err := rag.SearchQdrantFullCorpus(
			m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
			m.collection, dummyVector,
			1,
			"", "",
			m.qdrantPoints,
			true, // force refresh
			nil,
		)
		if err != nil {
			return appErrMsg{err: err, reason: "Cache warmup failed", stage: "search"}
		}

		// Read back the cache to update the header info.
		if cache, _, cerr := rag.LoadCorpusCache(m.collection); cerr == nil && cache != nil {
			age := time.Since(cache.CachedAt).Truncate(time.Second)
			m.cacheInfo = fmt.Sprintf("✓ %s pts (%s old)", formatNumber(cache.PointCount), age)
		}

		return systemLogMsg{
			content:  fmt.Sprintf("Cache warmup complete for collection '%s'", m.collection),
			feedback: fmt.Sprintf("Cache warmed for %s", m.collection),
		}
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
		sort.SliceStable(scoredPoints, func(i, j int) bool {
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

	system := m.systemPrompt
	if system == "" {
		if m.ragMode == "hybrid" {
			system = "You are QQuestio, an advanced, highly articulate enterprise hybrid RAG assistant. " +
				"Your goal is to provide comprehensive, detailed, and well-structured answers by synthesizing the retrieved context chunks with your general knowledge and model weights.\n" +
				"Adhere to the following guidelines:\n" +
				"1. BLEND LOCAL CONTEXT WITH GENERAL KNOWLEDGE: Combine the facts directly mentioned in the 'Retrieved Context Chunks' below with your general knowledge base to construct a complete, rich, and detailed explanation.\n" +
				"2. EXPLICIT DISTINCTION & ATTRIBUTION: You must explicitly state which parts of your explanation are retrieved directly from local sources, and which parts are contributed by your general knowledge. Cite local sources inline (e.g. '[Document: filename | Chunk X]').\n" +
				"3. CREATIVE & PROFESSIONAL EXTENSION: Connect different parts of the context with narrative explanation, background info, and details, ensuring a highly helpful and well-structured result.\n" +
				"4. TRANSPARENT UNCERTAINTY: If you are unsure or are presenting general knowledge info, clearly label it as general/historical knowledge to maintain clarity."
		} else {
			system = "You are QQuestio, a state-of-the-art enterprise RAG assistant. " +
				"Your primary mandate is to perform deep, highly rigorous, and completely grounded research to provide extremely accurate answers. " +
				"You are operating in a STRICT closed-book RAG environment. You must adhere to the following ABSOLUTE guidelines to prevent hallucinations:\n" +
				"1. STRICT CONSTRAINTS: Rely ONLY on the clear facts directly mentioned in the 'Retrieved Context Chunks' below. Do NOT assume, extrapolate, speculate, or invent any details, facts, numbers, dates, names, or URLs. If a fact is not explicitly stated in the context, treat it as completely untrue and non-existent.\n" +
				"2. ABSOLUTELY NO GENERAL KNOWLEDGE OR HALLUCINATION: You are forbidden from using any external or general knowledge not contained in the provided chunks. Do NOT make up any information under any circumstances. If the retrieved chunks do not contain the answer, you must state: 'I am sorry, but the retrieved context does not contain enough information to answer this question.' Do not attempt to enrich the answer with general knowledge or speculation.\n" +
				"3. SUPER DEEP & METICULOUS ANALYSIS: Carefully examine every single line of the retrieved context. Synthesize details across multiple chunks, cross-reference them, and provide a comprehensive, highly thorough, well-reasoned, and step-by-step grounded answer.\n" +
				"4. COMPULSORY SOURCE CITATION: Every claim, statement of fact, or explanation you write must be directly followed by an inline citation to its source document and chunk (e.g. '[Document: filename | Chunk X]'). If a statement cannot be cited, do not write it.\n" +
				"5. HIGHEST GROUNDING FIDELITY: Treat the retrieved context as the absolute and only source of truth. Prioritize absolute factual correctness over creative writing or helpfulness."
		}
	}
	msgs = append(msgs, rag.ChatMessage{Role: "system", Content: system})

	// 2. Conversation history (multi-turn)
	for i := 0; i < len(m.history); i++ {
		turn := m.history[i]
		if turn.Role == "user" {
			// Locate subsequent assistant turn to extract references if any
			var pastRefs []rag.QdrantPoint
			if i+1 < len(m.history) && m.history[i+1].Role == "assistant" {
				pastRefs = m.history[i+1].References
			}

			var histContextBuilder strings.Builder
			if len(pastRefs) > 0 {
				histContextBuilder.WriteString("Retrieved Context Chunks from Knowledge Base:\n")
				for idx, pt := range pastRefs {
					source := extractDocumentName(pt.Payload)
					textStr := pt.ExtractText()
					pointIDStr := fmt.Sprintf("%v", pt.ID)

					docName := source
					if docName == "" {
						docName = fmt.Sprintf("ID %s", pointIDStr)
					}
					histContextBuilder.WriteString(fmt.Sprintf("--- Chunk %d | Document: %s ---\n%s\n", idx+1, docName, textStr))
				}
				histContextBuilder.WriteString("---\n\n")
			}
			userMsg := fmt.Sprintf("%sQuestion: %s", histContextBuilder.String(), turn.Content)
			msgs = append(msgs, rag.ChatMessage{Role: "user", Content: userMsg})
		} else if turn.Role == "assistant" {
			msgs = append(msgs, rag.ChatMessage{Role: "assistant", Content: turn.Content})
		}
	}

	// 3. Current user query with RAG context injected as structured chunks
	var contextBuilder strings.Builder
	if len(m.lastPoints) > 0 {
		contextBuilder.WriteString("Retrieved Context Chunks from Knowledge Base:\n")
		for i, pt := range m.lastPoints {
			source := extractDocumentName(pt.Payload)
			textStr := pt.ExtractText()
			pointIDStr := fmt.Sprintf("%v", pt.ID)

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
