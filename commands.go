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
// It first condenses/rewrites the query for conversational context if history exists.
func (m *Model) generateEmbeddingCmd(query string) tea.Cmd {
	return func() tea.Msg {
		retrievalQuery := m.condenseQueryForRetrieval(m.ctx, query)
		vector, err := rag.GetEmbedding(m.ctx, m.cfg.EmbeddingURL, m.cfg.EmbeddingAPIKey, m.cfg.EmbeddingModel, retrievalQuery)
		if err != nil {
			return appErrMsg{err: err, reason: "Failed to generate embedding", stage: "embedding"}
		}
		return embeddingMsg{vector: vector}
	}
}

// condenseQueryForRetrieval returns the query text to embed for retrieval.
// It implements a three-layer fallback strategy:
// 1. LLM rewrite (default when history exists)
// 2. Heuristic fallback (pronoun detection)
// 3. Raw query passthrough
func (m *Model) condenseQueryForRetrieval(ctx context.Context, raw string) string {
	// Layer 3: No rewrite if disabled or no history
	if m.cfg.QueryRewrite == "off" || len(m.history) == 0 {
		return raw
	}

	// Layer 1: LLM rewrite (default)
	if m.cfg.QueryRewrite == "llm" {
		// Build messages for LLM rewrite
		var messages []rag.ChatMessage
		messages = append(messages, rag.ChatMessage{
			Role: "system",
			Content: "Rewrite the user's follow-up question as a single standalone search question that contains all necessary context. Output ONLY the rewritten question, no quotes, no commentary. If the question is already standalone, output it unchanged.",
		})

		// Add last few turns (up to 4)
		startIdx := 0
		if len(m.history) > 4 {
			startIdx = len(m.history) - 4
		}
		for i := startIdx; i < len(m.history); i++ {
			turn := m.history[i]
			if turn.Role == "user" || turn.Role == "assistant" {
				content := turn.Content
				if len(content) > 500 {
					content = content[:500]
				}
				messages = append(messages, rag.ChatMessage{
					Role:    turn.Role,
					Content: content,
				})
			}
		}
		messages = append(messages, rag.ChatMessage{
			Role:    "user",
			Content: raw,
		})

		// Call LLM with timeout
		ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		defer cancel()

		rewritten, err := rag.ChatComplete(ctx, m.cfg.OpenAIURL, m.cfg.OpenAIAPIKey, m.cfg.OpenAIModel, messages, 128)
		if err == nil && rewritten != "" && !strings.Contains(rewritten, "\n") {
			return rewritten
		}
		// Fall through to heuristic on error or empty result
	}

	// Layer 2: Heuristic fallback
	// Check if query is "follow-up shaped"
	isFollowUp := false
	fields := strings.Fields(raw)
	if len(fields) <= 8 {
		isFollowUp = true
	} else {
		lower := strings.ToLower(raw)
		singleWords := map[string]bool{
			"it": true, "its": true, "this": true, "that": true, "these": true, "those": true,
			"he": true, "she": true, "they": true, "them": true, "his": true, "her": true,
			"their": true, "above": true, "earlier": true, "previous": true, "more": true, "also": true,
		}
		for _, f := range strings.Fields(lower) {
			cleaned := strings.Trim(f, ".,!?:;\"'()[]{}")
			if singleWords[cleaned] {
				isFollowUp = true
				break
			}
		}
		if !isFollowUp {
			multiWordPhrases := []string{"and what", "what about"}
			for _, phrase := range multiWordPhrases {
				if strings.Contains(lower, phrase) {
					isFollowUp = true
					break
				}
			}
		}
	}

	if isFollowUp {
		// Find previous user turn
		for i := len(m.history) - 1; i >= 0; i-- {
			if m.history[i].Role == "user" {
				return m.history[i].Content + "\n" + raw
			}
		}
	}

	// Default: return raw query
	return raw
}

// searchQdrantCmd (Stage 2) performs similarity search against Qdrant collection
// with query-time context expansion.
//
// The active path is chosen automatically based on m.searchCap and m.searchMode:
func (m *Model) computeSearchDocs(docs, expand int) int {
	if expand < 0 {
		expand = 0
	}

	// If no reranker is configured/enabled, the search limit is exactly the user's limit.
	// We do not need a larger candidate pool because there is no second-stage reranking.
	if m.cfg.RerankerURL == "" || m.disableReranker {
		return docs
	}

	// Compute the base candidate pool size before expansion scaling.
	basePool := docs
	if m.rerankerPool > 0 {
		basePool = m.rerankerPool
	} else {
		basePool = docs * 5
		if basePool < 50 {
			basePool = 50
		}
	}

	searchDocs := basePool
	if expand > 0 {
		// Scale candidate pool by (expand + 1) to ensure a sufficiently wide
		// candidate pool is pulled from the full corpus to account for
		// similarity spanning adjacent chunks.
		searchDocs = basePool * (expand + 1)

		// Cap the scaled pool to prevent excessive resource usage, while
		// honoring explicitly set large reranker pools.
		capVal := 500
		if m.rerankerPool > capVal {
			capVal = m.rerankerPool
		}
		if searchDocs > capVal {
			searchDocs = capVal
		}
	}

	if searchDocs < docs {
		searchDocs = docs
	}
	return searchDocs
}

// computeRerankerPool returns the number of primary candidates to forward to
// the reranker API. This is intentionally decoupled from computeSearchDocs:
// the search pool is large to maximise recall; the reranker pool is smaller
// to protect calibration accuracy. Small-to-medium reranker models (e.g.
// Qwen3-Reranker-4B) degrade noticeably beyond ~20 candidates per call.
// The user can override this cap via /rerankerpool <N>.
func (m *Model) computeRerankerPool() int {
	if m.rerankerPool > 0 {
		return m.rerankerPool
	}
	pool := m.searchLimit * 3
	if pool < 10 {
		pool = 10
	}
	if pool > 20 {
		pool = 20
	}
	return pool
}

//   - Cap set (m.searchCap > 0): use Qdrant's HNSW query API with the cap as
//     the candidate pool size. Fast, but bounded by the cap (approximate recall).
//
//   - No cap, mode "local" (opt-in via /cap local): use SearchQdrantFullCorpus,
//     which streams every vector via /points/scroll and computes cosine
//     similarity on the client using all local CPU cores. Slower (network-bound),
//     but works when Qdrant refuses params.exact=true.
//
//   - No cap, default: use SearchWithContextExpansion, which calls
//     SearchQdrantExact (params.exact=true) for the primary top-N matches and
//     then fetches ±m.searchExpand adjacent chunks from the same document
//     via batched parallel scroll requests. This solves the "fragmented
//     context" problem: a single top match is a fragment; with expansion
//     the LLM sees the complete surrounding slice of the source document.
//
// The user-requested `docs` (m.searchLimit) is the final primary return count.
// Expansion is governed by m.searchExpand (0 = disabled, 1 = ±1 neighbor, etc).
func (m *Model) searchQdrantCmd(vector []float32) tea.Cmd {
	return func() tea.Msg {
		docs := m.searchLimit
		expand := m.searchExpand
		searchDocs := m.computeSearchDocs(docs, expand)
		if len(m.exactPhrases) > 0 {
			results, points, err := rag.SearchQdrantExactPhrases(
				m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
				m.collection, m.exactPhrases,
				searchDocs,
				m.filterKey, m.filterValue,
			)
			if err != nil {
				return appErrMsg{err: err, reason: "Exact phrase search failed", stage: "search"}
			}
			return searchResultMsg{context: results, points: points, primaryPoints: points, expansionMap: rag.ExpansionMap{}, expand: 0}
		}

		// HNSW path: user explicitly opted into a cap (approximate, bounded).
		// We bypass context expansion here because HNSW results are
		// already approximate and the expansion would still be
		// corpus-wide, so we just call SearchQdrant directly. We still
		// pass the expand=0 so the message shape is uniform downstream.
		// Honor /search exact in the capped path for exact scoring over
		// the capped candidate pool.
		if m.searchCap > 0 && m.exactPhrase == "" {
			candidateLimit := m.searchCap
			if candidateLimit < searchDocs {
				candidateLimit = searchDocs
			}
			exact := m.searchMode == "exact"
			results, points, err := rag.SearchQdrant(
				m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
				m.collection, vector,
				candidateLimit, searchDocs,
				m.filterKey, m.filterValue,
				exact,
			)
			if err != nil {
				return appErrMsg{err: err, reason: "Qdrant search failed", stage: "search"}
			}
			return searchResultMsg{context: results, points: points, primaryPoints: points, expansionMap: rag.ExpansionMap{}, expand: 0}
		}

		// Local-fallback path: user explicitly requested client-side brute force
		// via /cap local. Streams every vector and scores locally on all CPU cores.
		// Context expansion here would also be possible but it is a path only
		// used as a fallback when Qdrant refuses params.exact=true.
		if m.searchMode == "local" || m.exactPhrase != "" {
			forceRefresh := m.cacheForceRefresh
			results, points, fromCache, err := rag.SearchQdrantFullCorpus(
				m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
				m.collection, vector,
				searchDocs,
				m.filterKey, m.filterValue,
				m.qdrantPoints,
				forceRefresh,
				nil,
				m.exactPhrase,
			)
			if err != nil {
				return appErrMsg{err: err, reason: "Full-corpus local search failed", stage: "search"}
			}
			m.cacheForceRefresh = false
			if fromCache {
				if cache, _, cerr := rag.LoadCorpusCache(m.cfg.QdrantURL, m.collection); cerr == nil && cache != nil {
					age := time.Since(cache.CachedAt).Truncate(time.Second)
					m.cacheInfo = fmt.Sprintf("✓ %s pts (%s old)", formatNumber(cache.PointCount), age)
					m.cacheFilterAtWarmup = cache.FilterAtWarmup
				}
			}
			return searchResultMsg{context: results, points: points, primaryPoints: points, expansionMap: rag.ExpansionMap{}, expand: 0}
		}

		// DEFAULT no-cap path: full-corpus search + context expansion.
		// We use the detailed version so the rerank step can rerank only
		// the primary top-N (not the already-expanded set) and re-apply
		// ±expand around the reranked top-K.
		// In "auto" mode (default), we use exact=true when uncapped to match
		// documented behavior: exact brute-force when cap=0, HNSW when capped.
		exact := m.searchMode == "exact" || m.searchMode == "auto"
		res, err := rag.SearchWithContextExpansionDetailed(
			m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
			m.collection, vector,
			searchDocs, expand,
			m.filterKey, m.filterValue,
			exact,
		)
		if err != nil {
			return appErrMsg{err: err, reason: "Qdrant context-expanded search failed", stage: "search"}
		}
		// Boost phrase matches if there are quoted phrases (but not full exact search)
		if len(m.exactPhrases) > 0 && !m.forceExactPhrase {
			res.PrimaryPoints = rag.BoostPhraseMatches(res.PrimaryPoints, m.exactPhrases)
		}
		return searchResultMsg{
			context:       res.Context,
			points:        res.ExpandedPoints,
			primaryPoints: res.PrimaryPoints,
			expansionMap:  res.ExpansionMap,
			expand:        expand,
		}
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
		cache, _, err := rag.LoadCorpusCache(m.cfg.QdrantURL, m.collection)
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
		// Keep a fallback estimate for providers that do not return usage
		// metadata in their streaming response.
		m.currentPromptEstimate = estimateChatMessageTokens(messages)
		ctx, cancel := context.WithCancel(m.ctx)
		m.cancelRequest = cancel

		reader, err := rag.StartLiteLLMStream(ctx, m.cfg.OpenAIURL, m.cfg.OpenAIAPIKey, m.cfg.OpenAIModel, m.cfg.OpenAIMaxTokens, m.cfg.ContextLimit, messages)
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
		chunk, reasoning, done, err := m.streamReader.Next()
		if err != nil {
			return appErrMsg{err: err, reason: "Stream read error", stage: "stream"}
		}
		usage := m.streamReader.Usage()
		return streamChunkMsg{
			content:   chunk,
			reasoning: reasoning,
			done:      done,
			usage:     usage,
			hasUsage:  usage.TotalTokens > 0 || usage.PromptTokens > 0 || usage.CompletionTokens > 0,
		}
	}
}

// warmupCacheCmd scrolls the entire Qdrant collection and persists it to the
// on-disk cache. This is triggered by `/cache warmup` so subsequent queries
// can be served from cache. On completion it transitions back to idle.
func (m *Model) warmupCacheCmd() tea.Cmd {
	return func() tea.Msg {
		cache, err := rag.WarmupCorpusCache(
			m.ctx, m.cfg.QdrantURL, m.cfg.QdrantAPIKey,
			m.collection,
			func(processed, total int) {
				// Progress callback - could update status if needed
			},
		)
		if err != nil {
			return appErrMsg{err: err, reason: "Cache warmup failed", stage: "search"}
		}

		// Update the header info with the new cache.
		age := time.Since(cache.CachedAt).Truncate(time.Second)
		m.cacheInfo = fmt.Sprintf("✓ %s pts (%s old)", formatNumber(cache.PointCount), age)
		m.cacheFilterAtWarmup = cache.FilterAtWarmup

		return systemLogMsg{
			content:  fmt.Sprintf("Cache warmup complete for collection '%s' (%s points)", m.collection, formatNumber(cache.PointCount)),
			feedback: fmt.Sprintf("Cache warmed for %s (%s points)", m.collection, formatNumber(cache.PointCount)),
		}
	}
}

// executeSkillCmd runs the named skill asynchronously.
func (m *Model) executeSkillCmd(name, args string) tea.Cmd {
	return func() tea.Msg {
		skill, ok := m.skills.Get(name)
		if !ok {
			return skillResultMsg{err: fmt.Errorf("skill not found: %s", name)}
		}
		res, err := skill.Execute(m.ctx, []byte(args))
		// Truncate skill output to 8 KiB to prevent prompt-size explosion
		const maxOutputSize = 8192
		if len(res) > maxOutputSize {
			res = res[:maxOutputSize] + "\n[... Output truncated to 8 KiB ...]"
		}
		return skillResultMsg{name: name, input: args, output: res, err: err}
	}
}

// rerankPointsCmd (Optional Stage 2.5) reranks the retrieved Qdrant points
// using a generic reranker. Critically, it now:
//
//  1. Reranks only the PRIMARY top-N matches (not the already-expanded set),
//     so the reranker's score reflects pure relevance, not the artificially
//     inflated rank of duplicate adjacent chunks.
//  2. Caps the candidate pool sent to the reranker via computeRerankerPool()
//     so accuracy-sensitive small models (e.g. Qwen3-Reranker-4B) are not
//     overwhelmed by the full widened search pool.
//  3. Uses ExtractPrimaryText() (not ExtractText()) so the reranker receives
//     a single clean passage per document instead of a multi-field blob.
//  4. After reranking, takes the top-`searchLimit` reranked primaries and
//     re-applies ±expand around them using the cached ExpansionMap. This
//     makes /expand actually work in the rerank path: the LLM sees the
//     full document slice around the reranked matches, not just fragments.
//
// Takes the full searchResultMsg (instead of just points) so it has access
// to primaryPoints, expansionMap, and expand.
func (m *Model) rerankPointsCmd(result searchResultMsg) tea.Cmd {
	return func() tea.Msg {
		// Use primary points for reranking; if empty (e.g. HNSW/local paths
		// that don't separate primary from expanded), fall back to the full
		// points slice.
		primaries := result.primaryPoints
		if len(primaries) == 0 {
			primaries = result.points
		}
		if len(primaries) == 0 {
			return rerankResultMsg{context: "", points: nil}
		}

		// Cap the candidate pool sent to the reranker to preserve accuracy.
		// The search pool is intentionally larger for recall; the reranker
		// pool is smaller because reranker models are calibration-sensitive.
		rerankCap := m.computeRerankerPool()
		if len(primaries) > rerankCap {
			primaries = primaries[:rerankCap]
		}

		// Extract texts for reranking using only the primary text (not expanded).
		// This reduces token volume and avoids sending duplicate adjacent chunks.
		var texts []string
		var validIndices []int
		for i, pt := range primaries {
			primaryText := pt.ExtractPrimaryText()
			if primaryText == "" {
				// Skip candidates with empty primary text
				continue
			}
			texts = append(texts, primaryText)
			validIndices = append(validIndices, i)
		}

		if len(texts) == 0 {
			return rerankResultMsg{context: "", points: primaries, degraded: true}
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
			// Graceful degradation: return original points unchanged when reranker is unavailable.
			return rerankResultMsg{context: "", points: primaries, degraded: true}
		}

		// Map rerank scores back to the primaries slice using validIndices.
		scoreMap := make(map[int]float64)
		for i, item := range rerankItems {
			originalIdx := validIndices[i]
			scoreMap[originalIdx] = item.Score
		}

		// Sort the primaries by rerank score descending.
		type scoredPoint struct {
			point rag.QdrantPoint
			score float64
		}
		var scoredPoints []scoredPoint
		for i, pt := range primaries {
			score, ok := scoreMap[i]
			if !ok {
				score = -999.0
			}
			pt.OriginalScore = pt.Score
			pt.Score = float32(score)
			scoredPoints = append(scoredPoints, scoredPoint{point: pt, score: score})
		}
		sort.SliceStable(scoredPoints, func(i, j int) bool {
			return scoredPoints[i].score > scoredPoints[j].score
		})

		// Slice to the user-requested top-K.
		var rerankedTopK []rag.QdrantPoint
		limit := m.searchLimit
		if limit > len(scoredPoints) {
			limit = len(scoredPoints)
		}
		for i := 0; i < limit; i++ {
			rerankedTopK = append(rerankedTopK, scoredPoints[i].point)
		}

		// Re-apply ±expand around the reranked top-K using the cached map.
		// This is the key fix for the "expand does nothing" bug in the
		// rerank path: instead of throwing away the expansion (as the old
		// code did by re-extracting text from `finalPoints`), we expand
		// the reranked primaries with the chunks we already pulled from
		// Qdrant during the original search. Pure CPU, no extra network.
		finalContext, finalPoints := rag.ApplyExpansionToPrimaries(
			rerankedTopK,
			result.expansionMap,
			result.expand,
		)
		if finalPoints == nil {
			// Fallback: reranked top-K without expansion (shouldn't happen
			// in practice, but be defensive).
			finalPoints = rerankedTopK
			var sb strings.Builder
			for _, pt := range finalPoints {
				if t := pt.ExtractText(); t != "" {
					sb.WriteString(t)
					sb.WriteString("\n---\n")
				}
			}
			finalContext = sb.String()
		}

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
	if toolPrompt := m.skills.ForPrompt(); toolPrompt != "" {
		system += "\n\n" + toolPrompt + "\n" +
			"If you decide to use any of the available tools, you must output a tool call block using the exact syntax:\n" +
			"CALL: <tool_name> <arguments>\n" +
			"For example:\n" +
			"CALL: bash ls -la\n" +
			"Do not output anything else when calling a tool. Stop generating immediately after outputting the CALL block."
	}
	msgs = append(msgs, rag.ChatMessage{Role: "system", Content: system})

	// 2. Conversation history (multi-turn)
	// Keep historical references for the transcript and references panel, but do
	// not replay them into the active prompt. Retrieval is turn-scoped; replaying
	// old results contaminates refined searches with stale documents.
	for i := 0; i < len(m.history); i++ {
		turn := m.history[i]
		if turn.Role == "user" {
			msgs = append(msgs, rag.ChatMessage{Role: "user", Content: "Question: " + turn.Content})
		} else if turn.Role == "assistant" {
			msgs = append(msgs, rag.ChatMessage{Role: "assistant", Content: turn.Content})
		} else if turn.Role == "system" && strings.HasPrefix(turn.Content, "[ Context compacted") {
			// Include compaction summaries as user messages (not system) for better compatibility
			msgs = append(msgs, rag.ChatMessage{
				Role: "user",
				Content: "[ Summary of earlier conversation ]\n" + turn.Content,
			})
		}
		// Other system turns (slash feedback, warmup logs, skill logs) remain UI-only
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

// checkLLMInfoCmd tests the connection to the OpenAI/LLM endpoint and returns the result/error.
func (m *Model) checkLLMInfoCmd() tea.Cmd {
	return func() tea.Msg {
		err := rag.CheckLLMConnection(m.ctx, m.cfg.OpenAIURL, m.cfg.OpenAIAPIKey, m.cfg.OpenAIModel)
		return llmInfoMsg{err: err}
	}
}

// checkEmbedderInfoCmd tests the connection to the embedding endpoint and returns the result/error.
func (m *Model) checkEmbedderInfoCmd() tea.Cmd {
	return func() tea.Msg {
		err := rag.CheckEmbeddingConnection(m.ctx, m.cfg.EmbeddingURL, m.cfg.EmbeddingAPIKey, m.cfg.EmbeddingModel)
		return embedderInfoMsg{err: err}
	}
}
