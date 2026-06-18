package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"sort"
	"strings"
	"time"
)

type QdrantMatch struct {
	Value interface{} `json:"value,omitempty"`
}

type QdrantFieldCondition struct {
	Key   string      `json:"key"`
	Match QdrantMatch `json:"match"`
}

type QdrantFilter struct {
	Must   []QdrantFieldCondition `json:"must,omitempty"`
	Should []QdrantFieldCondition `json:"should,omitempty"`
}

// QdrantSearchParams controls server-side search behavior.
type QdrantSearchParams struct {
	Exact bool `json:"exact,omitempty"`
}

type QdrantQueryRequest struct {
	Query       []float32          `json:"query"`
	Filter      *QdrantFilter      `json:"filter,omitempty"`
	Limit       int                `json:"limit"`
	WithPayload bool               `json:"with_payload"`
	Params      *QdrantSearchParams `json:"params,omitempty"`
}

type QdrantSearchRequest struct {
	Vector      []float32           `json:"vector"`
	Filter      *QdrantFilter       `json:"filter,omitempty"`
	Limit       int                 `json:"limit"`
	WithPayload bool                `json:"with_payload"`
	Params      *QdrantSearchParams `json:"params,omitempty"`
}

type QdrantPoint struct {
	ID      interface{}            `json:"id"`
	Payload map[string]interface{} `json:"payload"`
	Score   float32                `json:"score"`
	Vector  []float32              `json:"vector,omitempty"`
}

type QdrantQueryResponse struct {
	ResultRaw json.RawMessage `json:"result"`
	Result    []QdrantPoint   `json:"-"`
	Status    string          `json:"status"`
}

// UnmarshalJSON implements custom JSON unmarshaling to dynamically support
// both flat result lists (Search API format) and points-wrapped result objects (Query API format).
func (q *QdrantQueryResponse) UnmarshalJSON(data []byte) error {
	type Alias QdrantQueryResponse
	var aux Alias
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	q.Status = aux.Status
	q.ResultRaw = aux.ResultRaw

	if len(q.ResultRaw) == 0 {
		return nil
	}

	trimmed := bytes.TrimSpace(q.ResultRaw)
	if len(trimmed) == 0 {
		return nil
	}

	// Format 1: Flat array of points [ {id, payload, score}, ... ]
	if trimmed[0] == '[' {
		var points []QdrantPoint
		if err := json.Unmarshal(trimmed, &points); err != nil {
			return fmt.Errorf("failed to unmarshal result array: %w", err)
		}
		q.Result = points
		return nil
	}

	// Format 2: Object wrapping points array { "points": [ ... ] }
	if trimmed[0] == '{' {
		var wrapper struct {
			Points []QdrantPoint `json:"points"`
		}
		if err := json.Unmarshal(trimmed, &wrapper); err != nil {
			return fmt.Errorf("failed to unmarshal result object: %w", err)
		}
		q.Result = wrapper.Points
		return nil
	}

	return fmt.Errorf("unexpected json type for result: %s", string(q.ResultRaw))
}

// SearchQdrant performs a vector similarity search in Qdrant.
//
// The function intentionally DECOUPLES the search scope from the return count:
//
//   - candidateLimit: the number of candidates Qdrant should consider during the
//     HNSW / exact nearest-neighbor search. This is sent to Qdrant as the
//     request `limit` and directly determines how much of the corpus Qdrant
//     looks at. A small value here destroys recall because Qdrant will only
//     compute the top-N nearest neighbors and never look at the rest of the
//     collection. Callers should pass the full corpus size (or a user-configured
//     cap) here.
//
//   - docs: the number of results to RETURN to the caller. After the response
//     is parsed, the points slice is truncated to at most `docs` entries
//     before text extraction. This is the user-facing "how many context
//     documents" value (e.g. the `/limit` setting).
//
// Request:  POST /collections/{collection}/points/query
//
//	{ "query": vector, "limit": candidateLimit, "with_payload": true }
//
// Response: Extract ONLY the text field from each point's payload.
// Returns:  Concatenated text payloads separated by "\n---\n", the points list (truncated to docs), and an error if any.
func SearchQdrant(ctx context.Context, baseURL, apiKey, collection string, vector []float32, candidateLimit, docs int, filterKey, filterValue string) (string, []QdrantPoint, error) {
	url := fmt.Sprintf("%s/collections/%s/points/query", strings.TrimSuffix(baseURL, "/"), collection)

	var filter *QdrantFilter
	if filterKey != "" && filterValue != "" {
		isDocKey := false
		docKeys := []string{"file_name", "filename", "file_path", "title", "document", "doc_name", "source", "source_file"}
		for _, dk := range docKeys {
			if strings.ToLower(filterKey) == dk {
				isDocKey = true
				break
			}
		}

		if isDocKey || filterKey == "*" || filterKey == "any_file" {
			var shouldConds []QdrantFieldCondition
			for _, dk := range docKeys {
				shouldConds = append(shouldConds, QdrantFieldCondition{
					Key: dk,
					Match: QdrantMatch{
						Value: filterValue,
					},
				})
			}
			filter = &QdrantFilter{
				Should: shouldConds,
			}
		} else {
			filter = &QdrantFilter{
				Must: []QdrantFieldCondition{
					{
						Key: filterKey,
						Match: QdrantMatch{
							Value: filterValue,
						},
					},
				},
			}
		}
	}

	reqBody := QdrantQueryRequest{
		Query:       vector,
		Filter:      filter,
		Limit:       candidateLimit,
		WithPayload: true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return "", nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("api-key", apiKey)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", nil, fmt.Errorf("HTTP status error: %d %s", resp.StatusCode, resp.Status)
	}

	var respBody QdrantQueryResponse
	if err := json.NewDecoder(resp.Body).Decode(&respBody); err != nil {
		return "", nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Truncate the candidate pool down to the user-requested return count.
	// This is the critical step that decouples search scope (candidateLimit)
	// from return count (docs). We may have asked Qdrant for thousands of
	// candidates; we now keep only the top `docs` for the LLM context.
	if docs > 0 && len(respBody.Result) > docs {
		respBody.Result = respBody.Result[:docs]
	}

	var texts []string
	for _, pt := range respBody.Result {
		textStr := pt.ExtractText()
		if textStr != "" {
			texts = append(texts, textStr)
		}
	}

	return strings.Join(texts, "\n---\n"), respBody.Result, nil
}

// SearchQdrantExact performs a TRUE full-corpus semantic search using Qdrant's
// native brute-force mode (params.exact=true). Unlike the scroll+client-side
// cosine approach, this lets Qdrant score every single vector server-side
// using SIMD-optimized math and returns only the top-N results.
//
// This is the correct API for full-corpus recall without a cap:
//   - Every vector in the collection is scored (no HNSW approximation).
//   - Only the top `docs` results are returned (no 1.4M vector transfer).
//   - Sub-second latency even for million-scale collections.
//
// Parameters:
//   - docs: number of top results to return.
//   - filterKey/filterValue: optional metadata filter pushed down to Qdrant.
//
// Returns: (context string, top points, error).
func SearchQdrantExact(
	ctx context.Context,
	baseURL, apiKey, collection string,
	vector []float32,
	docs int,
	filterKey, filterValue string,
) (string, []QdrantPoint, error) {
	url := fmt.Sprintf("%s/collections/%s/points/search",
		strings.TrimSuffix(baseURL, "/"), collection)

	var filter *QdrantFilter
	if filterKey != "" && filterValue != "" {
		filter = buildFilter(filterKey, filterValue)
	}

	reqBody := QdrantSearchRequest{
		Vector:      vector,
		Filter:      filter,
		Limit:       docs,
		WithPayload: true,
		Params:      &QdrantSearchParams{Exact: true},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	log.Printf("[QdrantExact] Querying collection %s at URL %s (limit: %d, exact: true, filterKey: %q, filterValue: %q)",
		collection, url, docs, filterKey, filterValue)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return "", nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("api-key", apiKey)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[QdrantExact] HTTP request failed: %v", err)
		return "", nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("[QdrantExact] HTTP response: %d %s", resp.StatusCode, resp.Status)

	if resp.StatusCode != http.StatusOK {
		return "", nil, fmt.Errorf("HTTP status error: %d %s", resp.StatusCode, resp.Status)
	}

	var respBody QdrantQueryResponse
	if err := json.NewDecoder(resp.Body).Decode(&respBody); err != nil {
		log.Printf("[QdrantExact] Failed to decode response: %v", err)
		return "", nil, fmt.Errorf("failed to decode response: %w", err)
	}

	log.Printf("[QdrantExact] Search returned %d points", len(respBody.Result))
	for i, pt := range respBody.Result {
		log.Printf("[QdrantExact]   point[%d] score=%f id=%v payloadKeys=%v",
			i, pt.Score, pt.ID, getPayloadKeys(pt.Payload))
	}

	var texts []string
	for _, pt := range respBody.Result {
		textStr := pt.ExtractText()
		if textStr != "" {
			texts = append(texts, textStr)
		}
	}

	return strings.Join(texts, "\n---\n"), respBody.Result, nil
}

func getPayloadKeys(payload map[string]interface{}) []string {
	if payload == nil {
		return nil
	}
	keys := make([]string, 0, len(payload))
	for k := range payload {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ExtractText extracts the main text payload content from a QdrantPoint.
// It gathers text from primary keys (text, content, document, etc.) and
// other non-metadata keys to build a large, extensive, and complete context.
func (pt QdrantPoint) ExtractText() string {
	if pt.Payload == nil {
		return ""
	}

	var parts []string
	seen := make(map[string]bool)

	// 1. Gather from primary keys first, in order of priority
	primaryKeys := []string{"text", "content", "document", "page_content", "description", "body", "passage", "chunk", "context"}
	for _, key := range primaryKeys {
		if val, ok := pt.Payload[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				sClean := strings.TrimSpace(s)
				if !seen[sClean] {
					parts = append(parts, sClean)
					seen[sClean] = true
				}
			}
		}
	}

	// 2. Sort keys to iterate deterministically and prevent random Go map iteration order
	var keys []string
	for k := range pt.Payload {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// 3. Gather from other non-metadata string keys to enrich context and prevent hallucinations
	var otherParts []string
	for _, k := range keys {
		v := pt.Payload[k]
		kl := strings.ToLower(k)
		isMeta := false
		// Skip known metadata keys
		for _, mk := range []string{"file", "filename", "file_name", "file_path", "source", "title", "url", "path", "id", "score", "page", "author", "date", "created_at"} {
			if kl == mk || strings.Contains(kl, "id") || strings.Contains(kl, "score") {
				isMeta = true
				break
			}
		}
		if !isMeta {
			if s, ok := v.(string); ok && len(s) > 5 {
				sClean := strings.TrimSpace(s)
				if !seen[sClean] {
					otherParts = append(otherParts, sClean)
					seen[sClean] = true
				}
			}
		}
	}

	// Combine all collected parts
	allParts := append(parts, otherParts...)
	if len(allParts) == 0 {
		return ""
	}

	// Join parts with double newlines to maintain structure and lines
	return strings.Join(allParts, "\n\n")
}

type QdrantCollectionInfo struct {
	Result struct {
		Status       string `json:"status"`
		PointsCount  int    `json:"points_count"`
		VectorsCount int    `json:"vectors_count"`
	} `json:"result"`
	Status string `json:"status"`
}

// ScrollRequest is the body for POST /collections/{name}/points/scroll.
type ScrollRequest struct {
	Limit       int           `json:"limit"`
	WithPayload bool          `json:"with_payload"`
	WithVector  bool          `json:"with_vector"`
	Offset      interface{}   `json:"offset,omitempty"`
	Filter      *QdrantFilter `json:"filter,omitempty"`
}

// ScrollResponse is the response from POST /collections/{name}/points/scroll.
type ScrollResponse struct {
	Result struct {
		Points         []QdrantPoint `json:"points"`
		NextPageOffset interface{}   `json:"next_page_offset"`
	} `json:"result"`
	Status string `json:"status"`
}

// SearchQdrantFullCorpus performs a TRUE full-corpus semantic search.
//
// It achieves completeness regardless of Qdrant's server-side
// `max_search_limit` by streaming every point in the collection via the
// /points/scroll API, computing cosine similarity client-side in Go, and
// keeping a top-N heap. The complete corpus is then cached to disk so that
// subsequent queries are pure-CPU and never touch the network.
//
// Parameters:
//   - ctx: cancellation context (checked on every batch so Esc-Esc aborts).
//   - collection: the Qdrant collection to search.
//   - vector: the query embedding (must be the same dim as the collection's vectors).
//   - docs: number of top results to return.
//   - filterKey / filterValue: optional metadata filter. When served from cache,
//     the filter is applied client-side; when served from Qdrant, it is pushed
//     down to the scroll API.
//   - livePointCount: the current /collections/{name} points_count (0 if
//     unknown). Used to decide whether the on-disk cache is stale.
//   - forceRefresh: if true, ignore the on-disk cache and re-scroll from Qdrant.
//
// Returns: (context string, top points, fromCache bool, error).
// ProgressFunc is a callback invoked as the full-corpus search streams
// points from Qdrant or scores them. The first argument is the number of
// points processed so far, the second is the running total (or 0 if
// unknown yet, e.g. during the scroll phase). Implementations should be
// cheap and non-blocking.
type ProgressFunc func(processed, total int)

func SearchQdrantFullCorpus(
	ctx context.Context,
	baseURL, apiKey, collection string,
	vector []float32,
	docs int,
	filterKey, filterValue string,
	livePointCount int,
	forceRefresh bool,
	progress ProgressFunc,
) (string, []QdrantPoint, bool, error) {
	// 1. Try the on-disk cache first (unless caller forced a refresh).
	if !forceRefresh {
		cache, cachedPoints, err := LoadCorpusCache(collection)
		if err == nil && cache != nil {
			// Stale check: if the live count differs from the cached count
			// (and we actually have a live count), the cache is stale.
			stale := false
			if livePointCount > 0 && cache.PointCount != livePointCount {
				stale = true
			}
			if !stale {
				// Apply the filter client-side.
				filtered := applyFilter(cachedPoints, filterKey, filterValue)
				// Compute top-N by cosine similarity.
				top, err := topNByCosine(vector, filtered, docs, func(p, total int) {
					if progress != nil {
						progress(p, total)
					}
				})
				if err != nil {
					return "", nil, false, err
				}
				texts := extractTexts(top)
				return strings.Join(texts, "\n---\n"), top, true, nil
			}
		}
	}

	// 2. Cache miss / stale / forced: scroll the entire collection from Qdrant.
	points, err := scrollAllPoints(ctx, baseURL, apiKey, collection, filterKey, filterValue, progress)
	if err != nil {
		return "", nil, false, fmt.Errorf("full-corpus scroll failed: %w", err)
	}
	if len(points) == 0 {
		return "", nil, false, nil
	}

	// 3. Persist to cache (best effort; don't fail the search if the disk is full).
	dim := len(vector)
	if dim == 0 && len(points) > 0 {
		dim = len(points[0].Vector)
	}
	if dim > 0 {
		_ = SaveCorpusCache(collection, dim, points)
	}

	// 4. Compute top-N.
	total := len(points)
	if progress != nil {
		progress(total, total)
	}
	top, err := topNByCosine(vector, points, docs, func(p, _ int) {
		if progress != nil {
			progress(p, total)
		}
	})
	if err != nil {
		return "", nil, false, err
	}
	texts := extractTexts(top)
	return strings.Join(texts, "\n---\n"), top, false, nil
}

// scrollAllPoints streams every point in the collection from Qdrant's
// /points/scroll endpoint, batching at 1000 points per request. Honors ctx
// cancellation between batches.
func scrollAllPoints(
	ctx context.Context,
	baseURL, apiKey, collection string,
	filterKey, filterValue string,
	progress ProgressFunc,
) ([]QdrantPoint, error) {
	url := fmt.Sprintf("%s/collections/%s/points/scroll", strings.TrimSuffix(baseURL, "/"), collection)

	var filter *QdrantFilter
	if filterKey != "" && filterValue != "" {
		filter = buildFilter(filterKey, filterValue)
	}

	const batchSize = 10000
	var all []QdrantPoint
	var offset interface{}
	client := &http.Client{Timeout: 60 * time.Second}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		reqBody := ScrollRequest{
			Limit:       batchSize,
			WithPayload: true,
			WithVector:  true,
			Offset:      offset,
			Filter:      filter,
		}
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal scroll request: %w", err)
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create scroll request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			req.Header.Set("api-key", apiKey)
		}

		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("scroll HTTP request failed: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			body := make([]byte, 512)
			n, _ := resp.Body.Read(body)
			resp.Body.Close()
			return nil, fmt.Errorf("scroll HTTP status error: %d %s (body: %s)",
				resp.StatusCode, resp.Status, string(body[:n]))
		}

		var scrollResp ScrollResponse
		if err := json.NewDecoder(resp.Body).Decode(&scrollResp); err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("failed to decode scroll response: %w", err)
		}
		resp.Body.Close()

		all = append(all, scrollResp.Result.Points...)
		if progress != nil {
			progress(len(all), 0)
		}

		// next_page_offset is null when there are no more points.
		if scrollResp.Result.NextPageOffset == nil {
			break
		}
		offset = scrollResp.Result.NextPageOffset
	}
	return all, nil
}

// buildFilter constructs the Qdrant filter the same way SearchQdrant does,
// so the two paths produce equivalent behavior.
func buildFilter(filterKey, filterValue string) *QdrantFilter {
	docKeys := []string{"file_name", "filename", "file_path", "title", "document", "doc_name", "source", "source_file"}
	isDocKey := false
	for _, dk := range docKeys {
		if strings.ToLower(filterKey) == dk {
			isDocKey = true
			break
		}
	}
	if isDocKey || filterKey == "*" || filterKey == "any_file" {
		var shouldConds []QdrantFieldCondition
		for _, dk := range docKeys {
			shouldConds = append(shouldConds, QdrantFieldCondition{
				Key:   dk,
				Match: QdrantMatch{Value: filterValue},
			})
		}
		return &QdrantFilter{Should: shouldConds}
	}
	return &QdrantFilter{
		Must: []QdrantFieldCondition{
			{Key: filterKey, Match: QdrantMatch{Value: filterValue}},
		},
	}
}

// applyFilter filters a slice of QdrantPoint by the same semantics as
// buildFilter (doc-keys use Should, others use Must).
func applyFilter(points []QdrantPoint, filterKey, filterValue string) []QdrantPoint {
	if filterKey == "" || filterValue == "" {
		return points
	}
	docKeys := []string{"file_name", "filename", "file_path", "title", "document", "doc_name", "source", "source_file"}
	isDocKey := false
	for _, dk := range docKeys {
		if strings.ToLower(filterKey) == dk {
			isDocKey = true
			break
		}
	}

	out := make([]QdrantPoint, 0, len(points))
	for _, p := range points {
		if isDocKey || filterKey == "*" || filterKey == "any_file" {
			// Should: any doc-key matches.
			matched := false
			for _, dk := range docKeys {
				if v, ok := p.Payload[dk]; ok {
					if s, ok := v.(string); ok && s == filterValue {
						matched = true
						break
					}
				}
			}
			if matched {
				out = append(out, p)
			}
		} else {
			// Must: exact key match.
			if v, ok := p.Payload[filterKey]; ok {
				if s, ok := v.(string); ok && s == filterValue {
					out = append(out, p)
				}
			}
		}
	}
	return out
}

// topNByCosine returns the top `n` points by cosine similarity to `query`.
// If `n <= 0`, all points are returned sorted by score descending.
// The progress callback (if non-nil) is invoked as points are processed.
//
// MAX-SPEED implementation:
//   - Splits the points slice into N chunks (N = runtime.NumCPU()).
//   - Each chunk is scored in its own goroutine using SIMD-friendly loops.
//   - Each worker maintains a local min-heap of size n.
//   - At the end, all per-worker heaps are merged into a single global top-N.
// This saturates all CPU cores for cosine scoring instead of the previous
// single-threaded loop.
func topNByCosine(
query []float32,
points []QdrantPoint,
n int,
progress func(processed, total int),
) ([]QdrantPoint, error) {
	if len(query) == 0 {
		return nil, fmt.Errorf("query vector is empty")
	}
	if len(points) == 0 {
		return nil, nil
	}

	// Pre-compute query norm.
	var qNorm float32
	for _, v := range query {
		qNorm += v * v
	}
	if qNorm == 0 {
		return nil, fmt.Errorf("query vector has zero norm")
	}
	qNorm = float32(math.Sqrt(float64(qNorm)))

	total := len(points)

	// Special case: caller wants all points sorted → still parallelize scoring.
	if n <= 0 || n >= total {
		return topNAllParallel(query, qNorm, points, progress)
	}

	// General case: parallel top-N with per-worker heaps + global merge.
	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}
	if workers > total {
		workers = total
	}

	chunkSize := (total + workers - 1) / workers
	var wg sync.WaitGroup
	localHeaps := make([][]scoredPoint, workers)

	var processed atomic.Int64
	for w := 0; w < workers; w++ {
		start := w * chunkSize
		if start >= total {
			localHeaps[w] = nil
			continue
		}
		end := start + chunkSize
		if end > total {
			end = total
		}
		wg.Add(1)
		go func(lo, hi int, widx int) {
			defer wg.Done()
			h := make([]scoredPoint, 0, n)
			for i := lo; i < hi; i++ {
				sp := scorePoint(query, qNorm, points[i], i)
				if len(h) < n {
					h = append(h, sp)
					if len(h) == n {
						heapifyMin(h)
					}
				} else if sp.score > h[0].score {
					h[0] = sp
					siftDownMin(h, 0)
				}
				// Throttle progress updates: only worker 0 reports and only
				// every ~8192 points per worker to avoid lock contention.
				if widx == 0 && ((i-lo)&8191) == 0 {
					done := int(processed.Add(int64(i - lo + 1)))
					if progress != nil {
						progress(done, total)
					}
				}
			}
			localHeaps[widx] = h
		}(start, end, w)
	}
	wg.Wait()

	if progress != nil {
		progress(total, total)
	}

	// Merge all per-worker heaps into a single global top-N.
	merged := make([]scoredPoint, 0, n)
	for _, h := range localHeaps {
		if len(h) == 0 {
			continue
		}
		for _, sp := range h {
			if len(merged) < n {
				merged = append(merged, sp)
				if len(merged) == n {
					heapifyMin(merged)
				}
			} else if sp.score > merged[0].score {
				merged[0] = sp
				siftDownMin(merged, 0)
			}
		}
	}
	// Sort descending by score for the final slice.
	sort.SliceStable(merged, func(i, j int) bool { return merged[i].score > merged[j].score })
	out := make([]QdrantPoint, len(merged))
	for i, sp := range merged {
		out[i] = sp.point
	}
	return out, nil
}

// topNAllParallel scores every point in parallel and returns all of them
// sorted by score descending. Used when the caller asked for n <= 0 or n >= total.
func topNAllParallel(query []float32, qNorm float32, points []QdrantPoint, progress func(processed, total int)) ([]QdrantPoint, error) {
	total := len(points)
	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}
	if workers > total {
		workers = total
	}
	chunkSize := (total + workers - 1) / workers

	scored := make([]scoredPoint, total)
	var wg sync.WaitGroup
	var processed atomic.Int64
	for w := 0; w < workers; w++ {
		start := w * chunkSize
		if start >= total {
			continue
		}
		end := start + chunkSize
		if end > total {
			end = total
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				scored[i] = scorePoint(query, qNorm, points[i], i)
			}
			done := int(processed.Add(int64(hi - lo)))
			if progress != nil && w == 0 {
				if progress != nil {
					progress(done, total)
				}
			}
		}(start, end)
	}
	wg.Wait()

	if progress != nil {
		progress(total, total)
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].score > scored[j].score })
	out := make([]QdrantPoint, len(scored))
	for i, sp := range scored {
		out[i] = sp.point
	}
	return out, nil
}

type scoredPoint struct {
	point QdrantPoint
	score float32
}

func scorePoint(query []float32, qNorm float32, p QdrantPoint, origIdx int) scoredPoint {
	if len(p.Vector) != len(query) {
		// Skip dimension-mismatched points (they cannot be scored).
		return scoredPoint{point: p, score: -1}
	}
	var dot, pNorm float32
	for i, v := range p.Vector {
		dot += v * query[i]
		pNorm += v * v
	}
	if pNorm == 0 {
		return scoredPoint{point: p, score: 0}
	}
	score := dot / (qNorm * float32(math.Sqrt(float64(pNorm))))
	// Preserve the original index in case callers want it.
	_ = origIdx
	return scoredPoint{point: p, score: score}
}

func heapifyMin(h []scoredPoint) {
	for i := len(h)/2 - 1; i >= 0; i-- {
		siftDownMin(h, i)
	}
}

func siftDownMin(h []scoredPoint, i int) {
	n := len(h)
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i
		if left < n && h[left].score < h[smallest].score {
			smallest = left
		}
		if right < n && h[right].score < h[smallest].score {
			smallest = right
		}
		if smallest == i {
			return
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
}

func extractTexts(points []QdrantPoint) []string {
	texts := make([]string, 0, len(points))
	for _, pt := range points {
		textStr := pt.ExtractText()
		if textStr != "" {
			texts = append(texts, textStr)
		}
	}
	return texts
}

// sqrt was previously a custom Newton's method implementation.
// Now uses math.Sqrt directly for SIMD-optimized hardware FPU instructions.

// GetCollectionInfo retrieves metadata stats for the active collection from Qdrant.
func GetCollectionInfo(ctx context.Context, baseURL, apiKey, collection string) (int, int, string, error) {
	url := fmt.Sprintf("%s/collections/%s", strings.TrimSuffix(baseURL, "/"), collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, 0, "", err
	}
	if apiKey != "" {
		req.Header.Set("api-key", apiKey)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, 0, "", fmt.Errorf("HTTP status: %s", resp.Status)
	}

	var info QdrantCollectionInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return 0, 0, "", err
	}

	return info.Result.PointsCount, info.Result.VectorsCount, info.Result.Status, nil
}
