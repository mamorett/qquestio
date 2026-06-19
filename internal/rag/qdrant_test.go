package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeQdrant is a minimal in-process Qdrant stand-in used to exercise the
// context-expansion code path without needing a real Qdrant server.
//
// It supports only the endpoints that SearchWithContextExpansion touches:
//   - POST /collections/{c}/points/search  (primary top-N, exact=true)
//   - POST /collections/{c}/points/scroll   (adjacent-chunk range lookups)
//
// The corpus is a 10-doc synthetic collection where the answer to a query
// is intentionally split across two adjacent chunks of one document.
type fakeQdrant struct {
	mu      sync.Mutex
	points  map[string]QdrantPoint // keyed by point ID (string form)
	dim     int
	searchN atomic.Int64 // number of /points/search calls
	scrollN atomic.Int64 // number of /points/scroll calls
	scrolls []string     // JSON bodies of each scroll request, in order
}

func newFakeQdrant(corpus []QdrantPoint) *fakeQdrant {
	f := &fakeQdrant{
		points:  make(map[string]QdrantPoint, len(corpus)),
		dim:     0,
		scrolls: nil,
	}
	for _, p := range corpus {
		if f.dim == 0 && len(p.Vector) > 0 {
			f.dim = len(p.Vector)
		}
		f.points[fmt.Sprintf("%v", p.ID)] = p
	}
	return f
}

func (f *fakeQdrant) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/collections/", func(w http.ResponseWriter, r *http.Request) {
		// Route by suffix.
		path := r.URL.Path
		if strings.HasSuffix(path, "/points/search") {
			f.serveSearch(w, r)
			return
		}
		if strings.HasSuffix(path, "/points/scroll") {
			f.serveScroll(w, r)
			return
		}
		http.NotFound(w, r)
	})
	return mux
}

func (f *fakeQdrant) serveSearch(w http.ResponseWriter, r *http.Request) {
	f.searchN.Add(1)
	var req QdrantSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	// Collect + score all points that match the (optional) filter.
	f.mu.Lock()
	matches := make([]QdrantPoint, 0, len(f.points))
	for _, p := range f.points {
		matches = append(matches, p)
	}
	f.mu.Unlock()

	// Apply filter (simplified — only file_name + chunk_index range).
	if req.Filter != nil {
		matches = applyServerFilter(matches, req.Filter)
	}

	// Compute cosine similarity to req.Vector.
	type scored struct {
		point QdrantPoint
		score float32
	}
	scoredAll := make([]scored, 0, len(matches))
	for _, p := range matches {
		scoredAll = append(scoredAll, scored{point: p, score: cosine(req.Vector, p.Vector)})
	}
	sort.SliceStable(scoredAll, func(i, j int) bool {
		return scoredAll[i].score > scoredAll[j].score
	})
	if req.Limit > 0 && len(scoredAll) > req.Limit {
		scoredAll = scoredAll[:req.Limit]
	}

	out := make([]QdrantPoint, 0, len(scoredAll))
	for _, s := range scoredAll {
		// Deep-copy payload so callers can't mutate our internal store.
		cp := QdrantPoint{
			ID:      s.point.ID,
			Score:   s.score,
			Payload: copyPayload(s.point.Payload),
		}
		out = append(out, cp)
	}

	resp := struct {
		Result []QdrantPoint `json:"result"`
		Status string        `json:"status"`
	}{Result: out, Status: "ok"}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (f *fakeQdrant) serveScroll(w http.ResponseWriter, r *http.Request) {
	f.scrollN.Add(1)
	body, _ := readAndRestore(r)
	f.mu.Lock()
	f.scrolls = append(f.scrolls, body)
	f.mu.Unlock()

	var req ScrollRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	// Apply server-side filter.
	matches := make([]QdrantPoint, 0, len(f.points))
	f.mu.Lock()
	for _, p := range f.points {
		matches = append(matches, p)
	}
	f.mu.Unlock()
	if req.Filter != nil {
		matches = applyServerFilter(matches, req.Filter)
	}

	// Trim to the requested limit.
	if req.Limit > 0 && len(matches) > req.Limit {
		matches = matches[:req.Limit]
	}

	resp := struct {
		Result struct {
			Points         []QdrantPoint `json:"points"`
			NextPageOffset interface{}   `json:"next_page_offset"`
		} `json:"result"`
		Status string `json:"status"`
	}{Status: "ok"}
	for _, p := range matches {
		resp.Result.Points = append(resp.Result.Points, QdrantPoint{
			ID:      p.ID,
			Payload: copyPayload(p.Payload),
			Score:   0,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func cosine(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrtF(na) * sqrtF(nb))
}

func sqrtF(x float32) float32 {
	// Avoid pulling math into tests unnecessarily; Newton step is plenty here.
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 8; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func copyPayload(p map[string]interface{}) map[string]interface{} {
	if p == nil {
		return nil
	}
	out := make(map[string]interface{}, len(p))
	for k, v := range p {
		out[k] = v
	}
	return out
}

func readAndRestore(r *http.Request) (string, error) {
	var sb strings.Builder
	buf := make([]byte, 1024)
	for {
		n, err := r.Body.Read(buf)
		if n > 0 {
			sb.Write(buf[:n])
		}
		if err != nil {
			break
		}
	}
	r.Body = stringReadCloser(sb.String())
	return sb.String(), nil
}

type stringRC struct {
	s   string
	pos int
}

func (s *stringRC) Read(p []byte) (int, error) {
	if s.pos >= len(s.s) {
		return 0, fmt.Errorf("EOF")
	}
	n := copy(p, s.s[s.pos:])
	s.pos += n
	return n, nil
}
func (s *stringRC) Close() error { return nil }

func stringReadCloser(s string) *stringRC { return &stringRC{s: s} }

// applyServerFilter is a simplified server-side filter for the fake Qdrant.
// It supports the two filter shapes that SearchWithContextExpansion actually
// emits: file_name match AND chunk_index range.
func applyServerFilter(points []QdrantPoint, f *QdrantFilter) []QdrantPoint {
	if f == nil {
		return points
	}
	docKey := ""
	docVal := ""
	lo, hi := -1<<31, 1<<31-1
	for _, c := range f.Must {
		if c.Match.Value != nil {
			if s, ok := c.Match.Value.(string); ok {
				docKey = c.Key
				docVal = s
			}
		}
		if c.Range != nil {
			if c.Range.Gte != nil {
				lo = int(*c.Range.Gte)
			}
			if c.Range.Lte != nil {
				hi = int(*c.Range.Lte)
			}
		}
	}
	out := make([]QdrantPoint, 0, len(points))
	for _, p := range points {
		if docKey != "" {
			if v, ok := p.Payload[docKey]; !ok {
				continue
			} else if s, ok := v.(string); !ok || s != docVal {
				continue
			}
		}
		if lo != -1<<31 || hi != 1<<31-1 {
			idx, ok := p.Payload["chunk_index"].(float64)
			if !ok {
				if i, ok2 := p.Payload["chunk_index"].(int); ok2 {
					idx = float64(i)
				} else {
					continue
				}
			}
			if int(idx) < lo || int(idx) > hi {
				continue
			}
		}
		out = append(out, p)
	}
	return out
}

// makeCorpus creates a 10-doc synthetic Qdrant collection where the answer
// to a query is split across chunks 3 and 4 of doc #5.
//
// Each document has 6 chunks, each chunk has 8-dim embeddings. We hand-pick
// the vectors so that the query is *strongly* similar to doc5/chunk3 AND
// doc5/chunk4 (because both contain key answer phrases) but only weakly
// similar to anything else.
func makeCorpus(t *testing.T) ([]QdrantPoint, []float32, []string) {
	t.Helper()
	const dim = 8
	const docs = 10
	const chunksPerDoc = 6

	corpus := make([]QdrantPoint, 0, docs*chunksPerDoc)
	docIDs := make([]string, 0, docs)

	idCounter := 1
	for d := 1; d <= docs; d++ {
		docID := fmt.Sprintf("doc%d.txt", d)
		docIDs = append(docIDs, docID)
		for c := 0; c < chunksPerDoc; c++ {
			// Default vector: ortho-ish, low cosine with the query.
			v := make([]float32, dim)
			for i := range v {
				v[i] = float32((d+c+i)%7) / 7.0 // noise
			}
			// Doc5, chunk3 and chunk4 are the "answer" chunks.
			// Make them strongly similar to a query vector we'll construct
			// to align with both (positive dot on the first 3 dims).
			if d == 5 && (c == 3 || c == 4) {
				v = []float32{1, 0.9, 0.8, 0.1, 0.05, 0.05, 0.05, 0.05}
			}
			text := fmt.Sprintf("doc=%s chunk=%d filler %d %d %d", docID, c, d, c, d*c)
			if d == 5 && c == 3 {
				text = "ANSWER-HALF-A " + text
			}
			if d == 5 && c == 4 {
				text = "ANSWER-HALF-B " + text
			}
			corpus = append(corpus, QdrantPoint{
				ID: uint64(idCounter),
				Payload: map[string]interface{}{
					"file_name":   docID,
					"chunk_index": float64(c),
					"text":        text,
				},
				Vector: v,
				Score:  0,
			})
			idCounter++
		}
	}
	query := []float32{1, 0.9, 0.8, 0.1, 0.05, 0.05, 0.05, 0.05}
	return corpus, query, docIDs
}

// TestSearchWithContextExpansion verifies the end-to-end pipeline:
//  1. Top-N exact search returns chunk3 (or chunk4) of doc5.
//  2. The scroll expansion pulls BOTH chunk3 AND chunk4 of doc5 (because
//     expand=1 means ±1 neighbor, so 3 ± 1 = {2,3,4}).
//  3. The returned context string contains the answer halves from BOTH chunks.
func TestSearchWithContextExpansion(t *testing.T) {
	corpus, query, _ := makeCorpus(t)

	srv := httptest.NewServer(newFakeQdrant(corpus).handler())
	defer srv.Close()

	dim := len(query)
	_ = dim

	// We need to use the package-private fakes. Easiest path: drive
	// SearchWithContextExpansion against our httptest server.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	const limit = 1
	const expand = 1
	res, err := SearchWithContextExpansionDetailed(
		ctx,
		srv.URL, "",
		"test_collection",
		query,
		limit, expand,
		"", "",
	)
	if err != nil {
		t.Fatalf("SearchWithContextExpansionDetailed returned error: %v", err)
	}
	contextStr := res.Context
	points := res.ExpandedPoints

	if len(points) == 0 {
		t.Fatalf("expected non-empty points, got 0")
	}

	// The context string should contain BOTH answer halves (chunk3 AND chunk4).
	// This is the core property we want to verify: even if only one of the two
	// answer chunks wins the top-1 similarity tie, the expansion pulls its
	// neighbor so the LLM sees the complete answer span.
	if !strings.Contains(contextStr, "ANSWER-HALF-A") {
		t.Errorf("context is missing ANSWER-HALF-A (chunk3 of doc5); got:\n%s", contextStr)
	}
	if !strings.Contains(contextStr, "ANSWER-HALF-B") {
		t.Errorf("context is missing ANSWER-HALF-B (chunk4 of doc5); got:\n%s", contextStr)
	}

	// We should have pulled at least 3 chunks from doc5 (top match + 2 neighbors).
	// The exact set depends on which answer chunk wins the top-1 tie; either way
	// the expansion is {top-1, top, top+1} so we see exactly 3 consecutive
	// chunks centered on the top match.
	doc5Chunks := 0
	for i := 0; i < 6; i++ {
		if strings.Contains(contextStr, fmt.Sprintf("chunk=%d filler 5", i)) {
			doc5Chunks++
		}
	}
	if doc5Chunks < 3 {
		t.Errorf("expected at least 3 consecutive doc5 chunks from expansion, got %d; context:\n%s", doc5Chunks, contextStr)
	}
}

// TestSearchWithContextExpansionExpandOff verifies the legacy path: with
// expand=0, the function returns only the primary top-N match (no adjacent
// chunks are pulled). This is the legacy /expand off behavior.
func TestSearchWithContextExpansionExpandOff(t *testing.T) {
	corpus, query, _ := makeCorpus(t)
	srv := httptest.NewServer(newFakeQdrant(corpus).handler())
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	const limit = 1
	const expand = 0
	res, err := SearchWithContextExpansionDetailed(
		ctx,
		srv.URL, "",
		"test_collection",
		query,
		limit, expand,
		"", "",
	)
	if err != nil {
		t.Fatalf("SearchWithContextExpansionDetailed returned error: %v", err)
	}
	contextStr := res.Context

	// With expand=0, we should have ONLY the primary match.
	// It should contain one of the answer halves but not both.
	hasA := strings.Contains(contextStr, "ANSWER-HALF-A")
	hasB := strings.Contains(contextStr, "ANSWER-HALF-B")
	if hasA && hasB {
		t.Errorf("expand=0 should not pull BOTH answer halves; got:\n%s", contextStr)
	}
	if !hasA && !hasB {
		t.Errorf("expand=0 should still return the primary match (one answer half); got:\n%s", contextStr)
	}
}

// TestSearchWithContextExpansionEmptyCorpus ensures the function behaves
// gracefully when the collection is empty (no scroll requests, no panics).
func TestSearchWithContextExpansionEmptyCorpus(t *testing.T) {
	srv := httptest.NewServer(newFakeQdrant(nil).handler())
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := []float32{0.1, 0.2, 0.3, 0.4}
	res, err := SearchWithContextExpansionDetailed(
		ctx,
		srv.URL, "",
		"empty",
		query,
		5, 1,
		"", "",
	)
	if err != nil {
		t.Fatalf("SearchWithContextExpansionDetailed on empty corpus returned error: %v", err)
	}
	points := res.ExpandedPoints
	if len(points) != 0 {
		t.Errorf("expected 0 points for empty corpus, got %d", len(points))
	}
}

// TestQdrantRangeSerialization is a unit test for the QdrantFilter → JSON
// encoding that the scroll-adjacent-chunks path relies on. The Qdrant API
// requires the range filter to be {"range": {"gte": ..., "lte": ...}} not
// {"match": {"value": {"gte": ..., "lte": ...}}}.
func TestQdrantRangeSerialization(t *testing.T) {
	lo := 3.0
	hi := 7.0
	f := &QdrantFilter{
		Must: []QdrantFieldCondition{
			{Key: "file_name", Match: QdrantMatch{Value: "doc5.txt"}},
			{Key: "chunk_index", Range: &QdrantRange{Gte: &lo, Lte: &hi}},
		},
	}
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(data)
	if !strings.Contains(s, `"range":{"gte":3,"lte":7}`) {
		t.Errorf("expected serialized JSON to contain range clause; got: %s", s)
	}
	if strings.Contains(s, `"match":{"value":{`) {
		t.Errorf("serialized JSON should NOT use match.value for range; got: %s", s)
	}
	if !strings.Contains(s, `"file_name"`) {
		t.Errorf("expected serialized JSON to contain file_name condition; got: %s", s)
	}
}

// TestSearchWithContextExpansionDetailed is a regression test for the
// "expand does nothing in the rerank path" bug. The detailed function
// must:
//  1. Return the primary top-N separately from the expanded set.
//  2. Populate ExpansionMap with EVERY chunk we fetched (primary + adjacent).
//  3. Allow ApplyExpansionToPrimaries to re-apply ±expand to a DIFFERENT
//     set of primaries (e.g. the post-rerank top-K) using only the cached
//     map, with no further network calls.
func TestSearchWithContextExpansionDetailed(t *testing.T) {
	corpus, query, _ := makeCorpus(t)
	srv := httptest.NewServer(newFakeQdrant(corpus).handler())
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := SearchWithContextExpansionDetailed(
		ctx, srv.URL, "", "test", query,
		1, // limit: 1 primary top match
		1, // expand: ±1
		"", "",
	)
	if err != nil {
		t.Fatalf("SearchWithContextExpansionDetailed error: %v", err)
	}

	// (1) primaries must contain exactly 1 point (the top match).
	if len(res.PrimaryPoints) != 1 {
		t.Errorf("expected exactly 1 primary point, got %d", len(res.PrimaryPoints))
	}

	// (2) the expansion map must include all chunks of doc5 (chunks 0-5)
	// because expand=1 around chunk 3 or 4 covers [0, 5].
	if _, ok := res.ExpansionMap["doc5.txt"]; !ok {
		t.Fatalf("expected ExpansionMap to contain doc5.txt; got: %v", mapKeys(res.ExpansionMap))
	}
	doc5Chunks := res.ExpansionMap["doc5.txt"]
	if len(doc5Chunks) < 3 {
		t.Errorf("expected ExpansionMap[doc5.txt] to have ≥3 chunks (top ± 1), got %d", len(doc5Chunks))
	}

	// (3) ApplyExpansionToPrimaries: simulate a rerank that DROPS chunk 5
	// from the primaries and keeps chunk 3 instead. Then re-apply expand=1
	// and verify we get the right window back.
	differentPrimaries := []QdrantPoint{
		{
			ID: uint64(1),
			Payload: map[string]interface{}{
				"file_name":   "doc5.txt",
				"chunk_index": float64(3),
			},
			Score: 0.99,
		},
	}
	gotCtx, gotPoints := ApplyExpansionToPrimaries(differentPrimaries, res.ExpansionMap, 1)
	if !strings.Contains(gotCtx, "ANSWER-HALF-A") {
		t.Errorf("re-applied expansion should contain ANSWER-HALF-A; got:\n%s", gotCtx)
	}
	if !strings.Contains(gotCtx, "ANSWER-HALF-B") {
		t.Errorf("re-applied expansion should contain ANSWER-HALF-B (the ±1 neighbor); got:\n%s", gotCtx)
	}
	if len(gotPoints) < 2 {
		t.Errorf("expected at least 2 expanded points for the new primaries, got %d", len(gotPoints))
	}
}

// TestApplyExpansionToPrimariesNoMap verifies the helper handles the
// empty-map case gracefully (e.g. when the HNSW/local paths bypass the
// expansion map and we still want rerank to "work").
func TestApplyExpansionToPrimariesNoMap(t *testing.T) {
	primaries := []QdrantPoint{
		{ID: 1, Payload: map[string]interface{}{"text": "hello"}, Score: 0.9},
		{ID: 2, Payload: map[string]interface{}{"text": "world"}, Score: 0.8},
	}
	gotCtx, gotPoints := ApplyExpansionToPrimaries(primaries, ExpansionMap{}, 5)
	if !strings.Contains(gotCtx, "hello") || !strings.Contains(gotCtx, "world") {
		t.Errorf("empty map fallback should still emit primaries; got:\n%s", gotCtx)
	}
	if len(gotPoints) != 2 {
		t.Errorf("expected 2 expanded points, got %d", len(gotPoints))
	}
}

// mapKeys returns the keys of an ExpansionMap for diagnostic messages.
func mapKeys(m ExpansionMap) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
