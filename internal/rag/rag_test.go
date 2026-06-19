package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGetEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("expected path /embedding, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-embedding-key" {
			t.Errorf("expected Authorization Bearer test-embedding-key, got %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"data": [{"embedding": [0.1, 0.2, 0.3]}]}`)
	}))
	defer server.Close()

	vec, err := GetEmbedding(context.Background(), server.URL, "test-embedding-key", "test-model", "test query")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(vec) != 3 || vec[0] != 0.1 || vec[1] != 0.2 || vec[2] != 0.3 {
		t.Errorf("unexpected embedding vector: %v", vec)
	}
}

func TestSearchQdrant(t *testing.T) {
	// Test Format 1: Flat array of points
	server1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST request, got %s", r.Method)
		}
		if r.Header.Get("api-key") != "secret" {
			t.Errorf("expected api-key secret, got %s", r.Header.Get("api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"result": [{"id": 1, "payload": {"text": "hello flat qdrant"}}]}`)
	}))
	defer server1.Close()

	res1, pts1, err := SearchQdrant(context.Background(), server1.URL, "secret", "my-col", []float32{0.1}, 1000, 5, "", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res1 != "hello flat qdrant" {
		t.Errorf("expected 'hello flat qdrant', got '%s'", res1)
	}
	if len(pts1) != 1 || pts1[0].ID != float64(1) {
		t.Errorf("unexpected points: %v", pts1)
	}

	// Test Format 2: Object wrapping points array
	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"result": {"points": [{"id": 1, "payload": {"text": "hello wrapped qdrant"}}]}}`)
	}))
	defer server2.Close()

	res2, pts2, err := SearchQdrant(context.Background(), server2.URL, "secret", "my-col", []float32{0.1}, 1000, 5, "", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res2 != "hello wrapped qdrant" {
		t.Errorf("expected 'hello wrapped qdrant', got '%s'", res2)
	}
	if len(pts2) != 1 || pts2[0].ID != float64(1) {
		t.Errorf("unexpected points: %v", pts2)
	}
}

func TestLiteLLMStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-openai-key" {
			t.Errorf("expected Authorization Bearer test-openai-key, got %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, "data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	reader, err := StartLiteLLMStream(context.Background(), server.URL, "test-openai-key", "llm", []ChatMessage{})
	if err != nil {
		t.Fatalf("unexpected error starting stream: %v", err)
	}
	defer reader.Close()

	chunk1, done1, err := reader.Next()
	if err != nil || done1 || chunk1 != "Hello" {
		t.Errorf("chunk1 failed: got chunk=%s, done=%v, err=%v", chunk1, done1, err)
	}

	chunk2, done2, err := reader.Next()
	if err != nil || done2 || chunk2 != " world" {
		t.Errorf("chunk2 failed: got chunk=%s, done=%v, err=%v", chunk2, done2, err)
	}

	chunk3, done3, err := reader.Next()
	if err != nil || !done3 || chunk3 != "" {
		t.Errorf("chunk3 failed: got chunk=%s, done=%v, err=%v", chunk3, done3, err)
	}
}

func TestRerank(t *testing.T) {
	// Test Format 1: Raw array of floats
	server1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `[0.98, 0.85]`)
	}))
	defer server1.Close()

	items1, err := Rerank(context.Background(), server1.URL, "apiKey", "model", "query", []string{"doc1", "doc2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items1) != 2 || items1[0].Index != 0 || items1[0].Score != 0.98 || items1[1].Index != 1 || items1[1].Score != 0.85 {
		t.Errorf("unexpected results for floats: %v", items1)
	}

	// Test Format 2: Array of objects
	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `[{"index": 0, "score": 0.98}, {"index": 1, "relevance_score": 0.85}]`)
	}))
	defer server2.Close()

	items2, err := Rerank(context.Background(), server2.URL, "apiKey", "model", "query", []string{"doc1", "doc2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items2) != 2 || items2[0].Index != 0 || items2[0].Score != 0.98 || items2[1].Index != 1 || items2[1].Score != 0.85 {
		t.Errorf("unexpected results for objects array: %v", items2)
	}

	// Test Format 3: Envelope object
	server3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"results": [{"index": 0, "score": 0.98}, {"index": 1, "relevance_score": 0.85}]}`)
	}))
	defer server3.Close()

	items3, err := Rerank(context.Background(), server3.URL, "apiKey", "model", "query", []string{"doc1", "doc2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items3) != 2 || items3[0].Index != 0 || items3[0].Score != 0.98 || items3[1].Index != 1 || items3[1].Score != 0.85 {
		t.Errorf("unexpected results for envelope object: %v", items3)
	}

	// Test Format 4: String/float index
	server4 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"results": [{"index": "0.0", "score": 0.95}, {"index": 1.0, "score": 0.80}]}`)
	}))
	defer server4.Close()

	items4, err := Rerank(context.Background(), server4.URL, "apiKey", "model", "query", []string{"doc1", "doc2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items4) != 2 || items4[0].Index != 0 || items4[0].Score != 0.95 || items4[1].Index != 1 || items4[1].Score != 0.80 {
		t.Errorf("unexpected results for float/string indices: %v", items4)
	}

	// Test Format 5: Missing index but document text is returned (text fallback)
	server5 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"results": [{"document": {"text": "doc2"}, "score": 0.90}, {"document": "doc1", "score": 0.70}]}`)
	}))
	defer server5.Close()

	items5, err := Rerank(context.Background(), server5.URL, "apiKey", "model", "query", []string{"doc1", "doc2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Note: since document "doc2" was returned first, it should match input doc2 (index 1)
	// and document "doc1" was returned second, it should match input doc1 (index 0)
	if len(items5) != 2 || items5[0].Index != 1 || items5[0].Score != 0.90 || items5[1].Index != 0 || items5[1].Score != 0.70 {
		t.Errorf("unexpected results for document text fallback matching: %v", items5)
	}
}

func TestQdrantPoint_ExtractText(t *testing.T) {
	tests := []struct {
		name     string
		payload  map[string]interface{}
		expected string
	}{
		{
			name: "Single primary key",
			payload: map[string]interface{}{
				"text": "Hello world from primary key",
			},
			expected: "Hello world from primary key",
		},
		{
			name: "Multiple primary keys",
			payload: map[string]interface{}{
				"text":    "First primary key content",
				"content": "Second primary key content",
			},
			expected: "First primary key content\n\nSecond primary key content",
		},
		{
			name: "Primary key and metadata keys",
			payload: map[string]interface{}{
				"text":      "Important content",
				"file_name": "ignore_me.txt",
				"score":     0.98,
				"id":        "123",
			},
			expected: "Important content",
		},
		{
			name: "Primary key and other non-metadata keys",
			payload: map[string]interface{}{
				"text":            "Primary content",
				"secondary_story": "Additional story details",
				"filename":        "source.txt",
			},
			expected: "Primary content\n\nAdditional story details",
		},
		{
			name: "No primary keys but other non-metadata keys",
			payload: map[string]interface{}{
				"unrecognized_field": "This is fallback content",
				"another_field":      "More fallback content",
				"source":             "doc.pdf",
			},
			expected: "This is fallback content\n\nMore fallback content", // Order is map iteration based, but checking existence of both is key
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pt := QdrantPoint{Payload: tc.payload}
			got := pt.ExtractText()
			if tc.name == "No primary keys but other non-metadata keys" {
				// Map iteration order is non-deterministic in Go, so check elements
				if !strings.Contains(got, "This is fallback content") || !strings.Contains(got, "More fallback content") {
					t.Errorf("expected to contain both fallback contents, got: %q", got)
				}
			} else {
				if got != tc.expected {
					t.Errorf("expected: %q, got: %q", tc.expected, got)
				}
			}
		})
	}
}

func TestSearchQdrant_Filter(t *testing.T) {
	// 1. Test document-identifying key (file_name) -> should produce Should conditions
	serverDoc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req QdrantQueryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if req.Filter == nil {
			t.Error("expected filter to be present, got nil")
		} else {
			if len(req.Filter.Should) == 0 {
				t.Error("expected should conditions to be present, got 0")
			} else {
				foundFilePathMatch := false
				for _, cond := range req.Filter.Should {
					if cond.Key == "file_path" && cond.Match.Value == "guide.txt" {
						foundFilePathMatch = true
						break
					}
				}
				if !foundFilePathMatch {
					t.Error("expected file_path condition in should array, but not found")
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"result": [{"id": 1, "payload": {"text": "hello flat filtered qdrant"}}]}`)
	}))
	defer serverDoc.Close()

	_, _, err := SearchQdrant(context.Background(), serverDoc.URL, "secret", "my-col", []float32{0.1}, 1000, 5, "file_name", "guide.txt")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 2. Test non-document key (chunk_index) -> should produce Must conditions
	serverNonDoc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req QdrantQueryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if req.Filter == nil {
			t.Error("expected filter to be present, got nil")
		} else {
			if len(req.Filter.Must) != 1 {
				t.Errorf("expected 1 must condition, got %d", len(req.Filter.Must))
			} else {
				cond := req.Filter.Must[0]
				if cond.Key != "chunk_index" || cond.Match.Value != "5" {
					t.Errorf("unexpected filter condition: key=%s, value=%v", cond.Key, cond.Match.Value)
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"result": [{"id": 1, "payload": {"text": "hello flat filtered qdrant"}}]}`)
	}))
	defer serverNonDoc.Close()

	_, _, err = SearchQdrant(context.Background(), serverNonDoc.URL, "secret", "my-col", []float32{0.1}, 1000, 5, "chunk_index", "5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCorpusCache_SaveLoad(t *testing.T) {
	// Override the cache dir for this test.
	tmpDir := t.TempDir()
	t.Setenv("QQUESTIO_CACHE_DIR", tmpDir)
	collection := "test_collection-save.load"

	// Build a small fake corpus.
	dim := 4
	points := []QdrantPoint{
		{ID: 1, Payload: map[string]interface{}{"text": "hello", "file_name": "a.txt"}, Vector: []float32{0.1, 0.2, 0.3, 0.4}, Score: 0.99},
		{ID: "uuid-2", Payload: map[string]interface{}{"text": "world", "file_name": "b.txt"}, Vector: []float32{-0.1, -0.2, -0.3, -0.4}, Score: 0.5},
		{ID: float64(3), Payload: map[string]interface{}{"text": "goodbye", "file_name": "c.txt"}, Vector: []float32{0.5, 0.5, 0.5, 0.5}, Score: 0.1},
	}

	if err := SaveCorpusCache(collection, dim, points); err != nil {
		t.Fatalf("SaveCorpusCache failed: %v", err)
	}

	cache, loaded, err := LoadCorpusCache(collection)
	if err != nil {
		t.Fatalf("LoadCorpusCache failed: %v", err)
	}
	if cache == nil {
		t.Fatal("expected cache, got nil")
	}
	if cache.Collection != collection {
		t.Errorf("expected collection=%q, got %q", collection, cache.Collection)
	}
	if cache.PointCount != len(points) {
		t.Errorf("expected point count %d, got %d", len(points), cache.PointCount)
	}
	if cache.Dimension != dim {
		t.Errorf("expected dim %d, got %d", dim, cache.Dimension)
	}
	if len(loaded) != len(points) {
		t.Fatalf("expected %d loaded points, got %d", len(points), len(loaded))
	}

	for i, orig := range points {
		got := loaded[i]
		// Score is not preserved (recomputed on load); check ID + payload + vector.
		if fmt.Sprintf("%v", got.ID) != fmt.Sprintf("%v", orig.ID) {
			t.Errorf("point %d: id mismatch: got %v, want %v", i, got.ID, orig.ID)
		}
		if got.Payload["text"] != orig.Payload["text"] {
			t.Errorf("point %d: text mismatch: got %v, want %v", i, got.Payload["text"], orig.Payload["text"])
		}
		if got.Payload["file_name"] != orig.Payload["file_name"] {
			t.Errorf("point %d: file_name mismatch", i)
		}
		if len(got.Vector) != dim {
			t.Errorf("point %d: vector dim %d, want %d", i, len(got.Vector), dim)
		}
		for j, v := range got.Vector {
			if v != orig.Vector[j] {
				t.Errorf("point %d vector[%d]: got %v, want %v", i, j, v, orig.Vector[j])
				break
			}
		}
	}
}

func TestCorpusCache_LoadMissing(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("QQUESTIO_CACHE_DIR", tmpDir)

	cache, points, err := LoadCorpusCache("nonexistent_collection_xyz")
	if err != nil {
		t.Errorf("expected nil error for missing cache, got %v", err)
	}
	if cache != nil {
		t.Errorf("expected nil cache, got %+v", cache)
	}
	if points != nil {
		t.Errorf("expected nil points, got %v", points)
	}
}

func TestCorpusCache_Delete(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("QQUESTIO_CACHE_DIR", tmpDir)
	collection := "test_collection-delete"

	if err := SaveCorpusCache(collection, 2, []QdrantPoint{
{ID: 1, Payload: map[string]interface{}{"x": "y"}, Vector: []float32{1, 0}},
}); err != nil {
		t.Fatalf("SaveCorpusCache failed: %v", err)
	}

	// Delete should succeed.
	if err := DeleteCorpusCache(collection); err != nil {
		t.Fatalf("DeleteCorpusCache failed: %v", err)
	}

	// Subsequent load should return nil.
	cache, _, err := LoadCorpusCache(collection)
	if err != nil {
		t.Fatalf("LoadCorpusCache after delete failed: %v", err)
	}
	if cache != nil {
		t.Errorf("expected nil cache after delete, got %+v", cache)
	}

	// Deleting again should be a no-op (no error).
	if err := DeleteCorpusCache(collection); err != nil {
		t.Errorf("second DeleteCorpusCache should not error, got %v", err)
	}
}

func TestCorpusCache_SafeName(t *testing.T) {
	// Collection names with unsafe characters should normalize to safe filenames.
	// This is tested indirectly: SaveCorpusCache with special chars should not crash.
	tmpDir := t.TempDir()
	t.Setenv("QQUESTIO_CACHE_DIR", tmpDir)

	collection := "my/unsafe name with spaces & symbols!@#"
	if err := SaveCorpusCache(collection, 2, []QdrantPoint{
{ID: 1, Payload: map[string]interface{}{"x": "y"}, Vector: []float32{1, 0}},
}); err != nil {
		t.Fatalf("SaveCorpusCache with unsafe name failed: %v", err)
	}

	cache, _, err := LoadCorpusCache(collection)
	if err != nil {
		t.Fatalf("LoadCorpusCache with unsafe name failed: %v", err)
	}
	if cache == nil || cache.Collection != collection {
		t.Errorf("expected collection=%q, got %+v", collection, cache)
	}
}

