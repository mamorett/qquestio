package rag

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
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

	res1, pts1, err := SearchQdrant(context.Background(), server1.URL, "secret", "my-col", []float32{0.1}, 5)
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

	res2, pts2, err := SearchQdrant(context.Background(), server2.URL, "secret", "my-col", []float32{0.1}, 5)
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
}
