package rag

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"
)

func TestExtractQuotedPhrases(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{
			input:    `"needle in haystack"`,
			expected: []string{"needle in haystack"},
		},
		{
			input:    `find "needle in haystack" please`,
			expected: []string{"needle in haystack"},
		},
		{
			input:    `"foo" "bar"`,
			expected: []string{"foo", "bar"},
		},
		{
			input:    `'foo' “bar”`,
			expected: []string{"foo", "bar"},
		},
		{
			input:    `„foo“ and «bar»`,
			expected: []string{"foo", "bar"},
		},
		{
			input:    `„foo” and «bar»`,
			expected: []string{"foo", "bar"},
		},
		{
			input:    `no quotes here`,
			expected: nil,
		},
	}

	for _, tc := range tests {
		got := ExtractQuotedPhrases(tc.input)
		if len(got) != len(tc.expected) {
			t.Errorf("ExtractQuotedPhrases(%q): got %v, expected %v", tc.input, got, tc.expected)
			continue
		}
		for i := range got {
			if got[i] != tc.expected[i] {
				t.Errorf("ExtractQuotedPhrases(%q)[%d]: got %q, expected %q", tc.input, i, got[i], tc.expected[i])
			}
		}
	}
}

func TestSearchQdrantExactPhrases(t *testing.T) {
	corpus := []QdrantPoint{
		{
			ID:      1,
			Vector:  []float32{1.0, 0.0},
			Payload: map[string]interface{}{"text": "Needle in haystack"},
		},
		{
			ID:      2,
			Vector:  []float32{0.0, 1.0},
			Payload: map[string]interface{}{"text": "needle in haystack"},
		},
		{
			ID:      3,
			Vector:  []float32{0.5, 0.5},
			Payload: map[string]interface{}{"text": "some other content"},
		},
		{
			ID:      4,
			Vector:  []float32{0.0, 0.0},
			Payload: map[string]interface{}{"text": "both foo and bar here"},
		},
		{
			ID:      5,
			Vector:  []float32{0.0, 0.0},
			Payload: map[string]interface{}{"text": "only foo here"},
		},
	}

	fake := newFakeQdrant(corpus)
	srv := httptest.NewServer(fake.handler())
	defer srv.Close()

	ctx := context.Background()

	t.Run("Case-insensitive exact match uppercase", func(t *testing.T) {
		fake.mu.Lock()
		fake.scrolls = nil
		fake.mu.Unlock()

		_, pts, err := SearchQdrantExactPhrases(
			ctx, srv.URL, "secret", "col",
			[]string{"Needle"},
			10, "", "",
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Case-insensitive: both "Needle" and "needle" should match
		if len(pts) != 2 {
			t.Errorf("expected 2 points (case-insensitive), got %d: %+v", len(pts), pts)
		}

		// Verify scroll request sent WithVector: false
		fake.mu.Lock()
		numScrolls := len(fake.scrolls)
		var body string
		if numScrolls > 0 {
			body = fake.scrolls[0]
		}
		fake.mu.Unlock()

		if numScrolls == 0 {
			t.Fatalf("expected at least one scroll request")
		}

		var req struct {
			WithVector bool `json:"with_vector"`
		}
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatalf("failed to decode scroll req body: %v", err)
		}
		if req.WithVector {
			t.Errorf("expected with_vector to be false, got true")
		}
	})

	t.Run("Case-insensitive exact match lowercase", func(t *testing.T) {
		_, pts, err := SearchQdrantExactPhrases(
			ctx, srv.URL, "secret", "col",
			[]string{"needle"},
			10, "", "",
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Case-insensitive: both "Needle" and "needle" should match
		if len(pts) != 2 {
			t.Errorf("expected 2 points (case-insensitive), got %d: %+v", len(pts), pts)
		}
	})

	t.Run("Multiple quoted phrases require all", func(t *testing.T) {
		_, pts, err := SearchQdrantExactPhrases(
			ctx, srv.URL, "secret", "col",
			[]string{"foo", "bar"},
			10, "", "",
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(pts) != 1 || pts[0].ID != float64(4) {
			t.Errorf("expected only point 4, got points: %+v", pts)
		}
	})
}
