package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

type RerankItem struct {
	Index int
	Score float64
}

// RerankRequest is a generic request structure for a reranker endpoint.
type RerankRequest struct {
	Model     string   `json:"model,omitempty"`
	Query     string   `json:"query"`
	Documents []string `json:"documents,omitempty"`
	Texts     []string `json:"texts,omitempty"`
}

// GenericRerankItem represents a returned item with index and score.
type GenericRerankItem struct {
	Index          interface{} `json:"index"`
	Score          *float64    `json:"score"`
	RelevanceScore *float64    `json:"relevance_score"`
	Document       interface{} `json:"document"`
}

// Rerank queries a generic, model-agnostic reranker endpoint and returns relevance scores.
func Rerank(ctx context.Context, baseURL, apiKey, model, query string, texts []string) ([]RerankItem, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	url := baseURL
	if !strings.Contains(url, "/rerank") {
		if strings.HasSuffix(strings.TrimSuffix(url, "/"), "/v1") {
			url = strings.TrimSuffix(url, "/") + "/rerank"
		} else {
			url = strings.TrimSuffix(url, "/") + "/v1/rerank"
		}
	}

	// 1. Try with documents first (standard for Cohere, Jina, llama.cpp, LiteLLM)
	reqBody := RerankRequest{
		Model:     model,
		Query:     query,
		Documents: texts,
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal rerank request: %w", err)
	}

	log.Printf("[Rerank] Requesting rerank (using documents field) from URL: %s, model: %s (query len: %d, documents: %d)", url, model, len(query), len(texts))
	if len(texts) > 0 {
		preview := texts[0]
		if len(preview) > 60 {
			preview = preview[:60] + "..."
		}
		log.Printf("[Rerank]   doc[0] preview: %q", preview)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create rerank request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("api-key", apiKey)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank request failed: %w", err)
	}

	// If the server rejected it due to schema validation (status 400 or 422),
	// retry using the 'texts' field (standard for HuggingFace TEI).
	if resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusUnprocessableEntity {
		resp.Body.Close()
		log.Printf("[Rerank] Server returned status %d. Retrying with texts field (TEI compatibility)...", resp.StatusCode)

		reqBodyTexts := RerankRequest{
			Model: model,
			Query: query,
			Texts: texts,
		}
		jsonBytes, err = json.Marshal(reqBodyTexts)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal retry rerank request: %w", err)
		}

		req, err = http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBytes))
		if err != nil {
			return nil, fmt.Errorf("failed to create retry rerank request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+apiKey)
			req.Header.Set("api-key", apiKey)
		}

		resp, err = client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("retry rerank request failed: %w", err)
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("rerank failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read rerank response: %w", err)
	}

	trimmed := strings.TrimSpace(string(bodyBytes))
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("empty response from rerank server")
	}

	log.Printf("[Rerank] Raw response: %s", trimmed)

	var items []RerankItem

	// 1. Try parsing as a raw array of floats/scores: [0.95, 0.82, ...]
	var floatScores []float64
	if err := json.Unmarshal(bodyBytes, &floatScores); err == nil {
		for i, score := range floatScores {
			items = append(items, RerankItem{Index: i, Score: score})
		}
		log.Printf("[Rerank] Parsed as float array, returned %d items", len(items))
		return items, nil
	}

	// 2. Try parsing as a raw array of objects: [{"index": 0, "score": 0.95}, ...]
	var objectScores []GenericRerankItem
	if err := json.Unmarshal(bodyBytes, &objectScores); err == nil {
		for i, item := range objectScores {
			idx := -1
			if val, ok := parseIndex(item.Index); ok {
				idx = val
			} else {
				docText := extractRerankText(item.Document)
				if docText != "" {
					for ti, txt := range texts {
						if strings.TrimSpace(txt) == strings.TrimSpace(docText) {
							idx = ti
							break
						}
					}
				}
			}
			if idx < 0 {
				idx = i
			}

			score := 0.0
			if item.Score != nil {
				score = *item.Score
			} else if item.RelevanceScore != nil {
				score = *item.RelevanceScore
			}
			items = append(items, RerankItem{Index: idx, Score: score})
		}
		log.Printf("[Rerank] Parsed as array of objects, returned %d items", len(items))
		for k, it := range items {
			if k < 5 {
				log.Printf("[Rerank]   item[%d]: Index=%d, Score=%f", k, it.Index, it.Score)
			}
		}
		return items, nil
	}

	// 3. Try parsing as a nested object: {"results": [{"index": 0, "score": 0.95}, ...]}
	var envelope struct {
		Results []GenericRerankItem `json:"results"`
	}
	if err := json.Unmarshal(bodyBytes, &envelope); err == nil && len(envelope.Results) > 0 {
		for i, item := range envelope.Results {
			idx := -1
			if val, ok := parseIndex(item.Index); ok {
				idx = val
			} else {
				docText := extractRerankText(item.Document)
				if docText != "" {
					for ti, txt := range texts {
						if strings.TrimSpace(txt) == strings.TrimSpace(docText) {
							idx = ti
							break
						}
					}
				}
			}
			if idx < 0 {
				idx = i
			}

			score := 0.0
			if item.Score != nil {
				score = *item.Score
			} else if item.RelevanceScore != nil {
				score = *item.RelevanceScore
			}
			items = append(items, RerankItem{Index: idx, Score: score})
		}
		log.Printf("[Rerank] Parsed as envelope results, returned %d items", len(items))
		for k, it := range items {
			if k < 5 {
				log.Printf("[Rerank]   item[%d]: Index=%d, Score=%f", k, it.Index, it.Score)
			}
		}
		return items, nil
	}

	return nil, fmt.Errorf("could not parse rerank response format: %s", trimmed)
}

func parseIndex(val interface{}) (int, bool) {
	if val == nil {
		return 0, false
	}
	switch n := val.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	case string:
		var i int
		if _, err := fmt.Sscanf(n, "%d", &i); err == nil {
			return i, true
		}
	}
	return 0, false
}

func extractRerankText(doc interface{}) string {
	if doc == nil {
		return ""
	}
	if s, ok := doc.(string); ok {
		return s
	}
	if m, ok := doc.(map[string]interface{}); ok {
		for _, key := range []string{"text", "content", "document"} {
			if v, ok := m[key]; ok {
				if s, ok := v.(string); ok {
					return s
				}
			}
		}
	}
	return ""
}
