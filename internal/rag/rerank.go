package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	Index          *int     `json:"index"`
	Score          *float64 `json:"score"`
	RelevanceScore *float64 `json:"relevance_score"`
}

// Rerank queries a generic, model-agnostic reranker endpoint and returns relevance scores.
func Rerank(ctx context.Context, url, apiKey, model, query string, texts []string) ([]RerankItem, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Build a generic request body containing both texts and documents for maximum compatibility
	reqBody := RerankRequest{
		Model:     model,
		Query:     query,
		Documents: texts,
		Texts:     texts,
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal rerank request: %w", err)
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

	var items []RerankItem

	// 1. Try parsing as a raw array of floats/scores: [0.95, 0.82, ...]
	var floatScores []float64
	if err := json.Unmarshal(bodyBytes, &floatScores); err == nil {
		for i, score := range floatScores {
			items = append(items, RerankItem{Index: i, Score: score})
		}
		return items, nil
	}

	// 2. Try parsing as a raw array of objects: [{"index": 0, "score": 0.95}, ...]
	var objectScores []GenericRerankItem
	if err := json.Unmarshal(bodyBytes, &objectScores); err == nil {
		for i, item := range objectScores {
			idx := i
			if item.Index != nil {
				idx = *item.Index
			}
			score := 0.0
			if item.Score != nil {
				score = *item.Score
			} else if item.RelevanceScore != nil {
				score = *item.RelevanceScore
			}
			items = append(items, RerankItem{Index: idx, Score: score})
		}
		return items, nil
	}

	// 3. Try parsing as a nested object: {"results": [{"index": 0, "score": 0.95}, ...]}
	var envelope struct {
		Results []GenericRerankItem `json:"results"`
	}
	if err := json.Unmarshal(bodyBytes, &envelope); err == nil && len(envelope.Results) > 0 {
		for i, item := range envelope.Results {
			idx := i
			if item.Index != nil {
				idx = *item.Index
			}
			score := 0.0
			if item.Score != nil {
				score = *item.Score
			} else if item.RelevanceScore != nil {
				score = *item.RelevanceScore
			}
			items = append(items, RerankItem{Index: idx, Score: score})
		}
		return items, nil
	}

	return nil, fmt.Errorf("could not parse rerank response format: %s", trimmed)
}
