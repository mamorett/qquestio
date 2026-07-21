package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type EmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// GetEmbedding calls the llama.cpp /embedding endpoint.
// Request:  POST { "model": model, "input": [query] }
// Response: { "data": [{ "embedding": [0.1, 0.2, ...] }] }
func GetEmbedding(ctx context.Context, baseURL, apiKey, model, text string) ([]float32, error) {
	url := AppendAPIPath(baseURL, "embeddings")

	reqBody := EmbeddingRequest{
		Model: model,
		Input: []string{text},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	client := newHTTPClient(HTTPTimeout)
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP status error: %d %s", resp.StatusCode, resp.Status)
	}

	var respBody EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&respBody); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(respBody.Data) == 0 {
		return nil, fmt.Errorf("received empty embedding data list")
	}

	return respBody.Data[0].Embedding, nil
}
