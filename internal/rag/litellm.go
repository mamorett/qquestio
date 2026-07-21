package rag

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// SSEReader wraps a bufio.Scanner over an SSE HTTP response body.
type SSEReader struct {
	body    io.ReadCloser
	scanner *bufio.Scanner
}

// ChatMessage represents a single message in the LLM chat history.
type ChatMessage struct {
	Role    string `json:"role"` // "system" | "user" | "assistant"
	Content string `json:"content"`
}

type LiteLLMRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

// StartLiteLLMStream opens the SSE connection.
// Request: POST /chat/completions { "model": m, "messages": msgs, "stream": true }
func StartLiteLLMStream(ctx context.Context, baseURL, apiKey, model string, messages []ChatMessage) (*SSEReader, error) {
	url := AppendAPIPath(baseURL, "chat/completions")

	reqBody := LiteLLMRequest{
		Model:    model,
		Messages: messages,
		Stream:   true,
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
	req.Header.Set("Accept", "text/event-stream")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	client := newHTTPClient(0)
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP stream request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Read body for error details
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP stream status error: %d %s, response: %s", resp.StatusCode, resp.Status, string(bodyBytes))
	}

	return &SSEReader{
		body:    resp.Body,
		scanner: bufio.NewScanner(resp.Body),
	}, nil
}

// Next reads one SSE chunk. Returns (content, isDone, error).
// Parses "data: {...}" lines, extracts choices[0].delta.content.
// Returns done=true on "data: [DONE]".
func (r *SSEReader) Next() (string, bool, error) {
	for r.scanner.Scan() {
		line := r.scanner.Text()
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Skip event metadata or comment lines
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		dataVal := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if dataVal == "[DONE]" {
			return "", true, nil
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(dataVal), &chunk); err != nil {
			return "", false, fmt.Errorf("failed to parse SSE chunk: %w", err)
		}

		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			return content, false, nil
		}
	}

	if err := r.scanner.Err(); err != nil {
		return "", false, err
	}

	return "", true, nil
}

// Close closes the underlying response body.
func (r *SSEReader) Close() error {
	if r.body != nil {
		return r.body.Close()
	}
	return nil
}
