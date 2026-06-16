package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type QdrantQueryRequest struct {
	Query       []float32 `json:"query"`
	Limit       int       `json:"limit"`
	WithPayload bool      `json:"with_payload"`
}

type QdrantPoint struct {
	ID      interface{}            `json:"id"`
	Payload map[string]interface{} `json:"payload"`
	Score   float32                `json:"score"`
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
// Request:  POST /collections/{collection}/points/query
//           { "query": vector, "limit": limit, "with_payload": true }
// Response: Extract ONLY the text field from each point's payload.
// Returns:  Concatenated text payloads separated by "\n---\n", the points list, and an error if any.
func SearchQdrant(ctx context.Context, baseURL, apiKey, collection string, vector []float32, limit int) (string, []QdrantPoint, error) {
	url := fmt.Sprintf("%s/collections/%s/points/query", strings.TrimSuffix(baseURL, "/"), collection)

	reqBody := QdrantQueryRequest{
		Query:       vector,
		Limit:       limit,
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

	var texts []string
	for _, pt := range respBody.Result {
		textStr := pt.ExtractText()
		if textStr != "" {
			texts = append(texts, textStr)
		}
	}

	return strings.Join(texts, "\n---\n"), respBody.Result, nil
}

// ExtractText extracts the main text payload content from a QdrantPoint.
func (pt QdrantPoint) ExtractText() string {
	if pt.Payload == nil {
		return ""
	}
	for _, key := range []string{"text", "content", "document", "page_content", "description", "body"} {
		if val, ok := pt.Payload[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				return s
			}
		}
	}
	var parts []string
	for k, v := range pt.Payload {
		isMeta := false
		for _, mk := range []string{"file", "filename", "file_name", "source", "title", "url", "path", "id", "score", "page", "author", "date", "created_at"} {
			if k == mk {
				isMeta = true
				break
			}
		}
		if !isMeta {
			if s, ok := v.(string); ok && len(s) > 5 {
				parts = append(parts, s)
			}
		}
	}
	if len(parts) > 0 {
		return strings.Join(parts, " ")
	}
	return ""
}

type QdrantCollectionInfo struct {
	Result struct {
		Status       string `json:"status"`
		PointsCount  int    `json:"points_count"`
		VectorsCount int    `json:"vectors_count"`
	} `json:"result"`
	Status string `json:"status"`
}

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
