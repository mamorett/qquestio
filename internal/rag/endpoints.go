package rag

import "strings"

// AppendAPIPath normalizes a base API URL and appends the path suffix if not already present.
func AppendAPIPath(baseURL, suffix string) string {
	url := baseURL
	
	// Normalize checks for embeddings vs embedding.
	checkSuffix := suffix
	if suffix == "embeddings" {
		checkSuffix = "embedding"
	}
	
	if !strings.Contains(url, "/"+checkSuffix) {
		trimmed := strings.TrimSuffix(url, "/")
		if strings.HasSuffix(trimmed, "/v1") {
			url = trimmed + "/" + suffix
		} else {
			url = trimmed + "/v1/" + suffix
		}
	}
	return url
}
