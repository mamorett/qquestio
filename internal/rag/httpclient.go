package rag

import (
	"net/http"
	"time"
)

// HTTPTimeout is the global timeout applied to standard HTTP requests.
// It can be configured by the host application.
var HTTPTimeout = 60 * time.Second

// newHTTPClient creates an HTTP client with the specified timeout.
func newHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
	}
}
