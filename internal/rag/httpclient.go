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
	transport, ok := http.DefaultTransport.(*http.Transport)
	var customTransport *http.Transport
	if ok {
		customTransport = transport.Clone()
		customTransport.ResponseHeaderTimeout = 30 * time.Second
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: customTransport,
	}
}
