package rag

import (
	"net/http"
	"net/url"
	"strings"
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
	} else {
		customTransport = &http.Transport{}
	}
	customTransport.ResponseHeaderTimeout = 30 * time.Second

	// Wrap the proxy function to bypass proxies for loopback/localhost.
	// This prevents tests and local endpoints from hanging in proxy environments.
	originalProxy := customTransport.Proxy
	if originalProxy == nil {
		originalProxy = http.ProxyFromEnvironment
	}
	customTransport.Proxy = func(req *http.Request) (*url.URL, error) {
		host := req.URL.Hostname()
		if host == "localhost" || host == "127.0.0.1" || host == "::1" || strings.HasPrefix(host, "127.") {
			return nil, nil
		}
		return originalProxy(req)
	}

	return &http.Client{
		Timeout:   timeout,
		Transport: customTransport,
	}
}
