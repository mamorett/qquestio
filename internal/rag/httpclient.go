package rag

import (
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// HTTPTimeout is the global timeout applied to standard HTTP requests.
// It can be configured by the host application.
var HTTPTimeout = 60 * time.Second

// VerboseLogging enables detailed diagnostic logging from internal/rag routines.
// Disabled by default to avoid corrupting terminal UI output.
var VerboseLogging bool

var (
	sharedTransport   *http.Transport
	initTransportOnce sync.Once
)

func getSharedTransport() *http.Transport {
	initTransportOnce.Do(func() {
		transport, ok := http.DefaultTransport.(*http.Transport)
		if ok {
			sharedTransport = transport.Clone()
		} else {
			sharedTransport = &http.Transport{}
		}
		// Let the client timeout govern the request lifecycle. Setting a hardcoded
		// ResponseHeaderTimeout of 30 seconds overrides the client timeout and triggers
		// premature timeouts on large databases/slow queries.
		sharedTransport.ResponseHeaderTimeout = 0

		// Wrap the proxy function to bypass proxies for local and private addresses.
		// This prevents local subnets and test endpoints from hanging in proxy environments.
		originalProxy := sharedTransport.Proxy
		if originalProxy == nil {
			originalProxy = http.ProxyFromEnvironment
		}
		sharedTransport.Proxy = func(req *http.Request) (*url.URL, error) {
			host := req.URL.Hostname()
			if isLocalOrPrivateHost(host) {
				return nil, nil
			}
			return originalProxy(req)
		}
	})
	return sharedTransport
}

// newHTTPClient creates an HTTP client with the specified timeout.
func newHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout:   timeout,
		Transport: getSharedTransport(),
	}
}

func isLocalOrPrivateHost(host string) bool {
	if host == "localhost" || host == "::1" || strings.HasPrefix(host, "127.") {
		return true
	}

	// Check if host is an IP
	if ip := net.ParseIP(host); ip != nil {
		if ip.IsLoopback() {
			return true
		}
		ip4 := ip.To4()
		if ip4 != nil {
			// 10.0.0.0/8
			if ip4[0] == 10 {
				return true
			}
			// 172.16.0.0/12
			if ip4[0] == 172 && ip4[1] >= 16 && ip4[1] <= 31 {
				return true
			}
			// 192.168.0.0/16
			if ip4[0] == 192 && ip4[1] == 168 {
				return true
			}
		}
		ip16 := ip.To16()
		if ip16 != nil {
			// fc00::/7 (ULA)
			if (ip16[0] & 0xfe) == 0xfc {
				return true
			}
		}
		return false
	}

	// Check if host is a local hostname (no dots) or ends with .local
	if !strings.Contains(host, ".") || strings.HasSuffix(host, ".local") {
		return true
	}

	return false
}
