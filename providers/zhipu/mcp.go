package zhipu

import (
	"context"
	"fmt"
	"github.com/iEvan-lhr/go-llm-client/mcp"
	"net"
	"net/url"
	"strings"
	"time"
)

var allowedMCPURLs = map[string]bool{"https://open.bigmodel.cn/api/anthropic": true, "https://open.bigmodel.cn/api/coding/paas/v4": true, "https://open.bigmodel.cn/api/v1": true}

func MCPAllowed(apiURL string) bool { return allowedMCPURLs[mcp.NormalizeURL(apiURL)] }

type SearchOptions struct{}
type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet,omitempty"`
	Source  string `json:"source,omitempty"`
	Favicon string `json:"favicon,omitempty"`
	Raw     any    `json:"-"`
}
type WebSearchTool struct{ client *mcp.Client }

func (t *WebSearchTool) Search(ctx context.Context, q string, _ SearchOptions) ([]SearchResult, error) {
	if strings.TrimSpace(q) == "" {
		return nil, fmt.Errorf("zhipu mcp: query is required")
	}
	var r struct {
		Results []SearchResult `json:"results"`
	}
	err := t.client.Call(ctx, "tools/call", map[string]any{"name": "webSearchPrime", "arguments": map[string]any{"query": q}}, &r)
	return r.Results, err
}

type ReadOptions struct{}
type WebPage struct {
	URL      string         `json:"url"`
	Title    string         `json:"title"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata,omitempty"`
	Links    []string       `json:"links,omitempty"`
	Raw      any            `json:"-"`
}
type WebReaderTool struct{ client *mcp.Client }

func (t *WebReaderTool) Read(ctx context.Context, u string, _ ReadOptions) (*WebPage, error) {
	p, e := url.Parse(u)
	if e != nil || p.Scheme != "http" && p.Scheme != "https" {
		return nil, fmt.Errorf("zhipu mcp: invalid URL")
	}
	ip := net.ParseIP(p.Hostname())
	if p.Hostname() == "localhost" || ip != nil && (ip.IsLoopback() || ip.IsPrivate()) {
		return nil, fmt.Errorf("zhipu mcp: SSRF URL rejected")
	}
	var r WebPage
	e = t.client.Call(ctx, "tools/call", map[string]any{"name": "webReader", "arguments": map[string]any{"url": u}}, &r)
	return &r, e
}
func newMCPTool(base, token, endpoint string, timeout time.Duration, retries int) *mcp.Client {
	return &mcp.Client{URL: endpoint, Token: token, Timeout: timeout, MaxRetries: retries}
}

// NewMCPTools creates the provider-neutral search and reader tools for one request URL.
func NewMCPTools(apiURL, token string, timeout time.Duration, retries int) (*WebSearchTool, *WebReaderTool, error) {
	if token == "" {
		return nil, nil, fmt.Errorf("zhipu mcp: API key is required")
	}
	if !MCPAllowed(apiURL) {
		return nil, nil, fmt.Errorf("当前调用地址不支持智谱 MCP")
	}
	return &WebSearchTool{newMCPTool(apiURL, token, "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp", timeout, retries)}, &WebReaderTool{newMCPTool(apiURL, token, "https://open.bigmodel.cn/api/mcp/web_reader/mcp", timeout, retries)}, nil
}
