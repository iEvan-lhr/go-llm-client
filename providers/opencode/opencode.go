// Package opencode implements OpenCode Console's System One API for Jev.
// OpenCode Go subscription chat models are provided by package opencodego.
package opencode

import (
	"context"
	"crypto/rand"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/internal/requester"
	"github.com/iEvan-lhr/go-llm-client/internal/systemone"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

const (
	DefaultAPIURL = "https://opencode.ai/inference/systemone/v1/systemone"
	DefaultModel  = "jev-1.13-free"
)

type clientImpl struct {
	config    spec.ClientConfig
	requester *requester.Requester
}

// NewClient uses an OpenCode Console API key. APIURL may override the complete
// System One endpoint or a System One root ending in /v1. It must not be the Go
// subscription root. The default model is the limited-time free Jev variant.
// An omitted SessionID is generated once and reused for this client's calls.
func NewClient(opts ...spec.ClientOption) (spec.SystemOneClient, error) {
	config := spec.NewClientConfig()
	config.APIURL = DefaultAPIURL
	config.UserAgent = "go-llm-client/1.0"
	for _, opt := range opts {
		opt(config)
	}
	if strings.TrimSpace(config.APIKey) == "" {
		return nil, fmt.Errorf("opencode System One: OpenCode Console API key is required")
	}
	if config.HTTPClient == nil {
		return nil, fmt.Errorf("opencode System One: HTTP client is required")
	}
	parsed, err := url.Parse(config.APIURL)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, fmt.Errorf("opencode System One: API URL must be an absolute HTTP(S) URL")
	}
	path := strings.TrimRight(parsed.Path, "/")
	if strings.Contains(path+"/", "/zen/go/") {
		return nil, fmt.Errorf("opencode System One: Jev requires the Console inference endpoint, not the Go subscription endpoint")
	}
	if strings.HasSuffix(path, "/v1") {
		path += "/systemone"
	}
	parsed.Path, parsed.RawPath = path, ""
	config.APIURL = parsed.String()
	if config.SessionID == "" {
		config.SessionID = rand.Text()
	}
	if config.UserAgent == "" {
		config.UserAgent = "go-llm-client/1.0"
	}
	if strings.ContainsAny(config.SessionID+config.UserAgent, "\r\n") {
		return nil, fmt.Errorf("opencode System One: session ID and user agent must not contain newlines")
	}
	return &clientImpl{config: *config, requester: &requester.Requester{HTTPClient: config.HTTPClient}}, nil
}

// Model satisfies spec.Client; System One must be called through SystemOne.
func (c *clientImpl) Model(name string) spec.Model { return systemOneModel{} }

type systemOneModel struct{}

func (systemOneModel) Chat(context.Context, []spec.Message, ...spec.Option) (*spec.Response, error) {
	return nil, fmt.Errorf("opencode: Jev is a System One decision model; use SystemOne with state and questions instead of Chat/Send/SendStream")
}

func (c *clientImpl) SystemOne(ctx context.Context, request spec.SystemOneRequest) (*spec.SystemOneResponse, error) {
	request.Model = strings.TrimPrefix(strings.TrimSpace(request.Model), "opencode/")
	if request.Model == "" {
		request.Model = DefaultModel
	}
	if !strings.HasPrefix(request.Model, "jev-") {
		return nil, fmt.Errorf("opencode System One: expected a Jev model, got %q", request.Model)
	}
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+c.config.APIKey)
	headers.Set("Content-Type", "application/json")
	headers.Set("User-Agent", c.config.UserAgent)
	headers.Set("x-opencode-session", c.config.SessionID)
	return systemone.Do(ctx, c.requester, c.config.APIURL, headers, request)
}
