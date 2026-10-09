// Package opencodego implements the OpenCode Go subscription provider.
package opencodego

import (
	"context"
	"crypto/rand"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/internal/requester"
	"github.com/iEvan-lhr/go-llm-client/providers/openai"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

const (
	DefaultAPIURL    = "https://opencode.ai/zen/go/v1"
	DefaultUserAgent = "go-llm-client/1.0"
)

type clientImpl struct {
	config    spec.ClientConfig
	requester *requester.Requester
}

type modelImpl struct {
	client *clientImpl
	name   string
}

// NewClient creates an OpenCode Go client. APIURL accepts an API root ending
// in /v1, or a complete endpoint that overrides automatic protocol selection.
// An omitted SessionID is generated once and reused for this client's calls.
func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	config := spec.NewClientConfig()
	config.APIURL = DefaultAPIURL
	for _, opt := range opts {
		opt(config)
	}
	if config.APIKey == "" {
		return nil, fmt.Errorf("opencode-go provider: API key is required, use spec.WithAPIKey()")
	}
	parsed, err := url.Parse(config.APIURL)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "https" && parsed.Scheme != "http") {
		return nil, fmt.Errorf("opencode-go provider: API URL must be an absolute HTTP(S) URL")
	}
	if config.HTTPClient == nil {
		return nil, fmt.Errorf("opencode-go provider: HTTP client is required")
	}
	if config.SessionID == "" {
		config.SessionID = rand.Text()
	}
	if config.UserAgent == "" {
		config.UserAgent = DefaultUserAgent
	}
	if strings.ContainsAny(config.SessionID+config.UserAgent, "\r\n") {
		return nil, fmt.Errorf("opencode-go provider: session ID and user agent must not contain newlines")
	}
	// Copy the caller's HTTP client so wrapping its transport doesn't change
	// requests made by other providers using that client.
	httpClient := *config.HTTPClient
	transport := httpClient.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	httpClient.Transport = &sessionTransport{transport, config.SessionID, config.UserAgent}
	config.HTTPClient = &httpClient
	return &clientImpl{
		config:    *config,
		requester: &requester.Requester{HTTPClient: config.HTTPClient},
	}, nil
}

func (c *clientImpl) Model(name string) spec.Model {
	return &modelImpl{client: c, name: strings.TrimPrefix(name, "opencode-go/")}
}

func (m *modelImpl) Chat(ctx context.Context, messages []spec.Message, opts ...spec.Option) (*spec.Response, error) {
	config := spec.NewRequestConfig()
	for _, opt := range opts {
		opt(config)
	}
	name := m.name
	if config.Model != "" {
		name = strings.TrimPrefix(config.Model, "opencode-go/")
	}
	if name == "" {
		return nil, fmt.Errorf("opencode-go provider: model is required")
	}
	parameters, err := spec.SnapshotParameters(config.Parameters)
	if err != nil {
		return nil, err
	}
	if parameters == nil {
		parameters = make(map[string]any)
	}
	config.Parameters = parameters
	endpoint, protocol := modelEndpoint(m.client.config.APIURL, name)
	if protocol == spec.ProtocolAnthropicMessages {
		return m.messages(ctx, endpoint, name, spec.CloneMessages(messages), config)
	}
	// These compatible model families use thinking.type instead of OpenAI's
	// reasoning_effort to enable/disable reasoning. An explicit effort remains.
	if protocol == spec.ProtocolChatCompletions && config.Thinking != nil &&
		(strings.HasPrefix(name, "glm-") || strings.HasPrefix(name, "kimi-") || strings.HasPrefix(name, "deepseek-")) {
		kind := "disabled"
		if *config.Thinking {
			kind = "enabled"
		}
		thinking, _ := parameters["thinking"].(map[string]any)
		if thinking == nil {
			thinking = make(map[string]any)
		}
		thinking["type"] = kind
		parameters["thinking"] = thinking
		config.Thinking = nil
	}
	delegate, err := openai.NewClient(
		spec.WithAPIKey(m.client.config.APIKey),
		spec.WithAPIURL(endpoint),
		spec.WithHTTPClient(m.client.config.HTTPClient),
	)
	if err != nil {
		return nil, err
	}
	return delegate.Model(name).Chat(ctx, messages, func(target *spec.RequestConfig) { *target = *config })
}

// Protocol routing follows https://opencode.ai/v2/docs/console/go/#endpoints.
// Unknown models use Chat Completions; an explicit endpoint supports future
// models without requiring a library update.
func modelEndpoint(apiURL, model string) (string, spec.Protocol) {
	parsed, _ := url.Parse(apiURL) // Validated by NewClient.
	path := strings.TrimRight(parsed.Path, "/")
	if strings.HasSuffix(path, "/messages") {
		return apiURL, spec.ProtocolAnthropicMessages
	}
	if strings.HasSuffix(path, "/responses") {
		parsed.Path = path
		return parsed.String(), spec.ProtocolResponses
	}
	if !strings.HasSuffix(path, "/v1") {
		return apiURL, spec.ProtocolChatCompletions
	}
	protocol, suffix := spec.ProtocolChatCompletions, "/chat/completions"
	switch {
	case strings.HasPrefix(model, "gpt-"), strings.HasPrefix(model, "grok-"), strings.HasPrefix(model, "muse-spark-"):
		protocol, suffix = spec.ProtocolResponses, "/responses"
	case strings.HasPrefix(model, "claude-"), strings.HasPrefix(model, "minimax-"), strings.HasPrefix(model, "qwen3."):
		protocol, suffix = spec.ProtocolAnthropicMessages, "/messages"
	}
	parsed.Path = path + suffix
	return parsed.String(), protocol
}

type sessionTransport struct {
	base      http.RoundTripper
	sessionID string
	userAgent string
}

func (t *sessionTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	clone := request.Clone(request.Context())
	clone.Header.Set("User-Agent", t.userAgent)
	clone.Header.Set("x-opencode-session", t.sessionID)
	return t.base.RoundTrip(clone)
}
