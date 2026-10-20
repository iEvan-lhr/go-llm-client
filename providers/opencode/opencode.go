// Package opencode implements OpenCode Console's System One API for Jev.
// OpenCode Go subscription chat models are provided by package opencodego.
package opencode

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/internal/requester"
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
	if config.UserAgent == "" {
		config.UserAgent = "go-llm-client/1.0"
	}
	if strings.ContainsAny(config.UserAgent, "\r\n") {
		return nil, fmt.Errorf("opencode System One: user agent must not contain newlines")
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
	// Marshal once to snapshot caller-owned maps and validate the actual wire
	// representation, including typed maps/slices and json.RawMessage values.
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("opencode System One: encode request: %w", err)
	}
	var wire struct {
		State     json.RawMessage `json:"state"`
		Questions map[string]struct {
			Type     spec.SystemOneQuestionType `json:"type"`
			Criteria json.RawMessage            `json:"criteria"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, fmt.Errorf("opencode System One: decode request: %w", err)
	}
	if len(wire.State) == 0 || !strings.ContainsRune("\"{[", rune(wire.State[0])) {
		return nil, fmt.Errorf("opencode System One: state must be a string, object, or array")
	}
	if len(wire.Questions) == 0 {
		return nil, fmt.Errorf("opencode System One: at least one question is required")
	}
	for id, question := range wire.Questions {
		switch question.Type {
		case spec.SystemOneNoul:
			if len(question.Criteria) > 0 && string(question.Criteria) != "null" && question.Criteria[0] != '{' {
				return nil, fmt.Errorf("opencode System One: noul question %q criteria must be an object", id)
			}
		case spec.SystemOneChoice:
			var criteria map[string]json.RawMessage
			if json.Unmarshal(question.Criteria, &criteria) != nil || len(criteria) < 1 || len(criteria) > 255 {
				return nil, fmt.Errorf("opencode System One: choice question %q requires 1..255 named criteria", id)
			}
		case spec.SystemOneScore:
			var criteria []json.RawMessage
			if json.Unmarshal(question.Criteria, &criteria) != nil || len(criteria) < 1 || len(criteria) > 10 {
				return nil, fmt.Errorf("opencode System One: score question %q requires 1..10 ordered criteria", id)
			}
		default:
			return nil, fmt.Errorf("opencode System One: question %q has unsupported type %q", id, question.Type)
		}
	}
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+c.config.APIKey)
	headers.Set("Content-Type", "application/json")
	headers.Set("User-Agent", c.config.UserAgent)
	raw, err := c.requester.Post(ctx, c.config.APIURL, headers, json.RawMessage(body))
	if err != nil {
		return nil, err
	}
	var response spec.SystemOneResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, fmt.Errorf("opencode System One: decode response: %w", err)
	}
	if response.Model == "" || len(response.Answers) != len(wire.Questions) {
		return nil, fmt.Errorf("opencode System One: response must contain a model and an answer for every question")
	}
	for id, question := range wire.Questions {
		answer, ok := response.Answers[id]
		if !ok || answer.Type != question.Type {
			return nil, fmt.Errorf("opencode System One: missing or mismatched answer for question %q", id)
		}
		switch answer.Type {
		case spec.SystemOneNoul:
			if answer.Noul == nil {
				return nil, fmt.Errorf("opencode System One: answer %q is missing noul", id)
			}
		case spec.SystemOneChoice, spec.SystemOneScore:
			if answer.Confidence == nil || len(answer.Probabilities) == 0 ||
				(answer.Type == spec.SystemOneScore && (answer.Score == nil || len(answer.Legend) == 0)) {
				return nil, fmt.Errorf("opencode System One: answer %q is missing decision fields", id)
			}
			if answer.Type == spec.SystemOneChoice {
				if _, ok := answer.Probabilities[answer.Choice]; !ok {
					return nil, fmt.Errorf("opencode System One: answer %q is missing its selected choice probability", id)
				}
			}
		}
	}
	response.RawResponse = raw
	return &response, nil
}
