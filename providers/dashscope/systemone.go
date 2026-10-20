package dashscope

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/internal/systemone"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

const DefaultSystemOneModel = "decision-model-preview"

var _ spec.SystemOneClient = (*clientImpl)(nil)

// SystemOne calls the decision-model API on the configured DashScope/MaaS
// deployment. Set APIURL to the deployment's /compatible-mode/v1/systemone
// endpoint or /compatible-mode/v1 root. Chat options and history do not apply.
func (c *clientImpl) SystemOne(ctx context.Context, request spec.SystemOneRequest) (*spec.SystemOneResponse, error) {
	if c.config.HTTPClient == nil {
		return nil, fmt.Errorf("dashscope System One: HTTP client is required")
	}
	endpoint, err := systemOneEndpoint(c.config.APIURL)
	if err != nil {
		return nil, err
	}
	request.Model = strings.TrimPrefix(strings.TrimSpace(request.Model), "dashscope/")
	if request.Model == "" {
		request.Model = DefaultSystemOneModel
	}
	if strings.ContainsAny(c.config.UserAgent, "\r\n") {
		return nil, fmt.Errorf("dashscope System One: user agent must not contain newlines")
	}
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+c.config.APIKey)
	headers.Set("Content-Type", "application/json")
	if c.config.UserAgent != "" {
		headers.Set("User-Agent", c.config.UserAgent)
	}
	return systemone.Do(ctx, c.requester, endpoint, headers, request)
}

func systemOneEndpoint(apiURL string) (string, error) {
	parsed, err := url.Parse(apiURL)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "https" && parsed.Scheme != "http") {
		return "", fmt.Errorf("dashscope System One: API URL must be an absolute HTTP(S) URL")
	}
	path := strings.TrimRight(parsed.Path, "/")
	switch {
	case strings.HasSuffix(path, "/systemone"):
	case strings.HasSuffix(path, "/v1/chat/completions"):
		path = strings.TrimSuffix(path, "/chat/completions") + "/systemone"
	case strings.HasSuffix(path, "/v1"):
		path += "/systemone"
	default:
		return "", fmt.Errorf("dashscope System One: set APIURL to the deployment's /v1 root or complete /systemone endpoint")
	}
	parsed.Path, parsed.RawPath = path, ""
	return parsed.String(), nil
}
