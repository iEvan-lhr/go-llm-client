package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/internal/requester"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

// clientImpl implements spec.Client.
type clientImpl struct {
	requester *requester.Requester
	config    spec.ClientConfig
}

// modelImpl implements spec.Model.
type modelImpl struct {
	client *clientImpl
	name   string
}

// NewClient creates an OpenAI-compatible client.
func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	config := spec.NewClientConfig()
	config.APIURL = "https://api.openai.com/v1/chat/completions"

	for _, opt := range opts {
		opt(config)
	}

	if config.APIKey == "" {
		return nil, fmt.Errorf("openai provider: API key is required, use spec.WithAPIKey()")
	}

	return &clientImpl{
		requester: &requester.Requester{
			HTTPClient: config.HTTPClient,
		},
		config: *config,
	}, nil
}

// Model implements spec.Client.
func (c *clientImpl) Model(name string) spec.Model {
	return &modelImpl{client: c, name: name}
}

// Chat selects the wire protocol from the configured endpoint. A /responses
// endpoint uses the Responses API; all other endpoints retain Chat Completions
// compatibility.
func (m *modelImpl) Chat(ctx context.Context, messages []spec.Message, opts ...spec.Option) (*spec.Response, error) {
	config := spec.NewRequestConfig()
	for _, opt := range opts {
		opt(config)
	}

	if isResponsesURL(m.client.config.APIURL) {
		return m.responses(ctx, messages, config)
	}
	return m.chatCompletions(ctx, messages, config)
}

func (m *modelImpl) chatCompletions(ctx context.Context, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	requestBody := cloneParameters(config.Parameters)
	requestBody["model"] = m.name
	requestBody["messages"] = messages

	if config.Temperature != nil {
		requestBody["temperature"] = *config.Temperature
	}
	if config.MaxTokens != nil {
		requestBody["max_tokens"] = *config.MaxTokens
	}
	if config.TopP != nil {
		requestBody["top_p"] = *config.TopP
	}
	if effort := effectiveReasoningEffort(config); effort != "" {
		requestBody["reasoning_effort"] = effort
	}
	if config.Streaming {
		requestBody["stream"] = true
	}

	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}

	var apiResp struct {
		Choices []struct {
			Message spec.Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rawBody, &apiResp); err != nil {
		return nil, fmt.Errorf("openai provider: failed to unmarshal response: %w", err)
	}

	var responseMessage spec.Message
	if len(apiResp.Choices) > 0 {
		responseMessage = apiResp.Choices[0].Message
	}

	return &spec.Response{
		Message:     responseMessage,
		RawResponse: rawBody,
	}, nil
}

func (m *modelImpl) responses(ctx context.Context, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	input, err := responsesInput(messages)
	if err != nil {
		return nil, err
	}

	requestBody := cloneParameters(config.Parameters)
	delete(requestBody, "messages")
	if maxTokens, ok := requestBody["max_tokens"]; ok {
		if _, exists := requestBody["max_output_tokens"]; !exists {
			requestBody["max_output_tokens"] = maxTokens
		}
		delete(requestBody, "max_tokens")
	}

	requestBody["model"] = m.name
	requestBody["input"] = input
	if config.Temperature != nil {
		requestBody["temperature"] = *config.Temperature
	}
	if config.MaxTokens != nil {
		requestBody["max_output_tokens"] = *config.MaxTokens
	}
	if config.TopP != nil {
		requestBody["top_p"] = *config.TopP
	}
	if effort := effectiveReasoningEffort(config); effort != "" {
		requestBody["reasoning"] = mergeReasoningConfig(requestBody["reasoning"], effort)
	}

	if config.Streaming {
		requestBody["stream"] = true
		return m.streamResponses(ctx, requestBody, config.StreamCallback)
	}

	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}

	message, err := parseResponsesMessage(rawBody)
	if err != nil {
		return nil, err
	}
	return &spec.Response{Message: message, RawResponse: rawBody}, nil
}

func (m *modelImpl) streamResponses(ctx context.Context, requestBody map[string]any, callback spec.StreamCallback) (*spec.Response, error) {
	resp, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var fullContent strings.Builder
	var terminalResponse json.RawMessage

	err = scanSSE(resp, func(data string) error {
		if data == "[DONE]" {
			return nil
		}

		var event struct {
			Type     string          `json:"type"`
			Delta    string          `json:"delta"`
			Code     string          `json:"code"`
			Message  string          `json:"message"`
			Response json.RawMessage `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return fmt.Errorf("openai responses: failed to decode stream event: %w", err)
		}

		switch event.Type {
		case "response.output_text.delta", "response.refusal.delta":
			if event.Delta == "" {
				return nil
			}
			fullContent.WriteString(event.Delta)
			if callback != nil {
				return callback(ctx, event.Delta)
			}
		case "response.completed", "response.incomplete":
			terminalResponse = append(terminalResponse[:0], event.Response...)
		case "response.failed":
			return responsesStreamFailure(event.Type, event.Response)
		case "error":
			if event.Message == "" {
				event.Message = data
			}
			if event.Code != "" {
				return fmt.Errorf("openai responses: stream error %s: %s", event.Code, event.Message)
			}
			return fmt.Errorf("openai responses: stream error: %s", event.Message)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	content := fullContent.String()
	role := spec.RoleAssistant
	if len(terminalResponse) > 0 {
		message, parseErr := parseResponsesMessage(terminalResponse)
		if parseErr != nil {
			return nil, parseErr
		}
		if content == "" {
			content = message.Content
		}
		if message.Role != "" {
			role = message.Role
		}
	}

	return &spec.Response{
		Message:     spec.Message{Role: role, Content: content},
		RawResponse: terminalResponse,
	}, nil
}

func (m *modelImpl) headers() http.Header {
	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)
	return headers
}

func cloneParameters(parameters map[string]any) map[string]any {
	cloned := make(map[string]any, len(parameters)+3)
	for key, value := range parameters {
		cloned[key] = value
	}
	return cloned
}

func effectiveReasoningEffort(config *spec.RequestConfig) spec.ReasoningEffort {
	if config.ReasoningEffort != "" {
		return config.ReasoningEffort
	}
	if config.Thinking == nil {
		return ""
	}
	if *config.Thinking {
		return spec.ReasoningEffortMedium
	}
	return spec.ReasoningEffortNone
}

func mergeReasoningConfig(value any, effort spec.ReasoningEffort) map[string]any {
	merged := make(map[string]any)
	switch reasoning := value.(type) {
	case map[string]any:
		for key, item := range reasoning {
			merged[key] = item
		}
	case map[string]string:
		for key, item := range reasoning {
			merged[key] = item
		}
	}
	merged["effort"] = string(effort)
	return merged
}

func isResponsesURL(apiURL string) bool {
	parsed, err := url.Parse(apiURL)
	if err != nil {
		return false
	}
	return strings.HasSuffix(strings.TrimRight(parsed.Path, "/"), "/responses")
}

type responsesInputMessage struct {
	Role    spec.Role `json:"role"`
	Content any       `json:"content"`
}

type responsesInputPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
}

func responsesInput(messages []spec.Message) ([]responsesInputMessage, error) {
	input := make([]responsesInputMessage, 0, len(messages))
	for _, message := range messages {
		if len(message.Parts) == 0 {
			input = append(input, responsesInputMessage{Role: message.Role, Content: message.Content})
			continue
		}

		parts := make([]responsesInputPart, 0, len(message.Parts))
		for _, part := range message.Parts {
			switch part.Type {
			case "text", "input_text":
				parts = append(parts, responsesInputPart{Type: "input_text", Text: part.Text})
			case "image_url", "input_image":
				if part.ImageURL == nil || part.ImageURL.URL == "" {
					return nil, fmt.Errorf("openai responses: image content part requires a URL")
				}
				parts = append(parts, responsesInputPart{
					Type:     "input_image",
					ImageURL: part.ImageURL.URL,
					Detail:   part.ImageURL.Detail,
				})
			case "input_file":
				if part.FileURL == "" {
					return nil, fmt.Errorf("openai responses: file content part requires a URL")
				}
				parts = append(parts, responsesInputPart{Type: "input_file", FileURL: part.FileURL})
			default:
				return nil, fmt.Errorf("openai responses: unsupported content part type %q", part.Type)
			}
		}
		input = append(input, responsesInputMessage{Role: message.Role, Content: parts})
	}
	return input, nil
}

type responsesResponse struct {
	OutputText string `json:"output_text"`
	Output     []struct {
		Role    spec.Role `json:"role"`
		Content []struct {
			Type    string `json:"type"`
			Text    string `json:"text"`
			Refusal string `json:"refusal"`
		} `json:"content"`
	} `json:"output"`
}

func parseResponsesMessage(rawBody []byte) (spec.Message, error) {
	var response responsesResponse
	if err := json.Unmarshal(rawBody, &response); err != nil {
		return spec.Message{}, fmt.Errorf("openai responses: failed to unmarshal response: %w", err)
	}

	role := spec.RoleAssistant
	var content strings.Builder
	if response.OutputText != "" {
		content.WriteString(response.OutputText)
	}
	for _, output := range response.Output {
		if output.Role != "" {
			role = output.Role
		}
		if response.OutputText != "" {
			continue
		}
		for _, part := range output.Content {
			switch part.Type {
			case "output_text":
				content.WriteString(part.Text)
			case "refusal":
				content.WriteString(part.Refusal)
			}
		}
	}

	return spec.Message{Role: role, Content: content.String()}, nil
}

func scanSSE(resp *http.Response, handleData func(string) error) error {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 10*1024*1024)

	var dataLines []string
	dispatch := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		return handleData(data)
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("openai responses: stream read failed: %w", err)
	}
	return dispatch()
}

func responsesStreamFailure(eventType string, rawResponse json.RawMessage) error {
	var response struct {
		Error *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(rawResponse, &response)

	if response.Error != nil && response.Error.Message != "" {
		return fmt.Errorf("openai responses: %s (%s): %s", eventType, response.Error.Code, response.Error.Message)
	}
	return fmt.Errorf("openai responses: %s", eventType)
}
