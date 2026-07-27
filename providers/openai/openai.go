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
	requestBody["messages"] = chatMessages(messages)

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
		return m.streamChatCompletions(ctx, requestBody, config)
	}

	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}

	var apiResp spec.ChatCompletionResponse
	if err := json.Unmarshal(rawBody, &apiResp); err != nil {
		return nil, fmt.Errorf("openai provider: failed to unmarshal response: %w", err)
	}

	var responseMessage spec.Message
	if len(apiResp.Choices) > 0 {
		responseMessage = apiResp.Choices[0].Message
		if responseMessage.Content == "" {
			responseMessage.Content = responseMessage.Refusal
		}
	}

	return &spec.Response{
		Protocol:       spec.ProtocolChatCompletions,
		ID:             apiResp.ID,
		Model:          apiResp.Model,
		Status:         chatFinishReason(apiResp.Choices),
		Message:        responseMessage,
		Usage:          apiResp.Usage,
		ChatCompletion: &apiResp,
		RawResponse:    rawBody,
	}, nil
}

func (m *modelImpl) responses(ctx context.Context, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	requestBody := cloneParameters(config.Parameters)
	delete(requestBody, "messages")
	if maxTokens, ok := requestBody["max_tokens"]; ok {
		if _, exists := requestBody["max_output_tokens"]; !exists {
			requestBody["max_output_tokens"] = maxTokens
		}
		delete(requestBody, "max_tokens")
	}

	requestBody["model"] = m.name
	switch {
	case config.ResponseInput != nil:
		requestBody["input"] = config.ResponseInput
	case requestBody["input"] != nil:
		// Keep a forward-compatible input supplied through Parameters.
	default:
		input, err := responsesInput(messages)
		if err != nil {
			return nil, err
		}
		requestBody["input"] = input
	}
	if config.Instructions != nil {
		requestBody["instructions"] = config.Instructions
	}
	if config.PreviousResponseID != "" {
		requestBody["previous_response_id"] = config.PreviousResponseID
	}
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
		return m.streamResponses(ctx, requestBody, config)
	}

	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}

	parsed, message, err := parseResponses(rawBody)
	if err != nil {
		return nil, err
	}
	return newResponsesResult(parsed, message, rawBody), nil
}

func (m *modelImpl) streamChatCompletions(ctx context.Context, requestBody map[string]any, config *spec.RequestConfig) (*spec.Response, error) {
	resp, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var completion spec.ChatCompletionResponse
	var content strings.Builder
	var refusal strings.Builder
	var reasoning strings.Builder
	role := spec.RoleAssistant
	toolCalls := make(map[int]*spec.ToolCall)
	toolOrder := make([]int, 0)

	err = scanSSE(resp, func(data string) error {
		if data == "[DONE]" {
			return nil
		}

		raw := json.RawMessage(append([]byte(nil), data...))
		if config.EventCallback != nil {
			if err := config.EventCallback(ctx, spec.StreamEvent{
				Protocol: spec.ProtocolChatCompletions,
				Type:     "chat.completion.chunk",
				Raw:      raw,
			}); err != nil {
				return err
			}
		}

		var streamError struct {
			Error *spec.APIError `json:"error"`
		}
		if err := json.Unmarshal(raw, &streamError); err == nil && streamError.Error != nil {
			return fmt.Errorf("openai chat stream error (%s): %s", streamError.Error.Code, streamError.Error.Message)
		}

		var chunk spec.ChatCompletionResponse
		if err := json.Unmarshal(raw, &chunk); err != nil {
			return fmt.Errorf("openai chat: failed to decode stream chunk: %w", err)
		}
		mergeChatCompletionMetadata(&completion, &chunk)

		for _, choice := range chunk.Choices {
			if choice.Index != 0 {
				continue
			}
			delta := choice.Delta
			if delta.Role != "" {
				role = delta.Role
			}
			if delta.Content != "" {
				content.WriteString(delta.Content)
				if config.StreamCallback != nil {
					if err := config.StreamCallback(ctx, delta.Content); err != nil {
						return err
					}
				}
			}
			if delta.Refusal != "" {
				refusal.WriteString(delta.Refusal)
				if content.Len() == 0 && config.StreamCallback != nil {
					if err := config.StreamCallback(ctx, delta.Refusal); err != nil {
						return err
					}
				}
			}
			if delta.ReasoningContent != "" {
				reasoning.WriteString(delta.ReasoningContent)
				if config.ReasoningCallback != nil {
					if err := config.ReasoningCallback(ctx, delta.ReasoningContent); err != nil {
						return err
					}
				}
			}
			mergeToolCallDeltas(toolCalls, &toolOrder, delta.ToolCalls)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	message := spec.Message{
		Role:             role,
		Content:          content.String(),
		Refusal:          refusal.String(),
		ReasoningContent: reasoning.String(),
		ToolCalls:        orderedToolCalls(toolCalls, toolOrder),
	}
	if message.Content == "" {
		message.Content = message.Refusal
	}
	if len(completion.Choices) == 0 {
		completion.Choices = []spec.ChatChoice{{Index: 0, Message: message}}
	} else {
		completion.Choices[0].Message = message
	}
	rawBody, _ := json.Marshal(completion)

	return &spec.Response{
		Protocol:       spec.ProtocolChatCompletions,
		ID:             completion.ID,
		Model:          completion.Model,
		Status:         chatFinishReason(completion.Choices),
		Message:        message,
		Usage:          completion.Usage,
		ChatCompletion: &completion,
		RawResponse:    rawBody,
	}, nil
}

func (m *modelImpl) streamResponses(ctx context.Context, requestBody map[string]any, config *spec.RequestConfig) (*spec.Response, error) {
	resp, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var fullContent strings.Builder
	var refusal strings.Builder
	var reasoning strings.Builder
	var terminalResponse json.RawMessage

	err = scanSSE(resp, func(data string) error {
		if data == "[DONE]" {
			return nil
		}

		raw := json.RawMessage(append([]byte(nil), data...))
		var event struct {
			spec.StreamEvent
			Code    string         `json:"code"`
			Message string         `json:"message"`
			Error   *spec.APIError `json:"error"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			return fmt.Errorf("openai responses: failed to decode stream event: %w", err)
		}
		event.Protocol = spec.ProtocolResponses
		event.Raw = raw
		if config.EventCallback != nil {
			if err := config.EventCallback(ctx, event.StreamEvent); err != nil {
				return err
			}
		}

		switch event.Type {
		case "response.output_text.delta":
			if event.Delta == "" {
				return nil
			}
			fullContent.WriteString(event.Delta)
			if config.StreamCallback != nil {
				return config.StreamCallback(ctx, event.Delta)
			}
		case "response.refusal.delta":
			refusal.WriteString(event.Delta)
			if fullContent.Len() == 0 && config.StreamCallback != nil {
				return config.StreamCallback(ctx, event.Delta)
			}
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			reasoning.WriteString(event.Delta)
			if config.ReasoningCallback != nil {
				return config.ReasoningCallback(ctx, event.Delta)
			}
		case "response.completed", "response.incomplete":
			terminalResponse = append(terminalResponse[:0], event.Response...)
		case "response.failed":
			terminalResponse = append(terminalResponse[:0], event.Response...)
			return responsesStreamFailure(event.Type, event.Response)
		case "error":
			if event.Error != nil && event.Message == "" {
				event.Code = event.Error.Code
				event.Message = event.Error.Message
			}
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

	if len(terminalResponse) > 0 {
		parsed, message, parseErr := parseResponses(terminalResponse)
		if parseErr != nil {
			return nil, parseErr
		}
		if fullContent.Len() > 0 {
			message.Content = fullContent.String()
		}
		if refusal.Len() > 0 {
			message.Refusal = refusal.String()
			if message.Content == "" {
				message.Content = refusal.String()
			}
		}
		if reasoning.Len() > 0 {
			message.ReasoningContent = reasoning.String()
		}
		return newResponsesResult(parsed, message, terminalResponse), nil
	}

	parsed := &spec.ResponsesAPIResponse{
		Model:      fmt.Sprint(requestBody["model"]),
		Status:     "completed",
		OutputText: fullContent.String(),
	}
	message := spec.Message{
		Role:             spec.RoleAssistant,
		Content:          fullContent.String(),
		Refusal:          refusal.String(),
		ReasoningContent: reasoning.String(),
	}
	if message.Content == "" {
		message.Content = message.Refusal
	}
	rawBody, _ := json.Marshal(parsed)
	return newResponsesResult(parsed, message, rawBody), nil
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

func chatFinishReason(choices []spec.ChatChoice) string {
	if len(choices) == 0 || choices[0].FinishReason == nil {
		return ""
	}
	return *choices[0].FinishReason
}

func mergeChatCompletionMetadata(target, chunk *spec.ChatCompletionResponse) {
	if target.ID == "" {
		target.ID = chunk.ID
		target.Object = chunk.Object
		target.Created = chunk.Created
		target.Model = chunk.Model
		target.ServiceTier = chunk.ServiceTier
		target.SystemFingerprint = chunk.SystemFingerprint
	}
	if chunk.Usage != nil {
		target.Usage = chunk.Usage
	}
	for _, choice := range chunk.Choices {
		if choice.FinishReason == nil {
			continue
		}
		for len(target.Choices) <= choice.Index {
			target.Choices = append(target.Choices, spec.ChatChoice{Index: len(target.Choices)})
		}
		target.Choices[choice.Index].FinishReason = choice.FinishReason
	}
}

func mergeToolCallDeltas(calls map[int]*spec.ToolCall, order *[]int, deltas []spec.ToolCall) {
	for position, delta := range deltas {
		index := position
		if delta.Index != nil {
			index = *delta.Index
		}
		call, exists := calls[index]
		if !exists {
			call = &spec.ToolCall{}
			calls[index] = call
			*order = append(*order, index)
		}
		if delta.ID != "" {
			call.ID = delta.ID
		}
		if delta.Type != "" {
			call.Type = delta.Type
		}
		if delta.Function.Name != "" {
			call.Function.Name += delta.Function.Name
		}
		call.Function.Arguments += delta.Function.Arguments
	}
}

func orderedToolCalls(calls map[int]*spec.ToolCall, order []int) []spec.ToolCall {
	result := make([]spec.ToolCall, 0, len(order))
	for _, index := range order {
		call := *calls[index]
		indexCopy := index
		call.Index = &indexCopy
		result = append(result, call)
	}
	return result
}

func chatMessages(messages []spec.Message) []any {
	result := make([]any, 0, len(messages))
	for _, message := range messages {
		if len(message.Parts) == 0 {
			clone := message
			result = append(result, &clone)
			continue
		}

		parts := make([]any, 0, len(message.Parts))
		for _, part := range message.Parts {
			switch part.Type {
			case "text", "input_text":
				parts = append(parts, map[string]any{"type": "text", "text": part.Text})
			case "image_url", "input_image":
				parts = append(parts, map[string]any{"type": "image_url", "image_url": part.ImageURL})
			case "file", "input_file":
				file := make(map[string]any)
				if part.FileID != "" {
					file["file_id"] = part.FileID
				}
				if part.FileData != "" {
					file["file_data"] = part.FileData
				}
				if part.FileURL != "" {
					file["file_url"] = part.FileURL
				}
				if part.Filename != "" {
					file["filename"] = part.Filename
				}
				parts = append(parts, map[string]any{"type": "file", "file": file})
			default:
				parts = append(parts, part)
			}
		}
		wire := map[string]any{
			"role":    message.Role,
			"content": parts,
		}
		if message.Name != "" {
			wire["name"] = message.Name
		}
		if message.ToolCallID != "" {
			wire["tool_call_id"] = message.ToolCallID
		}
		if len(message.ToolCalls) > 0 {
			wire["tool_calls"] = message.ToolCalls
		}
		result = append(result, wire)
	}
	return result
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
	FileID   string `json:"file_id,omitempty"`
	FileData string `json:"file_data,omitempty"`
	Filename string `json:"filename,omitempty"`
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
			case "file", "input_file":
				if part.FileURL == "" && part.FileID == "" && part.FileData == "" {
					return nil, fmt.Errorf("openai responses: file content part requires file_url, file_id, or file_data")
				}
				parts = append(parts, responsesInputPart{
					Type:     "input_file",
					FileURL:  part.FileURL,
					FileID:   part.FileID,
					FileData: part.FileData,
					Filename: part.Filename,
				})
			default:
				return nil, fmt.Errorf("openai responses: unsupported content part type %q", part.Type)
			}
		}
		input = append(input, responsesInputMessage{Role: message.Role, Content: parts})
	}
	return input, nil
}

func parseResponses(rawBody []byte) (*spec.ResponsesAPIResponse, spec.Message, error) {
	var response spec.ResponsesAPIResponse
	if err := json.Unmarshal(rawBody, &response); err != nil {
		return nil, spec.Message{}, fmt.Errorf("openai responses: failed to unmarshal response: %w", err)
	}

	role := spec.RoleAssistant
	var content strings.Builder
	var refusal strings.Builder
	var reasoning strings.Builder
	var toolCalls []spec.ToolCall
	if response.OutputText != "" {
		content.WriteString(response.OutputText)
	}
	for _, output := range response.Output {
		if output.Role != "" {
			role = output.Role
		}
		switch output.Type {
		case "function_call":
			toolCalls = append(toolCalls, spec.ToolCall{
				ID:   output.CallID,
				Type: "function",
				Function: spec.FunctionCall{
					Name:      output.Name,
					Arguments: output.Arguments,
				},
			})
		case "reasoning":
			for _, part := range output.Summary {
				reasoning.WriteString(part.Text)
			}
		}
		for _, part := range output.Content {
			switch part.Type {
			case "output_text":
				if response.OutputText == "" {
					content.WriteString(part.Text)
				}
			case "refusal":
				refusal.WriteString(part.Refusal)
			}
		}
	}

	message := spec.Message{
		Role:             role,
		Content:          content.String(),
		Refusal:          refusal.String(),
		ReasoningContent: reasoning.String(),
		ToolCalls:        toolCalls,
	}
	if message.Content == "" {
		message.Content = message.Refusal
	}
	return &response, message, nil
}

func newResponsesResult(response *spec.ResponsesAPIResponse, message spec.Message, rawBody []byte) *spec.Response {
	return &spec.Response{
		Protocol:  spec.ProtocolResponses,
		ID:        response.ID,
		Model:     response.Model,
		Status:    response.Status,
		Message:   message,
		Usage:     response.Usage,
		Responses: response,
		RawResponse: append(
			[]byte(nil),
			rawBody...,
		),
	}
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
