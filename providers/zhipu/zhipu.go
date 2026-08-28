// Package zhipu implements the ZHIPU AI Chat Completions API.
package zhipu

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/internal/requester"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

const defaultAPIURL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

type clientImpl struct {
	requester *requester.Requester
	config    spec.ClientConfig
}

type modelImpl struct {
	client *clientImpl
	name   string
}

// NewClient creates a ZHIPU AI client. The official Chat Completions endpoint
// is used unless it is overridden with spec.WithAPIURL.
func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	config := spec.NewClientConfig()
	config.APIURL = defaultAPIURL
	for _, opt := range opts {
		opt(config)
	}

	if config.APIKey == "" {
		return nil, fmt.Errorf("zhipu provider: API key is required, use spec.WithAPIKey()")
	}

	return &clientImpl{
		requester: &requester.Requester{HTTPClient: config.HTTPClient},
		config:    *config,
	}, nil
}

func (c *clientImpl) Model(name string) spec.Model {
	return &modelImpl{client: c, name: name}
}

func (m *modelImpl) Chat(ctx context.Context, messages []spec.Message, opts ...spec.Option) (*spec.Response, error) {
	config := spec.NewRequestConfig()
	for _, opt := range opts {
		opt(config)
	}

	requestBody := cloneParameters(config.Parameters)
	requestBody["model"] = m.name
	wireMessages, err := zhipuMessages(messages)
	if err != nil {
		return nil, err
	}
	requestBody["messages"] = wireMessages

	if config.Temperature != nil {
		requestBody["temperature"] = *config.Temperature
	}
	if config.MaxTokens != nil {
		requestBody["max_tokens"] = *config.MaxTokens
	}
	if config.TopP != nil {
		requestBody["top_p"] = *config.TopP
	}
	if config.Thinking != nil {
		thinking := map[string]any{}
		switch value := requestBody["thinking"].(type) {
		case map[string]any:
			for key, item := range value {
				thinking[key] = item
			}
		case map[string]string:
			for key, item := range value {
				thinking[key] = item
			}
		}
		thinkingType := "disabled"
		if *config.Thinking {
			thinkingType = "enabled"
		}
		thinking["type"] = thinkingType
		requestBody["thinking"] = thinking
	}
	if config.ReasoningEffort != "" {
		requestBody["reasoning_effort"] = config.ReasoningEffort
	}
	if config.WebSearch != nil {
		if err := applyHostedWebSearch(requestBody, *config.WebSearch); err != nil {
			return nil, err
		}
	}

	if config.Streaming {
		requestBody["stream"] = true
		return m.stream(ctx, requestBody, config)
	}

	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}

	var envelope struct {
		Error *spec.APIError `json:"error"`
	}
	if err := json.Unmarshal(rawBody, &envelope); err == nil && envelope.Error != nil {
		return nil, apiError(envelope.Error)
	}

	var completion spec.ChatCompletionResponse
	if err := json.Unmarshal(rawBody, &completion); err != nil {
		return nil, fmt.Errorf("zhipu provider: failed to unmarshal response: %w", err)
	}
	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("zhipu provider: invalid response, no choices found")
	}
	if completion.ID == "" {
		var metadata struct {
			RequestID string `json:"request_id"`
		}
		if err := json.Unmarshal(rawBody, &metadata); err == nil {
			completion.ID = metadata.RequestID
		}
	}

	message := completion.Choices[0].Message
	if message.Content == "" {
		message.Content = message.PlainText()
	}
	completion.Choices[0].Message = message
	if completion.Model == "" {
		completion.Model = m.name
	}
	return newResult(&completion, message, rawBody), nil
}

func (m *modelImpl) stream(ctx context.Context, requestBody map[string]any, config *spec.RequestConfig) (*spec.Response, error) {
	response, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, m.headers(), requestBody)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	var completion spec.ChatCompletionResponse
	var content strings.Builder
	var reasoning strings.Builder
	var webSearchCalls []spec.WebSearchCall
	var citations []spec.URLCitation
	role := spec.RoleAssistant
	toolCalls := make(map[int]*spec.ToolCall)
	toolOrder := make([]int, 0)

	err = scanSSE(response, func(data string) error {
		if data == "[DONE]" {
			return nil
		}

		raw := json.RawMessage(append([]byte(nil), data...))
		eventType := "chat.completion.chunk"
		var eventEnvelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &eventEnvelope); err == nil && eventEnvelope.Type != "" {
			eventType = eventEnvelope.Type
		}
		if config.EventCallback != nil {
			if err := config.EventCallback(ctx, spec.StreamEvent{
				Protocol: spec.ProtocolChatCompletions,
				Type:     eventType,
				Raw:      raw,
			}); err != nil {
				return err
			}
		}

		var envelope struct {
			Error *spec.APIError `json:"error"`
		}
		if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error != nil {
			return apiError(envelope.Error)
		}

		var chunk spec.ChatCompletionResponse
		if err := json.Unmarshal(raw, &chunk); err != nil {
			return fmt.Errorf("zhipu provider: failed to decode stream chunk: %w", err)
		}
		mergeCompletionMetadata(&completion, &chunk)
		if calls, chunkCitations := zhipuSearchMetadata(raw); len(calls) > 0 {
			webSearchCalls = calls
			citations = chunkCitations
		}

		for _, choice := range chunk.Choices {
			if choice.Index != 0 {
				continue
			}
			delta := choice.Delta
			if delta.Role != "" {
				role = delta.Role
			}
			if delta.ReasoningContent != "" {
				reasoning.WriteString(delta.ReasoningContent)
				if config.ReasoningCallback != nil {
					if err := config.ReasoningCallback(ctx, delta.ReasoningContent); err != nil {
						return err
					}
				}
			}
			text := delta.PlainText()
			if text != "" {
				content.WriteString(text)
				if config.StreamCallback != nil {
					if err := config.StreamCallback(ctx, text); err != nil {
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
		ReasoningContent: reasoning.String(),
		ToolCalls:        orderedToolCalls(toolCalls, toolOrder),
	}
	if len(completion.Choices) == 0 {
		completion.Choices = []spec.ChatChoice{{Index: 0, Message: message}}
	} else {
		completion.Choices[0].Message = message
	}
	if completion.Model == "" {
		completion.Model = m.name
	}
	rawBody, _ := json.Marshal(completion)
	result := newResult(&completion, message, rawBody)
	if len(webSearchCalls) > 0 {
		result.WebSearchCalls = webSearchCalls
		result.Citations = citations
	}
	return result, nil
}

func (m *modelImpl) headers() http.Header {
	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)
	return headers
}

func cloneParameters(parameters map[string]any) map[string]any {
	cloned := make(map[string]any, len(parameters)+6)
	for key, value := range parameters {
		cloned[key] = value
	}
	return cloned
}

// zhipuMessages translates the library's provider-neutral multimodal parts to
// the names used by ZHIPU AI. Text, image, video, file, and audio inputs are
// all represented as Chat Completions content items.
func zhipuMessages(messages []spec.Message) ([]any, error) {
	result := make([]any, 0, len(messages))
	for messageIndex, message := range messages {
		role := message.Role
		// ZHIPU currently has no separate developer role. Treat it as a system
		// instruction so callers can reuse provider-neutral histories.
		if role == spec.RoleDeveloper {
			role = spec.RoleSystem
		}

		if len(message.Parts) == 0 {
			clone := message
			clone.Role = role
			result = append(result, &clone)
			continue
		}

		parts := make([]any, 0, len(message.Parts))
		for partIndex, part := range message.Parts {
			switch part.Type {
			case "text", "input_text":
				parts = append(parts, map[string]any{"type": "text", "text": part.Text})
			case "image_url", "input_image":
				if part.ImageURL == nil || part.ImageURL.URL == "" {
					return nil, fmt.Errorf("zhipu provider: message %d content part %d requires an image URL", messageIndex, partIndex)
				}
				parts = append(parts, map[string]any{"type": "image_url", "image_url": part.ImageURL})
			case "video_url", "input_video":
				if part.VideoURL == nil || part.VideoURL.URL == "" {
					return nil, fmt.Errorf("zhipu provider: message %d content part %d requires a video URL", messageIndex, partIndex)
				}
				parts = append(parts, map[string]any{"type": "video_url", "video_url": part.VideoURL})
			case "file_url", "file", "input_file":
				if part.FileURL == "" {
					return nil, fmt.Errorf("zhipu provider: message %d content part %d requires a file URL; inline file data and file IDs are not supported", messageIndex, partIndex)
				}
				parts = append(parts, map[string]any{
					"type":     "file_url",
					"file_url": map[string]any{"url": part.FileURL},
				})
			case "input_audio", "audio":
				if part.InputAudio == nil || part.InputAudio.Data == "" || part.InputAudio.Format == "" {
					return nil, fmt.Errorf("zhipu provider: message %d content part %d requires audio data and format", messageIndex, partIndex)
				}
				parts = append(parts, map[string]any{"type": "input_audio", "input_audio": part.InputAudio})
			default:
				parts = append(parts, part)
			}
		}

		wire := map[string]any{"role": role, "content": parts}
		if message.Name != "" {
			wire["name"] = message.Name
		}
		if message.ToolCallID != "" {
			wire["tool_call_id"] = message.ToolCallID
		}
		if len(message.ToolCalls) > 0 {
			wire["tool_calls"] = message.ToolCalls
		}
		if message.ReasoningContent != "" {
			wire["reasoning_content"] = message.ReasoningContent
		}
		result = append(result, wire)
	}
	return result, nil
}

func newResult(completion *spec.ChatCompletionResponse, message spec.Message, rawBody []byte) *spec.Response {
	webSearchCalls, citations := zhipuSearchMetadata(rawBody)
	return &spec.Response{
		Protocol:       spec.ProtocolChatCompletions,
		ID:             completion.ID,
		Model:          completion.Model,
		Status:         finishReason(completion.Choices),
		Message:        message,
		Usage:          completion.Usage,
		ChatCompletion: completion,
		WebSearchCalls: webSearchCalls,
		Citations:      citations,
		RawResponse:    append([]byte(nil), rawBody...),
	}
}

func finishReason(choices []spec.ChatChoice) string {
	if len(choices) == 0 || choices[0].FinishReason == nil {
		return ""
	}
	return *choices[0].FinishReason
}

func mergeCompletionMetadata(target, chunk *spec.ChatCompletionResponse) {
	if target.ID == "" {
		target.ID = chunk.ID
		target.Object = chunk.Object
		target.Created = chunk.Created
		target.Model = chunk.Model
	}
	if chunk.Usage != nil {
		target.Usage = chunk.Usage
	}
	for _, choice := range chunk.Choices {
		for len(target.Choices) <= choice.Index {
			target.Choices = append(target.Choices, spec.ChatChoice{Index: len(target.Choices)})
		}
		if choice.FinishReason != nil {
			target.Choices[choice.Index].FinishReason = choice.FinishReason
		}
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
		call.Function.Name += delta.Function.Name
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

func scanSSE(response *http.Response, handleData func(string) error) error {
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)

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
		return fmt.Errorf("zhipu provider: stream read failed: %w", err)
	}
	return dispatch()
}

func apiError(apiErr *spec.APIError) error {
	if apiErr.Code != "" {
		return fmt.Errorf("zhipu provider: API error (%s): %s", apiErr.Code, apiErr.Message)
	}
	return fmt.Errorf("zhipu provider: API error: %s", apiErr.Message)
}

var _ spec.Client = (*clientImpl)(nil)
var _ spec.Model = (*modelImpl)(nil)
