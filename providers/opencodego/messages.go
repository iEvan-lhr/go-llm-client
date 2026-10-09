package opencodego

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

type contentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Data      string          `json:"data,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
}

type messageEnvelope struct {
	Type       string          `json:"type"`
	ID         string          `json:"id"`
	Model      string          `json:"model"`
	Role       spec.Role       `json:"role"`
	Content    json.RawMessage `json:"content"`
	StopReason string          `json:"stop_reason,omitempty"`
	Usage      json.RawMessage `json:"usage,omitempty"`
}

type wireMessage struct {
	Role    spec.Role `json:"role"`
	Content []any     `json:"content"`
}

func (m *modelImpl) messages(ctx context.Context, endpoint, name string, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	body, err := messagesRequest(name, messages, config)
	if err != nil {
		return nil, err
	}
	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)
	headers.Set("x-api-key", m.client.config.APIKey)
	headers.Set("anthropic-version", "2023-06-01")
	if config.Streaming {
		response, err := m.client.requester.PostStream(ctx, endpoint, headers, body)
		if err != nil {
			return nil, err
		}
		defer response.Body.Close()
		return consumeMessagesStream(ctx, response, config)
	}
	raw, err := m.client.requester.Post(ctx, endpoint, headers, body)
	if err != nil {
		return nil, err
	}
	return parseMessage(raw)
}

func messagesRequest(name string, messages []spec.Message, config *spec.RequestConfig) (map[string]any, error) {
	if config.ResponseInput != nil || config.PreviousResponseID != "" {
		return nil, fmt.Errorf("opencode-go Messages: Responses input and previous_response_id require a Responses model")
	}
	if config.WebSearch != nil {
		return nil, fmt.Errorf("opencode-go Messages: configure native search tools through Parameters[\"tools\"]")
	}
	body := config.Parameters // Already copied by Chat.
	wireMessages := make([]wireMessage, 0, len(messages))
	var system []any
	for _, message := range messages {
		blocks, err := messageBlocks(message)
		if err != nil {
			return nil, err
		}
		if message.Role == spec.RoleSystem || message.Role == spec.RoleDeveloper {
			system = append(system, blocks...)
			continue
		}
		role := message.Role
		if role == spec.RoleTool {
			role = spec.RoleUser
			if message.ToolCallID == "" {
				return nil, fmt.Errorf("opencode-go Messages: a tool result requires ToolCallID")
			}
			blocks = []any{map[string]any{"type": "tool_result", "tool_use_id": message.ToolCallID, "content": blocks}}
		}
		if role != spec.RoleUser && role != spec.RoleAssistant {
			return nil, fmt.Errorf("opencode-go Messages: unsupported message role %q", role)
		}
		if len(wireMessages) > 0 && wireMessages[len(wireMessages)-1].Role == role {
			wireMessages[len(wireMessages)-1].Content = append(wireMessages[len(wireMessages)-1].Content, blocks...)
		} else {
			wireMessages = append(wireMessages, wireMessage{Role: role, Content: blocks})
		}
	}
	body["model"], body["messages"], body["stream"] = name, wireMessages, config.Streaming
	if len(system) > 0 {
		body["system"] = system
	}
	if config.Instructions != nil {
		body["system"] = config.Instructions
	}
	if config.MaxTokens != nil {
		body["max_tokens"] = *config.MaxTokens
	} else if _, exists := body["max_tokens"]; !exists {
		body["max_tokens"] = 4096
	}
	if config.Temperature != nil {
		body["temperature"] = *config.Temperature
	}
	if config.TopP != nil {
		body["top_p"] = *config.TopP
	}
	if config.Thinking != nil {
		if !*config.Thinking {
			body["thinking"] = map[string]any{"type": "disabled"}
		} else {
			thinking, _ := body["thinking"].(map[string]any)
			if thinking == nil {
				thinking = map[string]any{"type": "enabled"}
			}
			if thinking["type"] != "adaptive" {
				thinking["type"] = "enabled"
				if _, exists := thinking["budget_tokens"]; !exists {
					thinking["budget_tokens"] = 1024
				}
			}
			body["thinking"] = thinking
		}
	}
	if config.ReasoningEffort == spec.ReasoningEffortNone {
		body["thinking"] = map[string]any{"type": "disabled"}
	} else if config.ReasoningEffort != "" {
		output, _ := body["output_config"].(map[string]any)
		if output == nil {
			output = make(map[string]any)
		}
		output["effort"] = config.ReasoningEffort
		body["output_config"] = output
	}
	if tools, ok := body["tools"]; ok {
		data, err := json.Marshal(tools)
		if err != nil {
			return nil, err
		}
		var converted []map[string]any
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.UseNumber()
		if err := decoder.Decode(&converted); err != nil {
			return nil, fmt.Errorf("opencode-go Messages: invalid tools: %w", err)
		}
		for i, tool := range converted {
			if tool["type"] != "function" {
				continue // Native Anthropic tools pass through unchanged.
			}
			function, ok := tool["function"].(map[string]any)
			if !ok {
				return nil, fmt.Errorf("opencode-go Messages: function tool requires a function object")
			}
			converted[i] = map[string]any{"name": function["name"], "input_schema": function["parameters"]}
			if converted[i]["input_schema"] == nil {
				converted[i]["input_schema"] = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			if description, ok := function["description"]; ok {
				converted[i]["description"] = description
			}
		}
		body["tools"] = converted
	}
	if choice, ok := body["tool_choice"]; ok {
		switch value := choice.(type) {
		case string:
			if value == "required" {
				value = "any"
			}
			body["tool_choice"] = map[string]any{"type": value}
		case map[string]any:
			if value["type"] == "function" {
				function, ok := value["function"].(map[string]any)
				if !ok {
					return nil, fmt.Errorf("opencode-go Messages: invalid function tool_choice")
				}
				body["tool_choice"] = map[string]any{"type": "tool", "name": function["name"]}
			}
		}
	}
	return body, nil
}

func messageBlocks(message spec.Message) ([]any, error) {
	if message.Role == spec.RoleAssistant && len(message.AnthropicContent) > 0 {
		var native []json.RawMessage
		if err := json.Unmarshal(message.AnthropicContent, &native); err != nil {
			return nil, fmt.Errorf("opencode-go Messages: invalid native content: %w", err)
		}
		blocks := make([]any, len(native))
		for i, block := range native {
			blocks[i] = block
		}
		return blocks, nil
	}
	blocks := make([]any, 0, len(message.Parts)+len(message.ToolCalls)+1)
	if len(message.Parts) == 0 && message.Content != "" {
		blocks = append(blocks, map[string]any{"type": "text", "text": message.Content})
	}
	for _, part := range message.Parts {
		switch part.Type {
		case "text", "input_text":
			blocks = append(blocks, map[string]any{"type": "text", "text": part.Text})
		case "image_url", "input_image":
			if part.ImageURL == nil {
				return nil, fmt.Errorf("opencode-go Messages: image content requires an image URL")
			}
			source, err := contentSource(part.ImageURL.URL)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, map[string]any{"type": "image", "source": source})
		case "file", "input_file":
			location := part.FileURL
			if part.FileData != "" {
				location = part.FileData
			}
			if location == "" {
				return nil, fmt.Errorf("opencode-go Messages: documents require a URL or base64 data URI")
			}
			source, err := contentSource(location)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, map[string]any{"type": "document", "source": source})
		default:
			return nil, fmt.Errorf("opencode-go Messages: unsupported content type %q", part.Type)
		}
	}
	for _, call := range message.ToolCalls {
		arguments := call.Function.Arguments
		if arguments == "" {
			arguments = "{}"
		}
		var input map[string]json.RawMessage
		if err := json.Unmarshal([]byte(arguments), &input); err != nil || input == nil {
			return nil, fmt.Errorf("opencode-go Messages: tool arguments must be a JSON object")
		}
		blocks = append(blocks, map[string]any{"type": "tool_use", "id": call.ID, "name": call.Function.Name, "input": json.RawMessage(arguments)})
	}
	// Empty tool results are valid and must still have a content block.
	if len(blocks) == 0 {
		blocks = append(blocks, map[string]any{"type": "text", "text": ""})
	}
	return blocks, nil
}

func contentSource(location string) (map[string]any, error) {
	if !strings.HasPrefix(location, "data:") {
		return map[string]any{"type": "url", "url": location}, nil
	}
	metadata, data, ok := strings.Cut(strings.TrimPrefix(location, "data:"), ",")
	if !ok || !strings.HasSuffix(metadata, ";base64") {
		return nil, fmt.Errorf("opencode-go Messages: content data URI must use base64 encoding")
	}
	return map[string]any{"type": "base64", "media_type": strings.TrimSuffix(metadata, ";base64"), "data": data}, nil
}

func parseMessage(raw []byte) (*spec.Response, error) {
	var envelope messageEnvelope
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, fmt.Errorf("opencode-go Messages: decode response: %w", err)
	}
	if envelope.Type != "message" {
		return nil, fmt.Errorf("opencode-go Messages: unexpected response: %s", raw)
	}
	var blocks []contentBlock
	if err := json.Unmarshal(envelope.Content, &blocks); err != nil {
		return nil, fmt.Errorf("opencode-go Messages: decode content: %w", err)
	}
	message := spec.Message{Role: envelope.Role, AnthropicContent: append(json.RawMessage(nil), envelope.Content...)}
	if message.Role == "" {
		message.Role = spec.RoleAssistant
	}
	for _, block := range blocks {
		switch block.Type {
		case "text":
			message.Content += block.Text
		case "thinking":
			message.ReasoningContent += block.Thinking
		case "tool_use":
			arguments := string(block.Input)
			if arguments == "" {
				arguments = "{}"
			}
			message.ToolCalls = append(message.ToolCalls, spec.ToolCall{ID: block.ID, Type: "function", Function: spec.FunctionCall{Name: block.Name, Arguments: arguments}})
		}
	}
	return &spec.Response{Protocol: spec.ProtocolAnthropicMessages, ID: envelope.ID, Model: envelope.Model, Status: envelope.StopReason,
		Message: message, Usage: messagesUsage(envelope.Usage), RawResponse: raw}, nil
}

func messagesUsage(raw json.RawMessage) *spec.Usage {
	usage := spec.NormalizeUsage(raw)
	if usage == nil {
		return nil
	}
	// Anthropic reports uncached input separately from cache reads and writes.
	if details := usage.PromptTokensDetails; details != nil {
		usage.InputTokens += details.CachedTokens + details.CacheWriteTokens
		copy := *details
		usage.InputTokensDetails = &copy
	}
	usage.PromptTokens = usage.InputTokens
	usage.CompletionTokens = usage.OutputTokens
	usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	return usage
}
