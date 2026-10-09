package opencodego

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

func consumeMessagesStream(ctx context.Context, response *http.Response, config *spec.RequestConfig) (*spec.Response, error) {
	envelope := messageEnvelope{Type: "message", Role: spec.RoleAssistant}
	blocks := make(map[int]*contentBlock)
	order := make([]int, 0)
	arguments := make(map[int]string)
	usage := make(map[string]json.RawMessage)
	mergeUsage := func(raw json.RawMessage) error {
		if len(raw) == 0 || string(raw) == "null" {
			return nil
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return err
		}
		for key, value := range fields {
			usage[key] = value
		}
		return nil
	}
	started, stopped := false, false
	err := scanMessagesSSE(response, func(raw []byte) (bool, error) {
		var event struct {
			Type         string           `json:"type"`
			Index        int              `json:"index"`
			Message      *messageEnvelope `json:"message"`
			ContentBlock *contentBlock    `json:"content_block"`
			Delta        struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Thinking    string `json:"thinking"`
				Signature   string `json:"signature"`
				PartialJSON string `json:"partial_json"`
				StopReason  string `json:"stop_reason"`
			} `json:"delta"`
			Usage json.RawMessage `json:"usage"`
			Error *spec.APIError  `json:"error"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			return false, fmt.Errorf("opencode-go Messages: decode stream event: %w", err)
		}
		if config.EventCallback != nil {
			if err := config.EventCallback(ctx, spec.StreamEvent{Protocol: spec.ProtocolAnthropicMessages, Type: event.Type, Raw: append(json.RawMessage(nil), raw...)}); err != nil {
				return false, err
			}
		}
		switch event.Type {
		case "error":
			if event.Error != nil {
				return false, fmt.Errorf("opencode-go Messages stream error (%s): %s", event.Error.Type, event.Error.Message)
			}
			return false, fmt.Errorf("opencode-go Messages stream error: %s", raw)
		case "message_start":
			if event.Message == nil {
				return false, fmt.Errorf("opencode-go Messages: missing stream message")
			}
			envelope = *event.Message
			started = true
			if err := mergeUsage(envelope.Usage); err != nil {
				return false, err
			}
		case "content_block_start":
			if event.ContentBlock == nil {
				return false, fmt.Errorf("opencode-go Messages: missing content block")
			}
			if _, exists := blocks[event.Index]; exists {
				return false, fmt.Errorf("opencode-go Messages: duplicate content block %d", event.Index)
			}
			blocks[event.Index] = event.ContentBlock
			order = append(order, event.Index)
			if event.ContentBlock.Text != "" && config.StreamCallback != nil {
				if err := config.StreamCallback(ctx, event.ContentBlock.Text); err != nil {
					return false, err
				}
			}
			if event.ContentBlock.Thinking != "" && config.ReasoningCallback != nil {
				if err := config.ReasoningCallback(ctx, event.ContentBlock.Thinking); err != nil {
					return false, err
				}
			}
		case "content_block_delta":
			block := blocks[event.Index]
			if block == nil {
				return false, fmt.Errorf("opencode-go Messages: delta for unknown content block %d", event.Index)
			}
			switch event.Delta.Type {
			case "text_delta":
				block.Text += event.Delta.Text
				if config.StreamCallback != nil {
					if err := config.StreamCallback(ctx, event.Delta.Text); err != nil {
						return false, err
					}
				}
			case "thinking_delta":
				block.Thinking += event.Delta.Thinking
				if config.ReasoningCallback != nil {
					if err := config.ReasoningCallback(ctx, event.Delta.Thinking); err != nil {
						return false, err
					}
				}
			case "signature_delta":
				block.Signature += event.Delta.Signature
			case "input_json_delta":
				arguments[event.Index] += event.Delta.PartialJSON
			}
		case "message_delta":
			if event.Delta.StopReason != "" {
				envelope.StopReason = event.Delta.StopReason
			}
			if err := mergeUsage(event.Usage); err != nil {
				return false, err
			}
		case "message_stop":
			if !started {
				return false, fmt.Errorf("opencode-go Messages: message_stop received before message_start")
			}
			stopped = true
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return nil, err
	}
	if !stopped {
		return nil, fmt.Errorf("opencode-go Messages: stream ended before message_stop")
	}
	content := make([]contentBlock, 0, len(order))
	for _, index := range order {
		block := blocks[index]
		if data, exists := arguments[index]; exists {
			var object map[string]json.RawMessage
			if err := json.Unmarshal([]byte(data), &object); err != nil || object == nil {
				return nil, fmt.Errorf("opencode-go Messages: invalid streamed tool arguments")
			}
			block.Input = json.RawMessage(data)
		}
		content = append(content, *block)
	}
	envelope.Content, err = json.Marshal(content)
	if err != nil {
		return nil, err
	}
	if len(usage) > 0 {
		envelope.Usage, err = json.Marshal(usage)
		if err != nil {
			return nil, err
		}
	}
	raw, err := json.Marshal(envelope)
	if err != nil {
		return nil, err
	}
	return parseMessage(raw)
}

// SSE data can span multiple lines; process the final buffered event at EOF.
func scanMessagesSSE(response *http.Response, consume func([]byte) (bool, error)) error {
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var lines []string
	dispatch := func() (bool, error) {
		if len(lines) == 0 {
			return false, nil
		}
		data := strings.Join(lines, "\n")
		lines = lines[:0]
		if data == "[DONE]" {
			return true, nil
		}
		return consume([]byte(data))
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			done, err := dispatch()
			if err != nil || done {
				return err
			}
		} else if strings.HasPrefix(line, "data:") {
			lines = append(lines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("opencode-go Messages: read stream: %w", err)
	}
	_, err := dispatch()
	return err
}
