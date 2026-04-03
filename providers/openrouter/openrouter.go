package openrouter

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

type clientImpl struct {
	requester *requester.Requester
	config    spec.ClientConfig
}

type modelImpl struct {
	client *clientImpl
	name   string
}

func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	config := spec.NewClientConfig()
	config.APIURL = "https://openrouter.ai/api/v1/chat/completions"

	for _, opt := range opts {
		opt(config)
	}

	if config.APIKey == "" {
		return nil, fmt.Errorf("openrouter provider: API key is required")
	}

	return &clientImpl{
		requester: &requester.Requester{
			HTTPClient: config.HTTPClient,
		},
		config: *config,
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

	requestBody := make(map[string]any)
	if config.Parameters != nil {
		for k, v := range config.Parameters {
			requestBody[k] = v
		}
	}

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

	// 【思考模式处理】OpenRouter 官方规范：传递 include_reasoning 从而分离思考内容
	if config.Thinking != nil {
		if *config.Thinking {
			requestBody["include_reasoning"] = true
		} else {
			requestBody["include_reasoning"] = false
		}
	} else {
		requestBody["include_reasoning"] = false
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)

	if referer, ok := requestBody["HTTP-Referer"].(string); ok {
		headers.Set("HTTP-Referer", referer)
		delete(requestBody, "HTTP-Referer")
	}
	if title, ok := requestBody["X-Title"].(string); ok {
		headers.Set("X-Title", title)
		delete(requestBody, "X-Title")
	}

	// ==================== 流式处理分支 ====================
	if config.Streaming {
		requestBody["stream"] = true

		resp, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, headers, requestBody)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		var fullContent strings.Builder
		var reasoningContent strings.Builder // 收集思考过程
		role := "assistant"

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if line == "" || strings.HasPrefix(line, ":") {
				continue
			}

			if !strings.HasPrefix(line, "data:") {
				continue
			}

			dataStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if dataStr == "[DONE]" {
				break
			}

			// 解析包含 OpenRouter 专属 reasoning 字段的 Delta
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content   string `json:"content"`
						Role      string `json:"role"`
						Reasoning string `json:"reasoning"` // 思考过程字段
					} `json:"delta"`
				} `json:"choices"`
			}

			if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
				continue
			}

			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta
				if delta.Role != "" {
					role = delta.Role
				}

				// 收集思考过程
				if delta.Reasoning != "" {
					reasoningContent.WriteString(delta.Reasoning)
				}

				// 收集正文并触发回调
				if delta.Content != "" {
					fullContent.WriteString(delta.Content)
					if config.StreamCallback != nil {
						if err := config.StreamCallback(ctx, delta.Content); err != nil {
							return nil, err
						}
					}
				}
			}
		}

		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("openrouter stream scan error: %w", err)
		}

		return &spec.Response{
			Message: spec.Message{
				Role:             spec.Role(role),
				Content:          fullContent.String(),
				ReasoningContent: reasoningContent.String(),
			},
		}, nil
	}

	// ==================== 非流式处理分支 ====================
	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, headers, requestBody)
	if err != nil {
		return nil, err
	}

	var apiResp struct {
		Choices []struct {
			Message struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				Reasoning string `json:"reasoning"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(rawBody, &apiResp); err != nil {
		return nil, fmt.Errorf("openrouter provider: failed to unmarshal response: %w", err)
	}

	var responseMessage spec.Message
	if len(apiResp.Choices) > 0 {
		msg := apiResp.Choices[0].Message
		responseMessage = spec.Message{
			Role:             spec.Role(msg.Role),
			Content:          msg.Content,
			ReasoningContent: msg.Reasoning,
		}
	}

	return &spec.Response{
		Message:     responseMessage,
		RawResponse: rawBody,
	}, nil
}
