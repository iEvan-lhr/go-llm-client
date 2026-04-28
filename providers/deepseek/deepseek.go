package deepseek

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

// clientImpl 实现了 spec.Client
type clientImpl struct {
	requester *requester.Requester
	config    spec.ClientConfig
}

// modelImpl 实现了 spec.Model
type modelImpl struct {
	client *clientImpl
	name   string
}

// NewClient 创建 DeepSeek 客户端，使用最新 V4 API 官方端点。
func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	config := spec.NewClientConfig()
	// 官方最新 Base URL: https://api.deepseek.com
	// 完整端点: https://api.deepseek.com/chat/completions
	config.APIURL = "https://api.deepseek.com/chat/completions"

	for _, opt := range opts {
		opt(config)
	}

	if config.APIKey == "" {
		return nil, fmt.Errorf("deepseek provider: API key is required")
	}

	return &clientImpl{
		requester: &requester.Requester{
			HTTPClient: config.HTTPClient,
		},
		config: *config,
	}, nil
}

// Model 返回一个实现了 spec.Model 的模型实例。
func (c *clientImpl) Model(name string) spec.Model {
	return &modelImpl{client: c, name: name}
}

// Chat 执行一次对话调用，完全适配 DeepSeek V4 API 规范。
func (m *modelImpl) Chat(ctx context.Context, messages []spec.Message, opts ...spec.Option) (*spec.Response, error) {
	config := spec.NewRequestConfig()
	for _, opt := range opts {
		opt(config)
	}

	// 1. 构建请求体，从 Parameters 初始化以支持透传
	requestBody := make(map[string]any)
	if config.Parameters != nil {
		for k, v := range config.Parameters {
			requestBody[k] = v
		}
	}

	// 2. 设置核心必选参数
	requestBody["model"] = m.name
	requestBody["messages"] = messages

	// 3. 设置通用 OpenAI 兼容参数
	if config.Temperature != nil {
		requestBody["temperature"] = *config.Temperature
	}
	if config.MaxTokens != nil {
		requestBody["max_tokens"] = *config.MaxTokens
	}
	if config.TopP != nil {
		requestBody["top_p"] = *config.TopP
	}
	if config.Streaming {
		requestBody["stream"] = true
	}

	// 4. 【关键适配】根据 Thinking 选项构造 reasoning_effort 参数
	// 这是 V4 API 控制推理强度的标准方式。
	// 用户可以通过 Parameters 透传 thinking 对象和 reasoning_effort 字段进行更精细的控制。
	if config.Thinking != nil {
		// 根据用户设置决定推理强度
		reasoningEffort := "high" // 默认值，对应标准推理
		if !*config.Thinking {
			requestBody["thinking"] = map[string]string{"type": "disabled"}
		} else {
			thinkingObj := map[string]string{"type": "enabled"}
			// 允许用户通过 Parameters 覆盖 reasoning_effort
			if _, ok := requestBody["reasoning_effort"]; !ok {
				requestBody["reasoning_effort"] = reasoningEffort
			}
			requestBody["thinking"] = thinkingObj
		}
	} else {
		requestBody["thinking"] = map[string]string{"type": "disabled"}
	}

	// 5. 【新增】支持 frequency_penalty 和 presence_penalty，可通过 Parameters 透传

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)

	// ==================== 流式处理分支 ====================
	if config.Streaming {
		resp, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, headers, requestBody)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		var fullContent strings.Builder
		var reasoningContent strings.Builder
		role := "assistant"

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			dataStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if dataStr == "[DONE]" {
				break
			}

			var chunk struct {
				Choices []struct {
					Delta struct {
						Content          string `json:"content"`
						Role             string `json:"role"`
						ReasoningContent string `json:"reasoning_content"`
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
				if delta.ReasoningContent != "" {
					reasoningContent.WriteString(delta.ReasoningContent)
				}
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
			return nil, fmt.Errorf("deepseek stream scan error: %w", err)
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
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(rawBody, &apiResp); err != nil {
		return nil, fmt.Errorf("deepseek provider: failed to unmarshal response: %w", err)
	}

	var responseMessage spec.Message
	if len(apiResp.Choices) > 0 {
		msg := apiResp.Choices[0].Message
		responseMessage = spec.Message{
			Role:             spec.Role(msg.Role),
			Content:          msg.Content,
			ReasoningContent: msg.ReasoningContent,
		}
	}

	return &spec.Response{
		Message:     responseMessage,
		RawResponse: rawBody,
	}, nil
}
