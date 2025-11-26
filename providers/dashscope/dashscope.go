package dashscope

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"github.com/ievan-lhr/go-llm-client/internal/requester"
	"github.com/ievan-lhr/go-llm-client/spec"
	"net/http"
	"strings"
)

// clientImpl 实现了 llm.Client
type clientImpl struct {
	requester *requester.Requester
	config    spec.ClientConfig // 使用通用的ClientConfig
}

// modelImpl 实现了 llm.Model
type modelImpl struct {
	client *clientImpl
	name   string
}

// NewClient 是创建Dashscope客户端的入口函数。
// 它接受一个或多个llm.ClientOption来配置客户端。
func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	// 1. 创建一个带有默认值的配置
	config := spec.NewClientConfig()
	config.APIURL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" // 设置默认URL

	// 2. 应用所有用户传入的选项，用户设置会覆盖默认值
	for _, opt := range opts {
		opt(config)
	}

	// 3. 校验必要的配置
	if config.APIKey == "" {
		return nil, fmt.Errorf("dashscope: API key is required, use llm.WithAPIKey() option")
	}

	// 4. 创建并返回客户端实例
	return &clientImpl{
		requester: &requester.Requester{
			HTTPClient: config.HTTPClient, // 使用配置好的HTTPClient
		},
		config: *config,
	}, nil
}

// Model 实现了 llm.Client 接口的方法
func (c *clientImpl) Model(name string) spec.Model {
	return &modelImpl{client: c, name: name}
}

// dashscopeChunk 定义了流式响应的数据结构
type dashscopeChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`

	// DashScope 特有的 Usage 字段（在最后一条消息中）
	Usage *struct {
		TotalTokens  int `json:"total_tokens"`
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// Chat 实现了 llm.Model 接口的方法
func (m *modelImpl) Chat(ctx context.Context, messages []spec.Message, opts ...spec.Option) (*spec.Response, error) {
	config := spec.NewRequestConfig()
	for _, opt := range opts {
		opt(config)
	}

	requestBody := config.Parameters
	if requestBody == nil {
		requestBody = make(map[string]any)
	}
	// 【适配逻辑】将通用的 Thinking 选项翻译为 provider 特定的参数
	if config.Thinking != nil {
		// 如果用户设置了Thinking选项，就将其转换为 "enable_thinking" 字段
		requestBody["enable_thinking"] = *config.Thinking
	}
	requestBody["model"] = m.name
	requestBody["messages"] = messages

	if config.Temperature != nil {
		requestBody["temperature"] = *config.Temperature
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)

	// ==================== 流式处理分支 ====================
	if config.Streaming {
		requestBody["stream"] = true
		// DashScope 协议要求：设置 stream_options 以获取 Token 消耗
		requestBody["stream_options"] = map[string]bool{"include_usage": true}

		// 使用 PostStream 获取 http.Response，而不是一次性读取 Body
		resp, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, headers, requestBody)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		// 准备变量以收集完整的响应
		var fullContent strings.Builder
		var role string = "assistant"

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			// SSE 格式通常以 "data: " 开头
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			dataStr := strings.TrimPrefix(line, "data: ")
			dataStr = strings.TrimSpace(dataStr)

			// 检查结束标记
			if dataStr == "[DONE]" {
				break
			}

			var chunk dashscopeChunk
			if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
				// 忽略解析错误的行，或者打印日志
				continue
			}

			// 提取 Delta Content
			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta
				if delta.Role != "" {
					role = delta.Role
				}
				if delta.Content != "" {
					fullContent.WriteString(delta.Content)
					// 触发用户回调
					if config.StreamCallback != nil {
						if err := config.StreamCallback(ctx, delta.Content); err != nil {
							return nil, err // 用户要求中断
						}
					}
				}
			}

			// 处理 Usage (如果有需求，可以在 Response 结构体中扩展 Usage 字段)
			// if chunk.Usage != nil { ... }
		}

		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("dashscope: stream scan error: %w", err)
		}

		// 流式结束后，返回一个聚合的 Response
		return &spec.Response{
			Message: spec.Message{
				Role:    spec.Role(role),
				Content: fullContent.String(),
			},
			// RawResponse 在流式模式下无法完整提供，故留空或仅存最后一段
		}, nil
	}

	// ==================== 非流式处理分支 (保持原样) ====================

	// 使用配置中的APIURL
	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, headers, requestBody)
	if err != nil {
		return nil, err
	}

	var apiResp struct {
		Choices []struct {
			Message spec.Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rawBody, &apiResp); err != nil {
		return nil, fmt.Errorf("dashscope: failed to unmarshal response: %w", err)
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
