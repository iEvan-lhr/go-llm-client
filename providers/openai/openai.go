package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/ievan-lhr/go-llm-client/internal/requester"
	"github.com/ievan-lhr/go-llm-client/spec"
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

// NewClient 是创建OpenAI兼容客户端的入口函数。
func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	// 1. 创建带有OpenAI默认值的配置
	config := spec.NewClientConfig()
	config.APIURL = "https://api.openai.com/v1/chat/completions" // OpenAI 官方默认URL

	// 2. 应用所有用户传入的选项，用户的设置会覆盖默认值
	for _, opt := range opts {
		opt(config)
	}

	// 3. 校验必要的配置
	if config.APIKey == "" {
		return nil, fmt.Errorf("openai provider: API key is required, use spec.WithAPIKey()")
	}

	// 4. 创建并返回客户端实例
	return &clientImpl{
		requester: &requester.Requester{
			HTTPClient: config.HTTPClient,
		},
		config: *config,
	}, nil
}

// Model 实现了 spec.Client 接口的方法
func (c *clientImpl) Model(name string) spec.Model {
	return &modelImpl{client: c, name: name}
}

// Chat 实现了 spec.Model 接口的方法
func (m *modelImpl) Chat(ctx context.Context, messages []spec.Message, opts ...spec.Option) (*spec.Response, error) {
	config := spec.NewRequestConfig()
	for _, opt := range opts {
		opt(config)
	}

	// 1. 基础请求体来自用户传入的任意参数
	requestBody := config.Parameters
	if requestBody == nil {
		requestBody = make(map[string]any)
	}

	// 2. 强制设置/覆盖核心及标准参数
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
	if config.Streaming {
		requestBody["stream"] = true
	}

	// 3. 准备请求头
	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)

	// 4. 调用通用 Requester
	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, headers, requestBody)
	if err != nil {
		return nil, err
	}

	// 5. 解析响应
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

	// 6. 返回通用响应
	return &spec.Response{
		Message:     responseMessage,
		RawResponse: rawBody,
	}, nil
}
