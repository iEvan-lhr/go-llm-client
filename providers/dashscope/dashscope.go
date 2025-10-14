package dashscope

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/ievan-lhr/go-llm-client/internal/requester"
	"github.com/ievan-lhr/go-llm-client/spec"
	"net/http"
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
