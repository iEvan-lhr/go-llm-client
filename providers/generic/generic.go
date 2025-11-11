package generic

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/ievan-lhr/go-llm-client/spec"
	"net/http"
	"regexp"

	"github.com/ievan-lhr/go-llm-client/internal/requester" // 请确保这是您正确的仓库路径
)

// clientImpl 实现了 llm.Client
type clientImpl struct {
	requester *requester.Requester
	config    spec.ClientConfig
}

// modelImpl 实现了 llm.Model
type modelImpl struct {
	client *clientImpl
	name   string
}

// thinkTagRegex 用于匹配并移除私有化Qwen模型返回内容中的<think>...</think>标签
var thinkTagRegex = regexp.MustCompile(`(?s)<think>.*?</think>\n\n`)

// NewClient 是创建通用（私有化）客户端的入口函数。
func NewClient(opts ...spec.ClientOption) (spec.Client, error) {
	config := spec.NewClientConfig()
	// 应用所有用户传入的选项
	for _, opt := range opts {
		opt(config)
	}

	// 校验必要的配置
	if config.APIKey == "" {
		return nil, fmt.Errorf("generic provider: API key is required, use llm.WithAPIKey()")
	}
	if config.APIURL == "" {
		return nil, fmt.Errorf("generic provider: API URL is required for private deployment, use llm.WithAPIURL()")
	}

	return &clientImpl{
		requester: &requester.Requester{
			HTTPClient: config.HTTPClient,
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
	// 为了不修改用户传入的原始messages切片，我们创建一个副本
	processedMessages := make([]spec.Message, len(messages))
	copy(processedMessages, messages)

	// 【适配逻辑】将通用的 Thinking 选项翻译为修改 system prompt 的行为
	if config.Thinking != nil && !*config.Thinking {
		// 如果用户明确要求关闭思考 (WithThinking(false))
		// 我们需要找到 system prompt 并在其末尾追加 "/no_think"
		//foundSystem := false
		for i, msg := range processedMessages {
			if msg.Role == spec.RoleSystem {
				messages[i].Content += "\n/no_think"
				//foundSystem = true
				break
			}
		}
	}

	requestBody := config.Parameters
	if requestBody == nil {
		requestBody = make(map[string]any)
	}

	// 强制设置核心参数
	requestBody["model"] = m.name // 这里的name将是 "/mnt/Qwen3-30B-A3B/"
	requestBody["messages"] = messages

	if config.Temperature != nil {
		requestBody["temperature"] = *config.Temperature
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	// 这里的APIKey就是完整的 "Bearer aieif=..." 字符串
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)

	// 调用通用 Requester
	rawBody, err := m.client.requester.Post(ctx, m.client.config.APIURL, headers, requestBody)
	if err != nil {
		return nil, err
	}

	// 解析响应
	var apiResp struct {
		Choices []struct {
			Message spec.Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rawBody, &apiResp); err != nil {
		return nil, fmt.Errorf("generic provider: failed to unmarshal response: %w", err)
	}

	if len(apiResp.Choices) == 0 {
		return nil, fmt.Errorf("generic provider: invalid response, no choices found")
	}

	responseMessage := apiResp.Choices[0].Message

	// 【核心适配】清理<think>...</think>标签
	responseMessage.Content = thinkTagRegex.ReplaceAllString(responseMessage.Content, "")

	return &spec.Response{
		Message:     responseMessage,
		RawResponse: rawBody,
	}, nil
}
