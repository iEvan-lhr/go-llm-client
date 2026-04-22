package dashscope

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/iEvan-lhr/go-llm-client/internal/requester"
	"github.com/iEvan-lhr/go-llm-client/spec"
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
			// 支持获取思考过程内容（OpenAI 兼容格式下的 reasoning_content）
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`

	// 适配 Responses API 下的事件类型和字段
	Type string `json:"type,omitempty"`
	Item *struct {
		Type   string `json:"type,omitempty"`
		Goal   string `json:"goal,omitempty"`
		Output string `json:"output,omitempty"`
	} `json:"item,omitempty"`
	Delta string `json:"delta,omitempty"`

	// Chat Completions API 用法统计
	Usage *struct {
		TotalTokens  int `json:"total_tokens"`
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		XTools       map[string]struct {
			Count int `json:"count"`
		} `json:"x_tools,omitempty"`
	} `json:"usage"`

	// Responses API 完成时的用法统计
	Response *struct {
		Usage *struct {
			TotalTokens  int `json:"total_tokens"`
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			XTools       map[string]struct {
				Count int `json:"count"`
			} `json:"x_tools,omitempty"`
		} `json:"usage"`
	} `json:"response,omitempty"`
}

// Chat 实现了 llm.Model 接口的方法
func (m *modelImpl) Chat(ctx context.Context, messages []spec.Message, opts ...spec.Option) (*spec.Response, error) {
	config := spec.NewRequestConfig()
	for _, opt := range opts {
		opt(config)
	}

	switch {
	case config.IsText2Image():
		return m.handleText2Image(ctx, messages, config)

	case config.IsImageEdit():
		// 预留 ImageEdit 实现位置（根据 DashScope API 文档后续扩展）
		return nil, fmt.Errorf("image edit is not supported yet for model %s", m.name)

	default:
		return m.handleChat(ctx, messages, config)
	}
}

// handleText2Image 处理文生图同步调用流程（qwen-image-2.0-pro）
func (m *modelImpl) handleText2Image(ctx context.Context, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	// 1. 提取 Prompt（取最后一条用户消息）
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided for text2image")
	}
	prompt := messages[len(messages)-1].Content
	if prompt == "" {
		return nil, fmt.Errorf("empty prompt for text2image")
	}

	// 2. 构建请求体（qwen-image-2.0-pro 要求 input.messages 格式）
	requestBody := map[string]any{
		"model": m.name,
		"input": map[string]any{
			"messages": []map[string]any{
				{
					"role": "user",
					"content": []map[string]any{
						{
							"text": prompt,
						},
					},
				},
			},
		},
		"parameters": map[string]any{
			"size":          "1664*928", // qwen-image-plus 默认分辨率
			"n":             1,
			"prompt_extend": true,
			"watermark":     false,
		},
	}

	// 从 config.Parameters 提取用户自定义参数
	if config.Parameters != nil {
		// negative_prompt 放在 parameters 中，不是 input 中
		if negPrompt, ok := config.Parameters["negative_prompt"]; ok {
			if paramMap, ok := requestBody["parameters"].(map[string]any); ok {
				paramMap["negative_prompt"] = negPrompt
			}
		}
		// 其他参数覆盖默认值
		for _, key := range []string{"size", "n", "prompt_extend", "watermark"} {
			if val, ok := config.Parameters[key]; ok {
				if paramMap, ok := requestBody["parameters"].(map[string]any); ok {
					paramMap[key] = val
				}
			}
		}
	}

	// 3. 构建请求头（同步调用，无需异步头）
	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)

	// 4. 发起请求（使用 multimodal-generation 端点）
	generationURL := "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
	if strings.Contains(m.client.config.APIURL, "dashscope-intl") {
		generationURL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
	}

	rawBody, err := m.client.requester.Post(ctx, generationURL, headers, requestBody)
	if err != nil {
		return nil, fmt.Errorf("dashscope qwen-image generation failed: %w", err)
	}

	// 5. 解析响应（同步返回，无需轮询任务）
	// 响应格式：output.choices[0].message.content[0].image
	var genResp struct {
		Output struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Message      struct {
					Role    string `json:"role"`
					Content []struct {
						Image string `json:"image"`
						Text  string `json:"text"`
					} `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		} `json:"output"`
		Usage struct {
			ImageCount int `json:"image_count"`
			Width      int `json:"width"`
			Height     int `json:"height"`
		} `json:"usage"`
		RequestId string `json:"request_id"`
		Code      string `json:"code"`
		Message   string `json:"message"`
	}

	if err := json.Unmarshal(rawBody, &genResp); err != nil {
		return nil, fmt.Errorf("dashscope failed to parse response: %w, response: %s", err, string(rawBody))
	}

	// 检查错误响应（优先检查 code 字段）
	if genResp.Code != "" {
		return nil, fmt.Errorf("dashscope generation failed (code: %s): %s", genResp.Code, genResp.Message)
	}

	// 提取图像 URL
	if len(genResp.Output.Choices) == 0 {
		return nil, fmt.Errorf("no choices in generation response: %s", string(rawBody))
	}

	content := genResp.Output.Choices[0].Message.Content
	if len(content) == 0 {
		return nil, fmt.Errorf("empty content in generation response")
	}

	var imageURL string
	for _, c := range content {
		if c.Image != "" {
			imageURL = c.Image
			break
		}
	}

	if imageURL == "" {
		return nil, fmt.Errorf("no image URL in generation response: %s", string(rawBody))
	}

	return &spec.Response{
		Message: spec.Message{
			Role:    spec.RoleAssistant,
			Content: imageURL,
		},
		RawResponse: rawBody,
	}, nil
}

// handleChat 处理标准聊天请求（流式/非流式）
func (m *modelImpl) handleChat(ctx context.Context, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	requestBody := make(map[string]any)
	if config.Parameters != nil {
		for k, v := range config.Parameters {
			requestBody[k] = v
		}
	}

	if config.Thinking != nil {
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
		requestBody["stream_options"] = map[string]bool{"include_usage": true}

		resp, err := m.client.requester.PostStream(ctx, m.client.config.APIURL, headers, requestBody)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		var fullContent strings.Builder
		role := "assistant"

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data:") {
				continue
			}

			dataStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if dataStr == "[DONE]" {
				break
			}

			var chunk dashscopeChunk
			if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
				continue
			}

			// 拦截输出：Responses API 的中间工具抓取过程
			if chunk.Type == "response.output_item.done" && chunk.Item != nil && chunk.Item.Type == "web_extractor_call" {
				log.Printf("\n[Web Extractor Action] Goal: %s\nOutput: %s\n", chunk.Item.Goal, chunk.Item.Output)
			}

			var contentToAppend string

			// 解析 Chat Completions API 格式
			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta
				if delta.Role != "" {
					role = delta.Role
				}
				// 对于 qwen3-max，它的思考过程会从这里下发
				if delta.ReasoningContent != "" {
					contentToAppend += delta.ReasoningContent
				}
				if delta.Content != "" {
					contentToAppend += delta.Content
				}
			} else if chunk.Type == "response.output_text.delta" || chunk.Type == "response.reasoning_summary_text.delta" {
				// 解析 Responses API 格式
				contentToAppend = chunk.Delta
			}

			// 如果存在内容，写入 Builder 并触发 Callback
			if contentToAppend != "" {
				fullContent.WriteString(contentToAppend)
				if config.StreamCallback != nil {
					if err := config.StreamCallback(ctx, contentToAppend); err != nil {
						return nil, err
					}
				}
			}

			// 拦截输出：打印工具调用次数
			if chunk.Type == "response.completed" && chunk.Response != nil && chunk.Response.Usage != nil {
				if len(chunk.Response.Usage.XTools) > 0 {
					log.Printf("\n[Usage Stats] Tools: %+v", chunk.Response.Usage.XTools)
				}
			} else if chunk.Usage != nil && len(chunk.Usage.XTools) > 0 {
				log.Printf("\n[Usage Stats] Tools: %+v", chunk.Usage.XTools)
			}
		}

		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("dashscope: stream scan error: %w", err)
		}

		return &spec.Response{
			Message: spec.Message{
				Role:    spec.Role(role),
				Content: fullContent.String(),
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

// Get 发起 HTTP GET 请求，返回原始响应体字节
// 适用于轮询异步任务状态等场景
func (m *modelImpl) Get(ctx context.Context, url string, headers http.Header) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("dashscope: failed to create GET request: %w", err)
	}

	// 设置请求头
	for key, values := range headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	// 【修复 3】防止 m.client.requester.HTTPClient 为空时发生 panic
	client := m.client.requester.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	// 发起请求
	resp, err := client.Do(req)
	if err != nil {
		// 区分上下文取消/超时 与 网络错误
		if ctx.Err() != nil {
			return nil, fmt.Errorf("dashscope: GET request cancelled or timed out: %w", ctx.Err())
		}
		return nil, fmt.Errorf("dashscope: GET request failed (url=%s): %w", url, err)
	}
	defer resp.Body.Close()

	// 读取响应体（带大小限制防 OOM）
	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024)) // 10MB 限制
	if err != nil {
		return nil, fmt.Errorf("dashscope: failed to read GET response body: %w", err)
	}

	// 错误状态码处理
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// 特殊错误分类（便于调用方实现重试策略）
		switch resp.StatusCode {
		case http.StatusTooManyRequests: // 429
			return nil, &RateLimitError{
				Message:      "rate limited by DashScope API",
				RetryAfter:   parseRetryAfter(resp.Header),
				StatusCode:   resp.StatusCode,
				ResponseBody: body,
			}
		case http.StatusServiceUnavailable, http.StatusGatewayTimeout, // 503, 504
			http.StatusInternalServerError: // 500
			return nil, &ServerError{
				Message:      fmt.Sprintf("DashScope server error: %s", resp.Status),
				StatusCode:   resp.StatusCode,
				ResponseBody: body,
			}
		default:
			return nil, fmt.Errorf(
				"dashscope: GET request failed with status %d %s, url=%s, response=%.*s",
				resp.StatusCode,
				resp.Status,
				url,
				500, string(body), // 限制响应体输出长度，避免日志爆炸
			)
		}
	}

	return body, nil
}

// ==================== 辅助错误类型（便于调用方分类处理） ====================

// RateLimitError 表示触发限流，包含建议的重试等待时间
type RateLimitError struct {
	Message      string
	RetryAfter   time.Duration // 建议等待时间，0 表示未知
	StatusCode   int
	ResponseBody []byte
}

func (e *RateLimitError) Error() string {
	if e.RetryAfter > 0 {
		return fmt.Sprintf("%s (retry after %v)", e.Message, e.RetryAfter)
	}
	return e.Message
}

// ServerError 表示服务端临时错误，通常可重试
type ServerError struct {
	Message      string
	StatusCode   int
	ResponseBody []byte
}

func (e *ServerError) Error() string {
	return e.Message
}

// ==================== 工具函数 ====================

// parseRetryAfter 解析 Retry-After 头（支持秒数或 HTTP 日期）
func parseRetryAfter(header http.Header) time.Duration {
	retryAfter := header.Get("Retry-After")
	if retryAfter == "" {
		return 0
	}

	// 尝试解析为整数秒
	if seconds, err := strconv.Atoi(retryAfter); err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}

	// 尝试解析为 HTTP 日期
	if t, err := http.ParseTime(retryAfter); err == nil {
		if wait := time.Until(t); wait > 0 {
			return wait
		}
	}
	return 0
}
