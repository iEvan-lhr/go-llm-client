package dashscope

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url" // [新增] 用于安全的 URL 解析
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

// handleText2Image 处理文生图异步任务流程
func (m *modelImpl) handleText2Image(ctx context.Context, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	// 1. 提取 Prompt（取最后一条用户消息）
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided for text2image")
	}
	prompt := messages[len(messages)-1].Content
	if prompt == "" {
		return nil, fmt.Errorf("empty prompt for text2image")
	}

	// 2. 构建请求体（自动映射 negative_prompt 到 input，其他参数到 parameters）
	requestBody := map[string]any{
		"model": m.name,
		"input": map[string]any{
			"prompt": prompt,
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
		if negPrompt, ok := config.Parameters["negative_prompt"]; ok {
			if inputMap, ok := requestBody["input"].(map[string]any); ok {
				inputMap["negative_prompt"] = negPrompt
			}
		}
		for _, key := range []string{"size", "n", "prompt_extend", "watermark"} {
			if val, ok := config.Parameters[key]; ok {
				if paramMap, ok := requestBody["parameters"].(map[string]any); ok {
					paramMap[key] = val
				}
			}
		}
	}

	// 3. 构建请求头（关键：启用异步模式）
	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+m.client.config.APIKey)
	headers.Set("X-DashScope-Async", "enable")

	// 4. 发起任务创建请求（固定使用文生图端点）
	text2ImageURL := "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
	if strings.Contains(m.client.config.APIURL, "dashscope-intl") {
		text2ImageURL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
	}

	rawBody, err := m.client.requester.Post(ctx, text2ImageURL, headers, requestBody)
	if err != nil {
		return nil, fmt.Errorf("dashscope text2image create task failed: %w", err)
	}

	// 5. 解析任务 ID
	var createResp struct {
		Output struct {
			TaskId     string `json:"task_id"`
			TaskStatus string `json:"task_status"`
		}
	}
	if err := json.Unmarshal(rawBody, &createResp); err != nil {
		return nil, fmt.Errorf("dashscope failed to parse task_id: %w, response: %s", err, string(rawBody))
	}
	if createResp.Output.TaskId == "" {
		return nil, fmt.Errorf("empty task_id in create response: %s", string(rawBody))
	}

	// 6. 轮询任务状态（生产级策略：前30秒每3秒，之后指数退避，总超时120秒）
	// 【修复 2】使用 url.Parse 安全提取 Scheme 和 Host 进行任务 API 拼接
	parsedURL, err := url.Parse(m.client.config.APIURL)
	if err != nil {
		return nil, fmt.Errorf("dashscope invalid api url: %w", err)
	}
	taskURL := fmt.Sprintf("%s://%s/api/v1/tasks/%s", parsedURL.Scheme, parsedURL.Host, createResp.Output.TaskId)

	var lastTaskResp []byte
	maxRetries := 40
	delay := 3 * time.Second // 初始轮询延迟

	for i := 0; i < maxRetries; i++ {
		// 检查上下文取消
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("text2image task cancelled: %w", err)
		}

		// 【修复 5】采用更加安全的动态翻倍退避策略
		if i > 0 {
			time.Sleep(delay)
			if i > 10 { // 前 10 次不退避 (约30秒)，之后开始翻倍
				delay *= 2
				if delay > 15*time.Second {
					delay = 15 * time.Second
				}
			}
		}

		// 查询任务状态
		taskRespBody, err := m.Get(ctx, taskURL, headers)
		lastTaskResp = taskRespBody
		if err != nil {
			continue // 网络错误重试
		}

		var taskResult struct {
			RequestId string `json:"request_id"`
			Output    struct {
				TaskId        string `json:"task_id"`
				TaskStatus    string `json:"task_status"`
				SubmitTime    string `json:"submit_time"`
				ScheduledTime string `json:"scheduled_time"`
				EndTime       string `json:"end_time"`
				Results       []struct {
					OrigPrompt   string `json:"orig_prompt"`
					ActualPrompt string `json:"actual_prompt"`
					Url          string `json:"url"`
				} `json:"results"`
			} `json:"output"`
			Usage struct {
				ImageCount int `json:"image_count"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(taskRespBody, &taskResult); err != nil {
			continue // 解析失败重试
		}

		switch taskResult.Output.TaskStatus {
		case "SUCCEEDED":
			if len(taskResult.Output.Results) == 0 {
				return nil, fmt.Errorf("task succeeded but no image results returned")
			}
			imageURL := taskResult.Output.Results[0].Url
			if imageURL == "" {
				return nil, fmt.Errorf("empty image URL in task result")
			}
			return &spec.Response{
				Message: spec.Message{
					Role:    spec.RoleAssistant,
					Content: imageURL,
				},
				RawResponse: taskRespBody,
			}, nil

		case "FAILED":
			return nil, fmt.Errorf("dashscope text2image task failed (code: %s): %s",
				taskResult.Output.TaskStatus, taskResult.Output.Results)

		case "PENDING", "RUNNING", "QUEUED":
			continue // 继续轮询

		default:
			return nil, fmt.Errorf("unknown task status: %s, response: %s",
				taskResult.Output.TaskStatus, string(taskRespBody))
		}
	}

	return nil, fmt.Errorf("text2image task timeout after 120 seconds, last response: %s", string(lastTaskResp))
}

// handleChat 处理标准聊天请求（流式/非流式）
func (m *modelImpl) handleChat(ctx context.Context, messages []spec.Message, config *spec.RequestConfig) (*spec.Response, error) {
	// 【修复 1】避免浅拷贝修改用户的 config.Parameters 引发的数据污染和并发 Panic
	requestBody := make(map[string]any)
	if config.Parameters != nil {
		for k, v := range config.Parameters {
			requestBody[k] = v
		}
	}

	// 【适配逻辑】将通用的 Thinking 选项翻译为 provider 特定的参数
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
		// DashScope 协议要求：设置 stream_options 以获取 Token 消耗
		requestBody["stream_options"] = map[string]bool{"include_usage": true}

		// 使用 PostStream 获取 http.Response
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
			// 【修复 4】兼顾带空格和不带空格的前置情况，并裁剪空白
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

			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta
				if delta.Role != "" {
					role = delta.Role
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
