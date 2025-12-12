package llm

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/ievan-lhr/go-llm-client/providers/dashscope"
	"github.com/ievan-lhr/go-llm-client/providers/generic"
	"github.com/ievan-lhr/go-llm-client/providers/openai"
	"github.com/ievan-lhr/go-llm-client/spec"
)

// --- 1. 无状态函数式 API (保留，不弃用) ---

var (
	clientCache = make(map[string]spec.Client)
	cacheMutex  = &sync.RWMutex{}
	thinking    = true
	noThinking  = false
)

func NoThinking() *bool {
	return &noThinking
}
func Thinking() *bool {
	return &thinking
}

// Config 包含了执行一次Chat调用所需的所有配置。
type Config struct {
	Provider     string
	Model        string
	APIKey       string
	APIURL       string
	SystemPrompt string
	Thinking     *bool
	Parameters   map[string]any
	Translation  *spec.TranslationOptions

	// 【新增】用于接收流式数据的回调函数
	StreamCallback spec.StreamCallback
}

// ChatMessages 是最核心的无状态调用函数，适用于多轮对话场景。
func ChatMessages(ctx context.Context, messages []spec.Message, cfg Config) (*spec.Response, error) {
	client, err := getOrCreateClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to get client for provider '%s': %w", cfg.Provider, err)
	}

	var opts []spec.Option
	if cfg.Parameters != nil {
		opts = append(opts, spec.WithParameters(cfg.Parameters))
	}
	if cfg.Thinking != nil {
		opts = append(opts, spec.WithThinking(*cfg.Thinking))
	}
	// 【新增】处理 Translation 配置
	if cfg.Translation != nil {
		opts = append(opts, spec.WithTranslation(cfg.Translation.SourceLang, cfg.Translation.TargetLang))
	}
	// 【新增】处理流式回调
	if cfg.StreamCallback != nil {
		opts = append(opts, spec.WithStreamCallback(cfg.StreamCallback))
	}

	model := client.Model(cfg.Model)
	return model.Chat(ctx, messages, opts...)
}

// Chat 是一个便捷的无状态调用函数，适用于简单的单轮问答。
func Chat(ctx context.Context, userPrompt string, cfg Config) (*spec.Response, error) {
	var messages []spec.Message
	if cfg.SystemPrompt != "" {
		messages = append(messages, spec.NewSystemMessage(cfg.SystemPrompt))
	}
	messages = append(messages, spec.NewUserMessage(userPrompt))
	return ChatMessages(ctx, messages, cfg)
}

// ChatText 是最简化的无状态调用函数，只返回回复的字符串。
func ChatText(ctx context.Context, userPrompt string, cfg Config) (string, error) {
	resp, err := Chat(ctx, userPrompt, cfg)
	if err != nil {
		return "", err
	}
	return resp.Message.Content, nil
}

// --- 2. 有状态的客户端 API (已弃用，请迁移至 client 包) ---

// Client 是一个有状态的、预配置好的LLM客户端。
//
// Deprecated: 请使用 client.Client 代替。
type Client struct {
	config  Config
	history []spec.Message
	client  spec.Client // 底层的provider client
}

// New 创建一个新的、有状态的LLM客户端实例。
//
// Deprecated: 请使用 client.New(cfg) 代替。
func New(cfg Config) (*Client, error) {
	providerClient, err := getOrCreateClient(cfg)
	if err != nil {
		return nil, err
	}

	var history []spec.Message
	if cfg.SystemPrompt != "" {
		history = append(history, spec.NewSystemMessage(cfg.SystemPrompt))
	}

	return &Client{
		config:  cfg,
		history: history,
		client:  providerClient,
	}, nil
}

// Send 向当前对话发送一条新消息，并返回完整的响应。
// 对话历史会被自动维护。
//
// Deprecated: 请使用 client.Client.Send 代替。
func (c *Client) Send(ctx context.Context, userPrompt string) (*spec.Response, error) {
	// 将新用户消息加入历史记录
	c.history = append(c.history, spec.NewUserMessage(userPrompt))

	// 使用当前完整的历史记录进行调用
	resp, err := ChatMessages(ctx, c.history, c.config)
	if err != nil {
		// 如果调用失败，将刚才加入的用户消息移除，保持历史记录的清洁
		c.history = c.history[:len(c.history)-1]
		return nil, err
	}

	// 将模型的回复也加入历史记录
	c.history = append(c.history, resp.Message)
	return resp, nil
}

// SendStream 是支持流式输出的 Send 方法。
// 它接收一个 callback 函数，实时处理返回的文本片段。
// 方法结束后，依然会返回完整的 Response 对象，并自动维护对话历史。
//
// Deprecated: 请使用 client.Client.SendStream 代替。
func (c *Client) SendStream(ctx context.Context, userPrompt string, callback spec.StreamCallback) (*spec.Response, error) {
	// 1. 将新用户消息加入历史记录
	c.history = append(c.history, spec.NewUserMessage(userPrompt))

	// 2. 创建临时配置，将回调函数注入进去
	// 我们不想修改 Client 自身的 c.config，因为那会影响后续的调用
	tempConfig := c.config
	tempConfig.StreamCallback = callback

	// 3. 使用当前完整的历史记录进行调用
	resp, err := ChatMessages(ctx, c.history, tempConfig)
	if err != nil {
		// 如果调用失败，回滚历史记录
		c.history = c.history[:len(c.history)-1]
		return nil, err
	}

	// 4. 将模型的回复（ChatMessages 会在流式结束后返回完整内容）也加入历史记录
	c.history = append(c.history, resp.Message)
	return resp, nil
}

// SendNoHistory 发送消息但不记录到历史。
//
// Deprecated: 请使用 client.Client.SendNoHistory 代替。
func (c *Client) SendNoHistory(ctx context.Context, userPrompt string) (*spec.Response, error) {
	var messages []spec.Message
	if len(c.history) != 0 {
		messages = append(messages, c.history...)
		messages = append(messages, spec.NewUserMessage(userPrompt))
	} else {
		messages = append(messages, spec.NewUserMessage(userPrompt))
	}
	// 使用当前完整的历史记录进行调用
	resp, err := ChatMessages(ctx, messages, c.config)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SendText 是Send方法的简化版，只返回回复的文本内容。
//
// Deprecated: 请使用 client.Client.SendText 代替。
func (c *Client) SendText(userPrompt string) string {
	resp, err := c.Send(context.Background(), userPrompt)
	if err != nil {
		log.Println(err.Error())
		return "对话错误，请联系管理员"
	}
	return resp.Message.Content
}

// SendTextNoHistory 是Send方法的简化版，只返回回复的文本内容。
//
// Deprecated: 请使用 client.Client.SendTextNoHistory 代替。
func (c *Client) SendTextNoHistory(userPrompt string) string {
	resp, err := c.SendNoHistory(context.Background(), userPrompt)
	if err != nil {
		log.Println(err.Error())
		return "对话错误，请联系管理员"
	}
	return resp.Message.Content
}

// ResetHistory 清空当前客户端的对话历史，并重新设置系统提示词。
//
// Deprecated: 请使用 client.Client.ResetHistory 代替。
func (c *Client) ResetHistory() {
	c.history = c.history[:0] // 清空切片
	if c.config.SystemPrompt != "" {
		c.history = append(c.history, spec.NewSystemMessage(c.config.SystemPrompt))
	}
}

// GetHistory 返回当前对话的完整历史记录。
//
// Deprecated: 请使用 client.Client.GetHistory 代替。
func (c *Client) GetHistory() []spec.Message {
	return c.history
}

// --- 内部辅助函数 ---

// getOrCreateClient 负责创建和缓存客户端实例。
func getOrCreateClient(cfg Config) (spec.Client, error) {
	cacheKey := fmt.Sprintf("%s|%s|%s", cfg.Provider, cfg.APIURL, cfg.APIKey)
	cacheMutex.RLock()
	client, found := clientCache[cacheKey]
	cacheMutex.RUnlock()

	if found {
		return client, nil
	}

	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	client, found = clientCache[cacheKey]
	if found {
		return client, nil
	}

	clientOpts := []spec.ClientOption{
		spec.WithAPIKey(cfg.APIKey),
	}
	if cfg.APIURL != "" {
		clientOpts = append(clientOpts, spec.WithAPIURL(cfg.APIURL))
	}

	// 这里可以根据需要传入 HTTPClient
	// if cfg.HTTPClient != nil { ... }

	var newClient spec.Client
	var err error

	switch cfg.Provider {
	case "dashscope":
		newClient, err = dashscope.NewClient(clientOpts...)
	case "generic":
		newClient, err = generic.NewClient(clientOpts...)
	case "openai":
		newClient, err = openai.NewClient(clientOpts...)
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}

	if err != nil {
		return nil, err
	}

	clientCache[cacheKey] = newClient
	return newClient, nil
}
