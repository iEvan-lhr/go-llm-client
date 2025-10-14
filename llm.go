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

// --- 1. 无状态函数式 API ---
// 适用于一次性的、不需要上下文记忆的调用。

// clientCache 用于缓存已初始化的客户端，避免重复创建，提高性能。
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
}

// Chat 是最核心的无状态调用函数，适用于多轮对话场景。
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

// --- 2. 有状态的客户端 API ---
// 适用于需要上下文记忆的、持续的对话机器人场景。

// Client 是一个有状态的、预配置好的LLM客户端。
type Client struct {
	config  Config
	history []spec.Message
	client  spec.Client // 底层的provider client
}

// New 创建一个新的、有状态的LLM客户端实例。
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
func (c *Client) SendNoHistory(ctx context.Context, userPrompt string) (*spec.Response, error) {
	var messages []spec.Message
	if len(c.history) != 0 {
		messages = append(messages, c.history...)
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
func (c *Client) SendText(userPrompt string) string {
	resp, err := c.Send(context.Background(), userPrompt)
	if err != nil {
		log.Println(err.Error())
		return "对话错误，请联系管理员"
	}
	return resp.Message.Content
}

// SendTextNoHistory 是Send方法的简化版，只返回回复的文本内容。
func (c *Client) SendTextNoHistory(userPrompt string) string {
	resp, err := c.SendNoHistory(context.Background(), userPrompt)
	if err != nil {
		log.Println(err.Error())
		return "对话错误，请联系管理员"
	}
	return resp.Message.Content
}

// ResetHistory 清空当前客户端的对话历史，并重新设置系统提示词。
func (c *Client) ResetHistory() {
	c.history = c.history[:0] // 清空切片
	if c.config.SystemPrompt != "" {
		c.history = append(c.history, spec.NewSystemMessage(c.config.SystemPrompt))
	}
}

// GetHistory 返回当前对话的完整历史记录。
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
