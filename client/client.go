package client

import (
	"context"
	"log"

	"github.com/ievan-lhr/go-llm-client/llm"
	"github.com/ievan-lhr/go-llm-client/spec"
)

// Client 是一个有状态的、预配置好的LLM客户端。
type Client struct {
	config  llm.Config
	history []spec.Message
	client  spec.Client // 持有底层的 provider client 实例
}

// New 创建一个新的、有状态的LLM客户端实例。
func New(cfg llm.Config) (*Client, error) {
	// 使用 llm 包的工厂方法获取实例
	providerClient, err := llm.GetClient(cfg)
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

// invoke 调用底层的 Chat 方法，统一封装 Option 的构建逻辑
func (c *Client) invoke(ctx context.Context, messages []spec.Message, tempConfig *llm.Config) (*spec.Response, error) {
	// 使用传入的临时配置，如果没有则使用 Client 自身的配置
	cfg := c.config
	if tempConfig != nil {
		cfg = *tempConfig
	}

	var opts []spec.Option
	if cfg.Parameters != nil {
		opts = append(opts, spec.WithParameters(cfg.Parameters))
	}
	if cfg.Thinking != nil {
		opts = append(opts, spec.WithThinking(*cfg.Thinking))
	}
	if cfg.StreamCallback != nil {
		opts = append(opts, spec.WithStreamCallback(cfg.StreamCallback))
	}

	// 直接使用结构体中保存的 client 实例，无需再次查询缓存
	model := c.client.Model(cfg.Model)
	return model.Chat(ctx, messages, opts...)
}

// Send 向当前对话发送一条新消息，并返回完整的响应。
// 对话历史会被自动维护。
func (c *Client) Send(ctx context.Context, userPrompt string) (*spec.Response, error) {
	c.history = append(c.history, spec.NewUserMessage(userPrompt))

	resp, err := c.invoke(ctx, c.history, nil)
	if err != nil {
		c.history = c.history[:len(c.history)-1]
		return nil, err
	}

	c.history = append(c.history, resp.Message)
	return resp, nil
}

// SendStream 是支持流式输出的 Send 方法。
// 它接收一个 callback 函数，实时处理返回的文本片段。
func (c *Client) SendStream(ctx context.Context, userPrompt string, callback spec.StreamCallback) (*spec.Response, error) {
	c.history = append(c.history, spec.NewUserMessage(userPrompt))

	// 创建临时配置以携带回调函数
	tempConfig := c.config
	tempConfig.StreamCallback = callback

	resp, err := c.invoke(ctx, c.history, &tempConfig)
	if err != nil {
		c.history = c.history[:len(c.history)-1]
		return nil, err
	}

	c.history = append(c.history, resp.Message)
	return resp, nil
}

// SendStreamNoHistory 执行一次性的流式对话。
// 特点：
// 1. 不携带之前的对话历史 (Clean Context)。
// 2. 依然会使用初始化时的 System Prompt。
// 3. 本次对话完全独立，不会污染 Client 的 history。
func (c *Client) SendStreamNoHistory(ctx context.Context, userPrompt string, callback spec.StreamCallback) (*spec.Response, error) {
	// 1. 重新构建消息列表，只包含 System Prompt (如果有) 和当前 User Prompt
	var messages []spec.Message

	// 如果配置了系统提示词，需要加上，保证人设一致
	if c.config.SystemPrompt != "" {
		messages = append(messages, spec.NewSystemMessage(c.config.SystemPrompt))
	}

	// 添加当前用户消息
	messages = append(messages, spec.NewUserMessage(userPrompt))

	// 2. 创建临时配置以携带回调函数
	tempConfig := c.config
	tempConfig.StreamCallback = callback

	// 3. 调用 invoke
	// invoke 内部只会使用传入的 messages，不会读取 c.history
	return c.invoke(ctx, messages, &tempConfig)
}

// SendNoHistory 发送消息但不记录到历史（单次问答），但会携带之前的历史上下文
func (c *Client) SendNoHistory(ctx context.Context, userPrompt string) (*spec.Response, error) {
	var messages []spec.Message
	// 复制现有历史，避免修改底层切片
	if len(c.history) > 0 {
		messages = make([]spec.Message, len(c.history))
		copy(messages, c.history)
	}
	messages = append(messages, spec.NewUserMessage(userPrompt))

	return c.invoke(ctx, messages, nil)
}

// SendText 是Send方法的简化版，只返回回复的文本内容。
func (c *Client) SendText(userPrompt string) string {
	resp, err := c.Send(context.Background(), userPrompt)
	if err != nil {
		log.Println("LLM Error:", err.Error())
		return "对话错误，请联系管理员"
	}
	return resp.Message.Content
}

// ResetHistory 清空当前客户端的对话历史，并重新设置系统提示词。
func (c *Client) ResetHistory() {
	c.history = c.history[:0]
	if c.config.SystemPrompt != "" {
		c.history = append(c.history, spec.NewSystemMessage(c.config.SystemPrompt))
	}
}

// GetHistory 返回当前对话的完整历史记录。
func (c *Client) GetHistory() []spec.Message {
	return c.history
}
