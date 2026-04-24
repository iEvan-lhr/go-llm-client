package client

import (
	"context"
	"fmt"
	"log"

	"github.com/iEvan-lhr/go-llm-client/llm"
	"github.com/iEvan-lhr/go-llm-client/spec"
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
func (c *Client) invoke(ctx context.Context, messages []spec.Message, tempConfig *llm.Config, extraOpts ...spec.Option) (*spec.Response, error) {
	// 使用传入的临时配置，如果没有则使用 Client 自身的配置
	cfg := c.config
	if tempConfig != nil {
		cfg = *tempConfig
	}

	var opts []spec.Option
	// 【新增】处理 WebExtractor：将工具组装到 Parameters 中，同时执行深拷贝避免污染全局配置
	// 【核心修复】适配 Chat Completions API 的联网搜索参数
	if cfg.WebExtractor != nil {
		newParams := make(map[string]any)
		if cfg.Parameters != nil {
			for k, v := range cfg.Parameters {
				newParams[k] = v
			}
		}

		// 使用顶级参数 enable_search，废弃 tools 数组形式以避免 OpenAI Schema 校验报错
		if cfg.WebExtractor.EnableSearch {
			newParams["enable_search"] = true
		}

		if cfg.WebExtractor.EnableExtractor {
			newParams["search_options"] = map[string]any{
				"search_strategy": "agent_max", // 对应 curl 中的配置
			}
		}

		// 如果官方在 Chat Completions 中需要开启代码解释器，通常也是通过特定顶级参数或特定的模型版本
		// 这里我们先满足联网搜索和抓取的核心需求

		cfg.Parameters = newParams
	}
	if cfg.Parameters != nil {
		opts = append(opts, spec.WithParameters(cfg.Parameters))
	}
	if cfg.ProviderOpts != nil {
		opts = append(opts, spec.WithProvider(cfg.ProviderOpts))
	}
	if cfg.Thinking != nil {
		opts = append(opts, spec.WithThinking(*cfg.Thinking))
	}
	// 【新增】处理 Translation 配置
	if cfg.Translation != nil {
		opts = append(opts, spec.WithTranslation(cfg.Translation.SourceLang, cfg.Translation.TargetLang))
	}
	if cfg.StreamCallback != nil {
		opts = append(opts, spec.WithStreamCallback(cfg.StreamCallback))
	}
	if len(extraOpts) > 0 {
		opts = append(opts, extraOpts...)
	}
	// 直接使用结构体中保存的 client 实例，无需再次查询缓存
	model := c.client.Model(cfg.Model)
	return model.Chat(ctx, messages, opts...)
}

// SendEmbedding 获取文本的向量表示。
// 参数 input 可以是一段文本 (string)，也可以是多段文本的切片 ([]string)。
func (c *Client) SendEmbedding(ctx context.Context, input any) (*spec.EmbeddingResponse, error) {
	// 获取底层具体的模型实例
	model := c.client.Model(c.config.Model)

	// 使用类型断言，判断当前模型提供商是否支持向量化接口
	if embedded, ok := model.(spec.Embedded); ok {
		return embedded.Embed(ctx, input)
	}

	// 如果断言失败，说明该 Provider 尚未实现 Embed 方法
	return nil, fmt.Errorf("provider '%s' model '%s' does not support embeddings (Embedder interface not implemented)", c.config.Provider, c.config.Model)
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

// SendParts 发送多模态消息，并写入历史
func (c *Client) SendParts(ctx context.Context, parts ...spec.ContentPart) (*spec.Response, error) {
	c.history = append(c.history, spec.NewUserPartsMessage(parts...))

	resp, err := c.invoke(ctx, c.history, nil)
	if err != nil {
		c.history = c.history[:len(c.history)-1]
		return nil, err
	}

	c.history = append(c.history, resp.Message)
	return resp, nil
}

// ============== 修改：SendText2Image 方法 ==============

// SendText2Image 发送文生图请求（支持自定义配置）
// 用法示例：
//
//	resp, err := client.SendText2Image(ctx, "一只可爱的猫")
//	resp, err := client.SendText2Image(ctx, "一只可爱的猫",
//	    WithText2ImageSize("2048*2048"),
//	    WithText2ImageWatermark(false),
//	    WithText2ImageNegativePrompt("低分辨率，模糊"))
func (c *Client) SendText2Image(ctx context.Context, userPrompt string, opts ...spec.Text2ImageOption) (*spec.Response, error) {
	c.history = append(c.history, spec.NewUserMessage(userPrompt))

	// 应用文生图配置选项
	tiConfig := applyText2ImageOptions(opts...)

	// 将文生图配置转换为 Parameters map
	parameters := map[string]any{
		"size": tiConfig.Size,
		"n":    tiConfig.ImageCount,
	}
	if tiConfig.Watermark != nil {
		parameters["watermark"] = *tiConfig.Watermark
	}
	if tiConfig.PromptExtend != nil {
		parameters["prompt_extend"] = *tiConfig.PromptExtend
	}
	if tiConfig.NegativePrompt != "" {
		parameters["negative_prompt"] = tiConfig.NegativePrompt
	}

	// 创建临时配置，注入 Parameters
	tempConfig := &llm.Config{
		Model:      c.config.Model,
		Parameters: parameters,
	}

	resp, err := c.invoke(ctx, c.history, tempConfig, spec.WithText2Image())
	if err != nil {
		c.history = c.history[:len(c.history)-1]
		return nil, err
	}

	c.history = append(c.history, resp.Message)
	return resp, nil
}

// applyText2ImageOptions 应用文生图选项到配置
func applyText2ImageOptions(opts ...spec.Text2ImageOption) *spec.Text2ImageConfig {
	cfg := &spec.Text2ImageConfig{
		Size:         "1024*1024", // 默认尺寸
		Watermark:    ptrBool(false),
		PromptExtend: ptrBool(true),
		ImageCount:   1,
	}
	for _, opt := range opts {
		opt(cfg)
	}
	return cfg
}

// ptrBool 辅助函数：返回 bool 指针
func ptrBool(b bool) *bool {
	return &b
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

func (c *Client) SendStreamParts(ctx context.Context, parts []spec.ContentPart, callback spec.StreamCallback) (*spec.Response, error) {
	c.history = append(c.history, spec.NewUserPartsMessage(parts...))

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

func (c *Client) SendImageURL(ctx context.Context, imageURL, question string) (*spec.Response, error) {
	return c.SendParts(ctx,
		spec.NewImageURLPart(imageURL),
		spec.NewTextPart(question),
	)
}

func (c *Client) SendImageBase64(ctx context.Context, mimeType, base64Data, question string) (*spec.Response, error) {
	return c.SendParts(ctx,
		spec.NewImageBase64Part(mimeType, base64Data),
		spec.NewTextPart(question),
	)
}

func (c *Client) SendPartsNoHistory(ctx context.Context, parts ...spec.ContentPart) (*spec.Response, error) {
	var messages []spec.Message
	if c.config.SystemPrompt != "" {
		messages = append(messages, spec.NewSystemMessage(c.config.SystemPrompt))
	}
	messages = append(messages, spec.NewUserPartsMessage(parts...))
	return c.invoke(ctx, messages, nil)
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
