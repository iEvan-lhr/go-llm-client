package llm

import (
	"context"
	"fmt"
	"github.com/ievan-lhr/go-llm-client/providers/openai"
	"github.com/ievan-lhr/go-llm-client/spec"
	"sync"

	"github.com/ievan-lhr/go-llm-client/providers/dashscope" // 导入所有支持的provider
	"github.com/ievan-lhr/go-llm-client/providers/generic"
)

// clientCache 用于缓存已初始化的客户端，避免重复创建http连接，提高性能。
var (
	clientCache = make(map[string]spec.Client)
	cacheMutex  = &sync.RWMutex{}
	NO_THINK    = false
	THINK       = true
)

// Config 包含了执行一次Chat调用所需的所有配置。
// 这是用户与本库交互的主要配置结构。
type Config struct {
	// --- 核心配置 ---
	Provider string // 指定要使用的提供商, e.g., "dashscope", "generic"
	Model    string // 指定要使用的模型, e.g., "qwen-plus", "/mnt/Qwen3-30B-A3B/"
	APIKey   string // 您的API Key

	// --- 可选配置 ---
	APIURL       string // 用于私有化部署，覆盖默认的API地址
	SystemPrompt string // (可选) 便捷地设置系统提示词

	// --- 高级控制 ---
	Thinking   *bool          // 统一的思考模式开关
	Parameters map[string]any // 传递任意自定义参数
}

func NoThinking() *bool {
	return &NO_THINK
}
func Thinking() *bool {
	return &THINK
}

// Chat 是本库最核心、最简单的统一调用函数。
// 它接收一个问题和一份配置，返回模型的回复。
func Chat(ctx context.Context, userPrompt string, cfg Config) (*spec.Response, error) {
	// 1. 构造消息列表
	var messages []spec.Message
	if cfg.SystemPrompt != "" {
		messages = append(messages, spec.NewSystemMessage(cfg.SystemPrompt))
	}
	messages = append(messages, spec.NewUserMessage(userPrompt))

	// 2. 调用更底层的ChatMessages函数
	return ChatMessages(ctx, messages, cfg)
}

// ChatMessages 是一个功能更完整的统一调用函数。
// 它允许用户传递一个完整的消息历史（多轮对话）。
func ChatMessages(ctx context.Context, messages []spec.Message, cfg Config) (*spec.Response, error) {
	// 1. 根据配置，获取或创建一个缓存的客户端
	client, err := getOrCreateClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to get client for provider '%s': %w", cfg.Provider, err)
	}

	// 2. 构造本次请求的选项
	var opts []spec.Option
	if cfg.Parameters != nil {
		opts = append(opts, spec.WithParameters(cfg.Parameters))
	}
	if cfg.Thinking != nil {
		opts = append(opts, spec.WithThinking(*cfg.Thinking))
	}

	// 3. 从客户端获取模型并执行Chat
	model := client.Model(cfg.Model)
	return model.Chat(ctx, messages, opts...)
}

// getOrCreateClient 是一个内部函数，负责创建和缓存客户端实例。
func getOrCreateClient(cfg Config) (spec.Client, error) {
	// 使用Provider、URL和Key生成一个唯一的缓存键
	cacheKey := fmt.Sprintf("%s|%s|%s", cfg.Provider, cfg.APIURL, cfg.APIKey)

	cacheMutex.RLock()
	client, found := clientCache[cacheKey]
	cacheMutex.RUnlock()

	if found {
		return client, nil // 从缓存中直接返回
	}

	// 如果缓存中没有，则创建新的实例
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	// 双重检查，防止在等待锁的过程中其他goroutine已经创建了实例
	client, found = clientCache[cacheKey]
	if found {
		return client, nil
	}

	// 准备创建客户端所需的选项
	clientOpts := []spec.ClientOption{
		spec.WithAPIKey(cfg.APIKey),
	}
	if cfg.APIURL != "" {
		clientOpts = append(clientOpts, spec.WithAPIURL(cfg.APIURL))
	}

	var newClient spec.Client
	var err error

	// 根据Provider名称，调用对应provider的构造函数
	switch cfg.Provider {
	case "dashscope":
		newClient, err = dashscope.NewClient(clientOpts...)
	case "generic":
		newClient, err = generic.NewClient(clientOpts...)
	case "openai": // <-- 2. 添加新的 case
		newClient, err = openai.NewClient(clientOpts...)
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}

	if err != nil {
		return nil, err
	}

	// 将新创建的客户端存入缓存
	clientCache[cacheKey] = newClient
	return newClient, nil
}
