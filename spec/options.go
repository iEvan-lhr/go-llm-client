package spec

import (
	"context"
	"net/http"
	"time"
)

// --- 1. Client Options ---
// 用于在创建顶级客户端（如Dashscope, OpenAI）时进行配置。

// ClientOption 是一个用于配置Client的函数类型。
type ClientOption func(c *ClientConfig)

// ClientConfig 存储了客户端级别的所有配置。
// 这是一个不导出的结构体，用户通过ClientOption函数来修改它。
type ClientConfig struct {
	APIKey     string
	APIURL     string
	HTTPClient *http.Client
	Text2Image bool
}

// NewClientConfig 创建一个带有默认值的客户端配置。
func NewClientConfig() *ClientConfig {
	return &ClientConfig{
		// 设置一个合理的默认HTTP客户端
		HTTPClient: &http.Client{Timeout: 240 * time.Second},
	}
}

// WithAPIKey 设置提供商的API Key。
// 这是最常用的选项之一。
func WithAPIKey(key string) ClientOption {
	return func(c *ClientConfig) {
		c.APIKey = key
	}
}

// WithAPIURL 覆盖提供商默认的API基础URL。
// 在需要连接到代理服务器或私有部署模型时非常有用。
func WithAPIURL(url string) ClientOption {
	return func(c *ClientConfig) {
		c.APIURL = url
	}
}

// WithHTTPClient 允许用户传入一个完全自定义的http.Client。
// 可用于配置复杂的网络设置，如自定义Transport、TLS配置或代理。
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *ClientConfig) {
		c.HTTPClient = client
	}
}

// --- 2. Request Options ---
// 用于在单次调用Chat方法时，微调该次请求的参数。

// Option 是一个用于配置单次API请求的函数类型。
type Option func(r *RequestConfig)

// StreamCallback 定义了流式回调函数。
// ctx: 上下文
// chunk: 本次收到的增量文本内容
// 返回 error 则中断流式接收
type StreamCallback func(ctx context.Context, chunk string) error

var DefaultTemperature = 0.2

// RequestConfig 存储了单次请求的所有配置。
type RequestConfig struct {
	Model       string
	Temperature *float32
	MaxTokens   *int
	TopP        *float32
	Streaming   bool

	// 【新增】StreamCallback 用于处理流式输出的每一个数据块
	StreamCallback StreamCallback

	// 【新增】Thinking 用于统一控制思考模式。
	// 使用指针 *bool 可以区分三种状态:
	// - nil:   用户未指定，使用Provider的默认行为。
	// - true:  用户明确要求开启思考模式。
	// - false: 用户明确要求关闭思考模式。
	Thinking *bool

	Parameters map[string]any

	text2Image bool
	imageEdit  bool
	Provider   map[string]any
}

func WithProvider(provider map[string]any) Option {
	return func(c *RequestConfig) {
		c.Provider = provider
	}
}

func (r *RequestConfig) SetText2Image(text2Image bool) {
	r.text2Image = text2Image
}

func (r *RequestConfig) IsText2Image() bool {
	return r.text2Image
}

func (r *RequestConfig) SetImageEdit(imageEdit bool) {
	r.imageEdit = imageEdit
}

func (r *RequestConfig) IsImageEdit() bool {
	return r.imageEdit
}

// NewRequestConfig 创建一个带有默认值的请求配置。
func NewRequestConfig() *RequestConfig {
	return &RequestConfig{
		Parameters: make(map[string]any),
		Streaming:  false,
		// Thinking 默认是 nil
	}
}

// WithThinking 是控制思考模式的通用选项。
// 用户只需要调用这个函数，库会自动适配不同的模型。
func WithThinking(enabled bool) Option {
	return func(r *RequestConfig) {
		r.Thinking = &enabled
	}
}

// WithModel 在单次请求中设置模型名称。
// 允许临时使用不同于客户端默认模型的其他模型。
func WithModel(model string) Option {
	return func(r *RequestConfig) {
		r.Model = model
	}
}

// WithTemperature 设置生成文本的随机性（温度）。
// 值越高，结果越随机；值越低，结果越确定。
func WithTemperature(temp float32) Option {
	return func(r *RequestConfig) {
		r.Temperature = &temp
	}
}

// WithMaxTokens 设置本次请求生成的最大token数。
func WithMaxTokens(max int) Option {
	return func(r *RequestConfig) {
		r.MaxTokens = &max
	}
}

// WithTopP 设置核心采样的概率阈值。
func WithTopP(topP float32) Option {
	return func(r *RequestConfig) {
		r.TopP = &topP
	}
}

// WithStreaming 启用流式响应。
// (注意: Provider的具体实现需要支持流式解析才能使其生效)。
func WithStreaming() Option {
	return func(r *RequestConfig) {
		r.Streaming = true
	}
}

// WithStreamCallback 启用流式响应并设置回调函数。
// 推荐使用此方法开启流式模式。
func WithStreamCallback(callback StreamCallback) Option {
	return func(r *RequestConfig) {
		r.Streaming = true
		r.StreamCallback = callback
	}
}

// WithParameters 附加一个map中所有的任意键值对参数。
// 如果key已存在，则会被覆盖。
func WithParameters(params map[string]any) Option {
	return func(r *RequestConfig) {
		for k, v := range params {
			r.Parameters[k] = v
		}
	}
}

// WithText2ImageParameters 附加一个map中所有的任意键值对参数。
// 如果key已存在，则会被覆盖。
func WithText2ImageParameters(params map[string]any) Option {
	return func(r *RequestConfig) {
		for k, v := range params {
			r.Parameters[k] = v
		}
		r.text2Image = true
	}
}

// WithText2Image 附加一个map中所有的任意键值对参数。
// 如果key已存在，则会被覆盖。
func WithText2Image() Option {
	return func(r *RequestConfig) {
		r.text2Image = true
	}
}

// WithParameter 附加单个任意键值对参数。
// 这是最灵活的选项，用于传递特定模型的专有参数。
func WithParameter(key string, value any) Option {
	return func(r *RequestConfig) {
		r.Parameters[key] = value
	}
}

// [在 spec/options.go 中添加以下代码]

// TranslationOptions 定义了翻译模型的特定参数
type TranslationOptions struct {
	SourceLang string `json:"source_lang"` // 例如 "auto", "English", "Chinese"
	TargetLang string `json:"target_lang"` // 例如 "English", "Chinese"
}

// WithTranslation 是一个专用选项，用于设置翻译参数。
// 它会将结构化的参数注入到通用的 Parameters map 中，从而被 Provider 自动处理。
func WithTranslation(sourceLang, targetLang string) Option {
	return func(r *RequestConfig) {
		// 确保 Parameters 已初始化
		if r.Parameters == nil {
			r.Parameters = make(map[string]any)
		}
		r.Parameters["translation_options"] = TranslationOptions{
			SourceLang: sourceLang,
			TargetLang: targetLang,
		}
	}
}

// ============== 新增：文生图配置结构体和选项 ==============

// Text2ImageConfig 文生图专用配置
type Text2ImageConfig struct {
	Size           string // 图像尺寸，如 "1024*1024", "2048*2048"
	Watermark      *bool  // 是否添加水印，nil 表示使用默认值
	NegativePrompt string // 负面提示词
	PromptExtend   *bool  // 是否扩展提示词，nil 表示使用默认值
	ImageCount     int    // 生成图像数量，默认 1
}

// Text2ImageOption 文生图配置选项函数类型
type Text2ImageOption func(*Text2ImageConfig)

// WithText2ImageSize 设置图像尺寸
// 支持的尺寸：512*512, 720*720, 1024*1024, 1664*928, 2048*2048 等
func WithText2ImageSize(size string) Text2ImageOption {
	return func(cfg *Text2ImageConfig) {
		cfg.Size = size
	}
}

// WithText2ImageWatermark 设置是否添加水印
func WithText2ImageWatermark(enable bool) Text2ImageOption {
	return func(cfg *Text2ImageConfig) {
		cfg.Watermark = &enable
	}
}

// WithText2ImageNegativePrompt 设置负面提示词
func WithText2ImageNegativePrompt(negativePrompt string) Text2ImageOption {
	return func(cfg *Text2ImageConfig) {
		cfg.NegativePrompt = negativePrompt
	}
}

// WithText2ImagePromptExtend 设置是否扩展提示词
func WithText2ImagePromptExtend(enable bool) Text2ImageOption {
	return func(cfg *Text2ImageConfig) {
		cfg.PromptExtend = &enable
	}
}

// WithText2ImageCount 设置生成图像数量
func WithText2ImageCount(count int) Text2ImageOption {
	return func(cfg *Text2ImageConfig) {
		cfg.ImageCount = count
	}
}
