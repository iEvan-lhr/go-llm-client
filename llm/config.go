package llm

import "github.com/iEvan-lhr/go-llm-client/spec"

// Config 包含了执行一次Chat调用所需的所有配置。
type Config struct {
	Provider     string
	Model        string
	APIKey       string
	APIURL       string
	SystemPrompt string
	Thinking     *bool
	Parameters   map[string]any
	//add
	Translation *spec.TranslationOptions
	// StreamCallback 用于接收流式数据的回调函数
	StreamCallback spec.StreamCallback
	// 图片相关操作
	Text2Image bool
	ImageEdit  bool
	// 新增网页抓取配置
	WebExtractor *WebExtractorOptions

	ProviderOpts map[string]any
}

var (
	thinking   = true
	noThinking = false
)

// NoThinking 返回 false 的指针，用于关闭思考模式
func NoThinking() *bool {
	return &noThinking
}

// Thinking 返回 true 的指针，用于开启思考模式
func Thinking() *bool {
	return &thinking
}

// WebExtractorOptions 配置网页抓取所需的工具选项
type WebExtractorOptions struct {
	EnableSearch          bool // 开启网页抓取必须同时开启联网搜索
	EnableExtractor       bool // 网页抓取
	EnableCodeInterpreter bool // 推荐同时开启，处理计算和数据分析问题效果更好
}
