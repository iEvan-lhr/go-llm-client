package llm

import "github.com/ievan-lhr/go-llm-client/spec"

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
