package llm

import (
	"context"
	"strings"
	"testing"
)

// --- 测试配置 ---

// testConfig 用于从环境变量中加载所有测试所需的配置
type testConfig struct {
	DashscopeAPIKey  string
	GenericAPIKey    string
	GenericAPIURL    string
	GenericModelPath string
}

// loadTestConfig 是一个辅助函数，用于加载配置并处理跳过测试的逻辑
func loadTestConfig(t *testing.T) testConfig {
	cfg := testConfig{
		DashscopeAPIKey:  "",
		GenericAPIKey:    "",
		GenericAPIURL:    "https://acs-qwen3-30b.nipponpaint.com.cn/v1/chat/completions",
		GenericModelPath: "/mnt/Qwen3-30B-A3B/",
	}
	return cfg
}

// --- Dashscope Provider 测试 ---

func TestChat_Dashscope_Simple(t *testing.T) {
	cfg := loadTestConfig(t)
	if cfg.DashscopeAPIKey == "" {
		cfg.DashscopeAPIKey = "sk-e0c3fe1f6d874552bc79447193a01cd8"
	}

	config := Config{
		Provider: "dashscope",
		Model:    "qwen-turbo",
		APIKey:   cfg.DashscopeAPIKey,
	}

	resp, err := Chat(context.Background(), "hello", config)
	if err != nil {
		t.Fatalf("llm.Chat(dashscope) 返回错误: %v", err)
	}

	if resp.Message.Content == "" {
		t.Error("Dashscope响应内容为空")
	}

	t.Logf("Dashscope基础测试成功, 模型回复: %s", resp.Message.Content)
}

func TestChat_Dashscope_WithThinking(t *testing.T) {
	cfg := loadTestConfig(t)
	if cfg.DashscopeAPIKey == "" {
		t.Skip("跳过Dashscope测试: 未设置 DASHSCOPE_API_KEY")
	}

	thinking := true
	config := Config{
		Provider: "dashscope",
		Model:    "qwen-plus",
		APIKey:   cfg.DashscopeAPIKey,
		Thinking: &thinking,
	}

	resp, err := Chat(context.Background(), "你是谁？", config)
	if err != nil {
		t.Fatalf("llm.Chat(dashscope, thinking=true) 返回错误: %v", err)
	}

	if resp.Message.ReasoningContent == "" {
		t.Fatal("Dashscope思考模式未返回 reasoning_content")
	}

	t.Logf("Dashscope思考模式测试成功, 思考过程: %s", resp.Message.ReasoningContent)
}

// --- Generic Provider (私有化部署) 测试 ---

func TestChat_Generic_Simple(t *testing.T) {
	cfg := loadTestConfig(t)
	if cfg.GenericAPIKey == "" || cfg.GenericAPIURL == "" || cfg.GenericModelPath == "" {
		t.Skip("跳过Generic测试: 未完整设置 GENERIC_API_KEY, GENERIC_API_URL, GENERIC_MODEL_PATH")
	}

	config := Config{
		Provider:     "generic",
		Model:        cfg.GenericModelPath,
		APIKey:       cfg.GenericAPIKey,
		APIURL:       cfg.GenericAPIURL,
		SystemPrompt: "You are a helpful assistant.",
		Thinking:     Thinking(),
	}

	resp, err := Chat(context.Background(), "你是谁？", config)
	if err != nil {
		t.Fatalf("llm.Chat(generic) 返回错误: %v", err)
	}

	if resp.Message.Content == "" {
		t.Error("Generic响应内容为空")
	}
	// 验证<think>标签是否被移除
	if strings.Contains(resp.Message.Content, "<think>") {
		t.Error("Generic响应内容中未成功移除<think>标签")
	}

	t.Logf("Generic基础测试成功, 模型回复: %s", resp.Message.Content)
}

func TestChat_Generic_NoThinking(t *testing.T) {
	cfg := loadTestConfig(t)
	if cfg.GenericAPIKey == "" || cfg.GenericAPIURL == "" || cfg.GenericModelPath == "" {
		t.Skip("跳过Generic测试: 未完整设置 GENERIC_API_KEY, GENERIC_API_URL, GENERIC_MODEL_PATH")
	}

	thinking := false
	config := Config{
		Provider:     "generic",
		Model:        cfg.GenericModelPath,
		APIKey:       cfg.GenericAPIKey,
		APIURL:       cfg.GenericAPIURL,
		SystemPrompt: "You are a helpful assistant.",
		Thinking:     &thinking, // 明确禁用思考
	}

	// 这是一个技巧，用于测试被修改后的Prompt
	// 我们需要一种方式来检查最终发送的messages，这需要对库进行小的修改
	// 目前，我们先假设它工作正常，并检查结果
	resp, err := Chat(context.Background(), "你是谁？", config)
	if err != nil {
		t.Fatalf("llm.Chat(generic, thinking=false) 返回错误: %v", err)
	}

	// 在关闭思考模式时，reasoning_content 理论上应该为空或null
	if resp.Message.ReasoningContent != "" {
		t.Errorf("Generic关闭思考模式后, 仍收到reasoning_content: %s", resp.Message.ReasoningContent)
	}

	t.Logf("Generic关闭思考模式测试成功, 模型回复: %s", resp.Message.Content)
}
