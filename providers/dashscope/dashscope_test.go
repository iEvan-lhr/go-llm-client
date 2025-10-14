package dashscope

import (
	"context"
	"github.com/ievan-lhr/go-llm-client/spec"
	"os"
	"testing"
)

// getAPIKey 辅助函数
func getAPIKey(t *testing.T) string {
	apiKey := os.Getenv("DASHSCOPE_API_KEY")
	if apiKey == "" {
		apiKey = ""
	}
	return apiKey
}

// TestDashscopeClient_Chat 测试基础聊天功能
func TestDashscopeClient_Chat(t *testing.T) {
	apiKey := getAPIKey(t)
	client, err := NewClient(spec.WithAPIKey(apiKey))
	if err != nil {
		t.Fatalf("创建客户端失败: %v", err)
	}
	model := client.Model("qwen-turbo")
	messages := []spec.Message{
		spec.NewUserMessage("hello"),
	}

	resp, err := model.Chat(context.Background(), messages)
	if err != nil {
		t.Fatalf("Chat() 返回错误: %v", err)
	}
	if resp == nil {
		t.Fatal("Chat() 响应为空")
	}
	if resp.Message.Content == "" {
		t.Error("响应内容为空")
	}
	if resp.Message.ReasoningContent != "" {
		t.Errorf("基础模式不应返回思考过程, 但收到了: %s", resp.Message.ReasoningContent)
	}

	t.Logf("基础测试成功, 模型回复: %s", resp.Message.Content)
}

// TestDashscopeClient_ChatWithThinking 测试思考模式并验证reasoning_content字段
func TestDashscopeClient_ChatWithThinking(t *testing.T) {
	apiKey := getAPIKey(t)
	client, err := NewClient(spec.WithAPIKey(apiKey))
	if err != nil {
		t.Fatalf("创建客户端失败: %v", err)
	}

	model := client.Model("qwen-plus")
	messages := []spec.Message{
		spec.NewUserMessage("你是谁？"),
	}

	resp, err := model.Chat(context.Background(), messages, spec.WithThinking(false))
	if err != nil {
		t.Fatalf("带思考模式的Chat() 返回错误: %v", err)
	}
	if resp == nil {
		t.Fatal("带思考模式的Chat() 响应为空")
	}
	if resp.Message.Content == "" {
		t.Error("带思考模式的响应内容为空")
	}

	t.Logf("思考模式测试成功!")
	t.Logf("模型回复: %s", resp.Message.Content)
	t.Logf("思考过程: %s", resp.Message.ReasoningContent)
}

func TestCall(t *testing.T) {
	//client, _ := NewClient(llm.WithAPIKey("..."))
	//model := client.Model("qwen-plus")
	//
	//// 统一的调用方式
	////resp, _ := model.Chat(context.Background(), messages, llm.WithThinking(false))
}
