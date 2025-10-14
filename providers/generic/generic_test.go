package generic

import (
	"context"
	"fmt"
	"github.com/ievan-lhr/go-llm-client/spec"
	"log"
	"testing"
)

func TestGen(t *testing.T) {
	// 您的私有化部署信息
	privateAPIURL := "http://acs-qwen3-30b.com.cn/v1/chat/completions"
	privateAPIKey := "" // 完整的Auth字符串
	privateModelPath := "/mnt/Qwen3-30B-A3B/"

	// 1. 调用 generic.NewClient 并传入您的私有配置
	client, err := NewClient(
		spec.WithAPIURL(privateAPIURL),
		spec.WithAPIKey(privateAPIKey),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 2. 获取模型实例，传入模型路径
	model := client.Model(privateModelPath)

	// 3. 准备消息
	messages := []spec.Message{
		spec.NewSystemMessage("You are a helpful assistant."),
		spec.NewUserMessage("你是谁？"),
	}

	// 4. 调用，无需任何特殊参数
	resp, err := model.Chat(context.Background(), messages,
		spec.WithThinking(false), // 仍然可以控制thinking
	)
	if err != nil {
		log.Fatal(err)
	}

	// 5. 您将得到被清理过的干净回复
	fmt.Printf("模型回复 (已清理): \n%s\n", resp.Message.Content)

	// 6. 思考过程依然可以获取
	fmt.Printf("\n思考过程: \n%s\n", resp.Message.ReasoningContent)
}
