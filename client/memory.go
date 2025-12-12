package client

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ievan-lhr/go-llm-client/spec"
)

// SendByMemory 是一种“外部状态”调用方式。
// 它不使用 Client 内部的 history，而是使用传入的 memoryJSON 字符串作为上下文。
// 方法内部会自动创建 context.Background()。
//
// 返回：本次的响应结果、更新后的 memoryJSON 字符串、错误信息。
// 适用场景：Web 服务后端，需要将上下文存储在 Redis/数据库中，每次请求时取出传入。
func (c *Client) SendByMemory(userPrompt string, memoryJSON string) (*spec.Response, string, error) {
	// 在内部新建上下文，简化调用方传参
	ctx := context.Background()

	var messages []spec.Message

	// 1. 解析传入的记忆字符串 (JSON -> []Message)
	if memoryJSON != "" {
		if err := json.Unmarshal([]byte(memoryJSON), &messages); err != nil {
			return nil, memoryJSON, fmt.Errorf("failed to parse memory json: %w", err)
		}
	}

	// 2. 如果记忆为空，且配置了系统提示词，则自动注入 System Prompt
	// (这保证了即使是外部记忆，也能通过 Client 统一管理 System Prompt)
	if len(messages) == 0 && c.config.SystemPrompt != "" {
		messages = append(messages, spec.NewSystemMessage(c.config.SystemPrompt))
	}

	// 3. 追加当前用户问题
	messages = append(messages, spec.NewUserMessage(userPrompt))

	// 4. 调用底层模型 (使用 invoke 方法复用 Config 逻辑)
	// 注意：这里传 nil 作为 tempConfig，表示使用 Client 初始化的默认配置
	resp, err := c.invoke(ctx, messages, nil)
	if err != nil {
		// 如果出错，返回原始记忆，不包含本次失败的对话
		return nil, memoryJSON, err
	}

	// 5. 将 AI 的回答追加到上下文中
	messages = append(messages, resp.Message)

	// 6. 序列化更新后的记忆 ([]Message -> JSON)
	newMemoryBytes, err := json.Marshal(messages)
	if err != nil {
		return nil, memoryJSON, fmt.Errorf("failed to marshal new memory: %w", err)
	}

	return resp, string(newMemoryBytes), nil
}
