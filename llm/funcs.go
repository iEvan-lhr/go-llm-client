package llm

import (
	"context"
	"fmt"
	"github.com/ievan-lhr/go-llm-client/spec"
)

// ChatMessages 是最核心的无状态调用函数，适用于多轮对话场景。
func ChatMessages(ctx context.Context, messages []spec.Message, cfg Config) (*spec.Response, error) {
	client, err := GetClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to get client for provider '%s': %w", cfg.Provider, err)
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

	model := client.Model(cfg.Model)
	return model.Chat(ctx, messages, opts...)
}

// Chat 是一个便捷的无状态调用函数，适用于简单的单轮问答。
func Chat(ctx context.Context, userPrompt string, cfg Config) (*spec.Response, error) {
	var messages []spec.Message
	if cfg.SystemPrompt != "" {
		messages = append(messages, spec.NewSystemMessage(cfg.SystemPrompt))
	}
	messages = append(messages, spec.NewUserMessage(userPrompt))
	return ChatMessages(ctx, messages, cfg)
}

// ChatText 是最简化的无状态调用函数，只返回回复的字符串。
func ChatText(ctx context.Context, userPrompt string, cfg Config) (string, error) {
	resp, err := Chat(ctx, userPrompt, cfg)
	if err != nil {
		return "", err
	}
	return resp.Message.Content, nil
}
