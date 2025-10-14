package spec

import "context"

// Model 是一个具体LLM模型的抽象接口。
type Model interface {
	Chat(ctx context.Context, messages []Message, opts ...Option) (*Response, error)
}

// Client 是与特定LLM提供商交互的顶层客户端。
type Client interface {
	Model(name string) Model
}
