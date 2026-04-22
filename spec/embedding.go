package spec

import "context"

// Embedded 定义了支持向量化能力的方法集
// 采用可选接口设计，不强制所有的 Model 都必须实现
type Embedded interface {
	Embed(ctx context.Context, input any) (*EmbeddingResponse, error)
}

// EmbeddingData 单条向量数据的结构
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingResponse 向量化接口的完整响应结构 (兼容 OpenAI 规范)
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// EmbeddingUsage 向量化调用的 Token 消耗
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
