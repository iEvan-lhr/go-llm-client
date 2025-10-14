package spec

// Response 是从模型Chat方法返回的通用响应结构
type Response struct {
	// Message 是模型返回的核心消息内容
	Message Message

	// Usage 包含了本次调用的token使用情况等元数据 (可选, 未来可扩展)
	// Usage UsageStats

	// RawResponse 存储了来自API的原始、未经修改的http响应体
	RawResponse []byte
}
