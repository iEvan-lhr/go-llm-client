package spec

// Role 定义了消息发送者的角色
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message 代表一次对话中的单条消息
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`

	// 【新增】ReasoningContent 用于存储模型返回的思考过程或工具调用信息。
	// `omitempty` 表示如果该字段为空，则在序列化为JSON时忽略它。
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// NewSystemMessage 创建一条系统消息
func NewSystemMessage(content string) Message {
	return Message{Role: RoleSystem, Content: content}
}

// NewUserMessage 创建一条用户消息
func NewUserMessage(content string) Message {
	return Message{Role: RoleUser, Content: content}
}

// NewAssistantMessage 创建一条助手（AI）消息
func NewAssistantMessage(content string) Message {
	return Message{Role: RoleAssistant, Content: content}
}
