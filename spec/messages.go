package spec

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Role 定义了消息发送者的角色
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message 代表一次对话中的单条消息
type Message struct {
	Role    Role          `json:"role"`
	Content string        `json:"content"`
	Parts   []ContentPart `json:"content_part"`
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

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

func (m *Message) MarshalJSON() ([]byte, error) {
	type alias struct {
		Role    Role `json:"role"`
		Content any  `json:"content"`
	}

	var content any
	if len(m.Parts) > 0 {
		content = m.Parts
	} else {
		content = m.Content
	}

	return json.Marshal(alias{
		Role:    m.Role,
		Content: content,
	})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    Role            `json:"role"`
		Content json.RawMessage `json:"content"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	m.Role = raw.Role

	if len(raw.Content) == 0 || string(raw.Content) == "null" {
		return nil
	}

	// content 是字符串
	if raw.Content[0] == '"' {
		return json.Unmarshal(raw.Content, &m.Content)
	}

	// content 是数组（多模态）
	if raw.Content[0] == '[' {
		return json.Unmarshal(raw.Content, &m.Parts)
	}

	// 兜底：如果未来 provider 返回别的结构，至少别直接炸
	m.Content = string(raw.Content)
	return nil
}

func NewUserPartsMessage(parts ...ContentPart) Message {
	return Message{
		Role:  RoleUser,
		Parts: parts,
	}
}

func NewTextPart(text string) ContentPart {
	return ContentPart{
		Type: "text",
		Text: text,
	}
}

func NewImageURLPart(url string) ContentPart {
	return ContentPart{
		Type: "image_url",
		ImageURL: &ImageURL{
			URL: url,
		},
	}
}

func NewImageURLPartWithDetail(url, detail string) ContentPart {
	return ContentPart{
		Type: "image_url",
		ImageURL: &ImageURL{
			URL:    url,
			Detail: detail,
		},
	}
}

func NewImageBase64Part(mimeType, base64Data string) ContentPart {
	return NewImageURLPart(fmt.Sprintf("data:%s;base64,%s", mimeType, base64Data))
}

func NewImageBytesPart(mimeType string, data []byte) ContentPart {
	return NewImageBase64Part(mimeType, base64.StdEncoding.EncodeToString(data))
}

func NewImageFilePart(path, mimeType string) (ContentPart, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ContentPart{}, err
	}
	return NewImageBytesPart(mimeType, data), nil
}

// PlainText 如果你还想兼容 SendText 这种调用，可以加一个取纯文本的方法
func (m *Message) PlainText() string {
	if m.Content != "" {
		return m.Content
	}
	var sb strings.Builder
	for _, p := range m.Parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}
