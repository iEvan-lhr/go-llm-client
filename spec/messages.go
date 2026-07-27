package spec

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Role 定义了消息发送者的角色
type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
	RoleFunction  Role = "function"
)

// Message 代表一次对话中的单条消息
type Message struct {
	Role       Role          `json:"role"`
	Content    string        `json:"content"`
	Parts      []ContentPart `json:"content_part"`
	Name       string        `json:"name,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall    `json:"tool_calls,omitempty"`
	Refusal    string        `json:"refusal,omitempty"`
	// ReasoningContent stores provider-specific reasoning text.
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

// NewToolMessage creates a Chat Completions tool result message.
func NewToolMessage(toolCallID, content string) Message {
	return Message{Role: RoleTool, ToolCallID: toolCallID, Content: content}
}

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
	FileURL  string    `json:"file_url,omitempty"`
	FileID   string    `json:"file_id,omitempty"`
	FileData string    `json:"file_data,omitempty"`
	Filename string    `json:"filename,omitempty"`
}

func (m *Message) MarshalJSON() ([]byte, error) {
	type alias struct {
		Role             Role       `json:"role"`
		Content          any        `json:"content"`
		Name             string     `json:"name,omitempty"`
		ToolCallID       string     `json:"tool_call_id,omitempty"`
		ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
		Refusal          string     `json:"refusal,omitempty"`
		ReasoningContent string     `json:"reasoning_content,omitempty"`
	}

	var content any
	if len(m.Parts) > 0 {
		content = m.Parts
	} else if m.Content == "" && len(m.ToolCalls) > 0 {
		content = nil
	} else {
		content = m.Content
	}

	return json.Marshal(alias{
		Role:             m.Role,
		Content:          content,
		Name:             m.Name,
		ToolCallID:       m.ToolCallID,
		ToolCalls:        m.ToolCalls,
		Refusal:          m.Refusal,
		ReasoningContent: m.ReasoningContent,
	})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role             Role            `json:"role"`
		Content          json.RawMessage `json:"content"`
		Name             string          `json:"name"`
		ToolCallID       string          `json:"tool_call_id"`
		ToolCalls        []ToolCall      `json:"tool_calls"`
		Refusal          string          `json:"refusal"`
		ReasoningContent string          `json:"reasoning_content"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	m.Role = raw.Role
	m.Name = raw.Name
	m.ToolCallID = raw.ToolCallID
	m.ToolCalls = raw.ToolCalls
	m.Refusal = raw.Refusal
	m.ReasoningContent = raw.ReasoningContent

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

func NewInputFilePart(url string) ContentPart {
	return ContentPart{
		Type:    "input_file",
		FileURL: url,
	}
}

func NewInputFileBase64Part(mimeType, base64Data string) ContentPart {
	return ContentPart{
		Type:     "input_file",
		FileData: fmt.Sprintf("data:%s;base64,%s", mimeType, base64Data),
	}
}

func NewInputFileBytesPart(mimeType string, data []byte) ContentPart {
	return NewInputFileBase64Part(mimeType, base64.StdEncoding.EncodeToString(data))
}

func NewInputFileLocalPart(path, mimeType string) (ContentPart, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ContentPart{}, err
	}
	part := NewInputFileBytesPart(mimeType, data)
	part.Filename = filepath.Base(path)
	return part, nil
}

// NewInputFileIDPart creates an input_file referencing an uploaded OpenAI file.
func NewInputFileIDPart(fileID string) ContentPart {
	return ContentPart{Type: "input_file", FileID: fileID}
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
