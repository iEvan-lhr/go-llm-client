package spec

import "encoding/json"

// Response 是从模型Chat方法返回的通用响应结构
type Response struct {
	// Message 是模型返回的核心消息内容
	Message Message

	// Protocol identifies whether Chat Completions or Responses was used.
	Protocol Protocol

	// ID, Model, Status, and Usage expose common protocol metadata.
	ID     string
	Model  string
	Status string
	Usage  *Usage

	// ChatCompletion and Responses preserve each protocol's typed envelope.
	ChatCompletion *ChatCompletionResponse
	Responses      *ResponsesAPIResponse

	// RawResponse 存储了来自API的原始、未经修改的http响应体
	RawResponse []byte

	// OCRResult 存储了 OCR 模型提取的结构化布局或键值信息
	OCRResult *OCRResult
}

type OCRStyle struct {
	Bold       bool    `json:"bold"`
	CharScale  float64 `json:"charScale"`
	Color      string  `json:"color"`
	DeleteLine bool    `json:"deleteLine"`
	FontName   string  `json:"fontName"`
	FontSize   float64 `json:"fontSize"`
	Italic     bool    `json:"italic"`
	Underline  bool    `json:"underline"`
}

type OCRBlock struct {
	Style OCRStyle `json:"style"`
	Text  string   `json:"text"`
}

type OCRPos struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

type OCRLayout struct {
	Alignment       string     `json:"alignment"`
	Blocks          []OCRBlock `json:"blocks"`
	FirstLinesChars int        `json:"firstLinesChars"`
	Index           int        `json:"index"`
	Level           int        `json:"level"`
	MarkdownContent string     `json:"markdownContent"`
	PageNum         int        `json:"pageNum"`
	Pos             []OCRPos   `json:"pos"`
	SubType         string     `json:"subType"`
	Text            string     `json:"text"`
	Type            string     `json:"type"`
	UniqueId        string     `json:"uniqueId"`
}

type OCRResult struct {
	Layouts  []OCRLayout    `json:"layouts,omitempty"`
	KVResult map[string]any `json:"kv_result,omitempty"`
}

// UnmarshalJSON implements custom unmarshaling to handle both layout objects and general key-value maps
func (o *OCRResult) UnmarshalJSON(data []byte) error {
	type alias OCRResult
	var temp alias
	if err := json.Unmarshal(data, &temp); err == nil && len(temp.Layouts) > 0 {
		*o = OCRResult(temp)
		return nil
	}

	var kv map[string]any
	if err := json.Unmarshal(data, &kv); err == nil {
		o.KVResult = kv
		return nil
	}

	return nil
}
