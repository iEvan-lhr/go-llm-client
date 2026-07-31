# Go LLM Client

一个轻量、统一且易于扩展的 Go 语言大模型客户端库。旨在屏蔽不同大模型供应商（Dashscope/Qwen, OpenAI, DeepSeek 等）的接口差异，提供统一的**有状态（Stateful）**和**无状态（Stateless）**调用体验。

## ✨ 核心特性

* **统一接口**：一套代码适配 Dashscope (阿里云百炼)、OpenAI 及各类私有化部署模型（Generic）。
* **客户端模式 (Client)**：内置上下文记忆管理，像聊天一样简单地调用。
* **多模态与文生图 (New 🚀)**：原生支持文生图 (Text-to-Image) 异步任务模型（如 DashScope 的 `qwen-image-plus`），轻松集成 AI 绘画能力。
* **流式响应 (Streaming)**：支持打字机效果，提供便捷的回调函数 (`StreamCallback`)。
* **灵活的对话控制**：支持带历史对话、不带历史对话 (`SendNoHistory`) 以及流式不记录 (`SendStreamNoHistory`) 等多种模式。
* **思考模式支持**：针对 DeepSeek R1 / Qwen 等推理模型，自动处理 `<think>` 标签或特定参数。

## 📦 安装

```bash
go get github.com/iEvan-lhr/go-llm-client

```

## 🚀 快速开始 (Recommended)

推荐使用 `client` 包创建一个有状态的客户端。它会自动为您维护对话历史，同时也支持单次临时问答。

### 1. 基础流式对话 (无历史记录模式)

这是最常用的场景之一，适用于一次性的问答、翻译或摘要任务。

```go
package main

import (
    "context"
    "fmt"
    "os"

    // 引入两个核心包
    "github.com/ievan-lhr/go-llm-client/client" // 核心客户端，管理会话
    "github.com/ievan-lhr/go-llm-client/llm"    // 包含配置定义和通用类型
)

func main() {
    // 1. 初始化客户端
    // 注意：使用 client.New 而不是 llm.New
    c, err := client.New(llm.Config{
       Provider: "dashscope", // 支持 "dashscope", "openai", "generic"
       Model:    "qwen-plus",
       APIKey:   os.Getenv("DASHSCOPE_API_KEY"),
       // 如果是 Dashscope，通常不需要手动设置 APIURL，库内有默认值。
       // 这里仅作示例，展示如何自定义 URL
       APIURL:   "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    })
    if err != nil {
       panic(err)
    }

    fmt.Print("AI: ")

    // 2. 调用流式方法 (SendStreamNoHistory)
    // 特点：实时返回内容，且本次对话不会污染客户端的历史记忆
    _, err = c.SendStreamNoHistory(context.Background(), "诸葛亮是谁？", func(ctx context.Context, chunk string) error {
       // 实时打印每一个输出片段
       fmt.Print(chunk)
       return nil
    })

    if err != nil {
       fmt.Printf("\nError: %v\n", err)
    }
    fmt.Println("\n--- 完成 ---")
}

```

### 2. 多轮对话 (自动维护历史)

如果您需要实现一个聊天机器人，使用 `SendStream` 或 `Send` 方法，客户端会自动记录上下文。

```go
func main() {
    // ... 初始化 client (同上) ...

    // 第一轮：发送并记录历史
    c.SendStream(context.Background(), "你好，我叫小明", func(ctx context.Context, chunk string) error {
        fmt.Print(chunk)
        return nil
    })
    fmt.Println()

    // 第二轮：大模型会记得上面的名字
    c.SendStream(context.Background(), "我刚才说了我叫什么？", func(ctx context.Context, chunk string) error {
        fmt.Print(chunk)
        return nil
    })
}

```

## 🎨 图像生成 (Text-to-Image)

库内部已封装好异步长轮询逻辑，您可以像发送普通文本一样简单地调用文生图 API。

```go
func main() {
    // 初始化时请确保使用支持画图的模型，如 qwen-image-plus
    c, err := client.New(llm.Config{
       Provider: "dashscope",
       Model:    "qwen-image-plus",
       APIKey:   os.Getenv("DASHSCOPE_API_KEY"), 
       Text2Image: true,
    })
    if err != nil {
       panic(err)
    }

    // 画图任务通常耗时较长，建议设置充足的超时时间 (如 60-120 秒)
    ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
    defer cancel()

    fmt.Println("正在为您生成图片，请稍候...")
    
    // 调用专门的 SendText2Image 接口
    resp, err := c.SendText2Image(ctx, "一只穿着宇航服的可爱橘猫，在月球表面散步，背景是璀璨的星空，8k分辨率，3D渲染")
    if err != nil {
       fmt.Printf("生成失败: %v\n", err)
       return
    }

    // Content 将返回生成好的图片下载 URL (通常有 24 小时有效期)
    fmt.Println("图片生成成功！URL:", resp.Message.Content)
}

```

## OpenAI Chat Completions 与 Responses API

`openai` provider 会根据接口路径自动选择协议：`/chat/completions` 继续使用原有 Chat Completions 格式，`/responses` 使用 OpenAI Responses 格式。下面的配置会向 `POST https://www.poke2api.com/v1/responses` 发送请求：

```go
c, err := client.New(llm.Config{
	Provider: "openai",
	Model:    "gpt-5.6-sol",
	APIKey:   os.Getenv("OPENAI_API_KEY"),
	APIURL:   "https://www.poke2api.com/v1/responses",
	// 默认 10 分钟；长时间联网或推理任务可以自定义。
	Timeout: 15 * time.Minute,
	// 可选：none、minimal、low、medium、high、xhigh；具体可用等级由模型决定。
	ReasoningEffort: llm.ReasoningEffortHigh,
})
if err != nil {
	panic(err)
}

// 文本
resp, err := c.SendNoHistory(context.Background(), "用一句话介绍 Go")
fmt.Println(resp.Message.Content)
fmt.Println(resp.ID, resp.Status, resp.Usage)

// 图片理解（也支持 data:image/...;base64 URL）
resp, err = c.SendPartsNoHistory(
	context.Background(),
	spec.NewImageURLPartWithDetail("https://example.com/image.jpg", "high"),
	spec.NewTextPart("描述这张图片"),
)

// 流式文本输出
resp, err = c.SendStreamNoHistory(context.Background(), "写一首短诗", func(_ context.Context, chunk string) error {
	fmt.Print(chunk)
	return nil
})
```

图片流式理解可使用 `SendStreamParts`；最终完整文本仍会写入 `resp.Message.Content`。协议级完整结果分别位于 `resp.ChatCompletion` 和 `resp.Responses`。非流式原始 JSON 位于 `resp.RawResponse`；流式原始 chunk/事件位于 `StreamEvent.Raw`。

### Responses 联网搜索

OpenAI Responses API 可通过托管的 `web_search` 工具联网。`SendWebSearch` 会保留 `Parameters` 中已有的 Function Calling 工具，并追加联网工具：

```go
resp, err := c.SendWebSearch(
	context.Background(),
	"查询今天的重要 AI 新闻，并标注来源",
	spec.WebSearchConfig{
		SearchContextSize: spec.WebSearchContextSizeMedium,
		ExternalWebAccess: spec.Bool(true),
		Filters: &spec.WebSearchFilters{
			AllowedDomains: []string{"openai.com", "reuters.com"},
		},
		IncludeSources: true,
		ToolChoice:     "required", // auto 表示由模型决定是否搜索
	},
)
if err != nil {
	return err
}

fmt.Println(resp.Message.Content)
for _, call := range resp.WebSearchCalls {
	for _, source := range call.Action.Sources {
		fmt.Println(source.Title, source.URL)
	}
}
for _, citation := range resp.Citations {
	fmt.Println(citation.Title, citation.URL)
}
```

`WebSearchConfig` 还支持 `ReturnTokenBudget`、`UserLocation`、`SearchContentTypes`、`ImageSettings` 和 `IncludeResults`。搜索正文中的 `url_citation` 会解析到 `resp.Citations`；请求 `IncludeSources` 后，完整来源位于 `resp.WebSearchCalls[].Action.Sources`。向最终用户展示联网结果时，应让引用清晰可见并可点击。

该封装只允许 `/responses` 端点使用 `WithWebSearch`。第三方兼容服务是否真正透传 OpenAI 托管工具仍取决于服务商；可以填写 `openai_llm_test.go` 的配置后运行 `go test -run TestOpenAIResponsesWebSearch -v` 实测。

### Responses 连续对话

`SendResponse` 直接接受 Responses API 的 `input`。`ContinueResponse` 会发送 `previous_response_id`，不会重复发送本地聊天历史：

```go
first, err := c.SendResponse(ctx, "我叫小明")
if err != nil {
	return err
}

next, err := c.ContinueResponse(ctx, first.ID, "我叫什么？")
```

也可以通过配置或单次 Option 显式设置：

```go
resp, err := c.SendResponse(
	ctx,
	"解释一下 RAG",
	spec.WithInstructions("你是一个助手"),
	spec.WithPreviousResponseID("resp_xxxxx"),
)
```

### Function Calling

请求中的 `tools`、`tool_choice`、`parallel_tool_calls` 等字段可通过 `Parameters` 或 `spec.WithParameter` 透传。返回的函数调用会同时保留在 `resp.Responses.Output` 中，并转换到 `resp.Message.ToolCalls` 方便统一处理：

```go
resp, err := c.SendResponse(
	ctx,
	"上海天气如何？",
	spec.WithParameter("tools", []map[string]any{{
		"type": "function",
		"name": "get_weather",
		"description": "查询城市天气",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		},
	}}),
)

call := resp.Message.ToolCalls[0]
toolResult := `{"temperature":30,"condition":"sunny"}`
resp, err = c.ContinueResponse(
	ctx,
	resp.ID,
	[]spec.ResponseInputItem{
		spec.NewFunctionCallOutput(call.ID, toolResult),
	},
)
```

Chat Completions 的工具结果可用 `spec.NewToolMessage(toolCallID, content)` 追加到 `messages`。

### 完整流式事件

`StreamCallback` 只接收最终文本增量；`ReasoningCallback` 接收推理摘要增量；`EventCallback` 会收到每一个原始 SSE 事件，包括 output item、函数参数、MCP/hosted tool 和完成事件：

```go
resp, err := c.SendResponse(
	ctx,
	"完成这个任务",
	spec.WithStreamCallback(func(_ context.Context, delta string) error {
		fmt.Print(delta)
		return nil
	}),
	spec.WithEventCallback(func(_ context.Context, event spec.StreamEvent) error {
		log.Printf("event=%s raw=%s", event.Type, event.Raw)
		return nil
	}),
)
```

`llm.Config.Parameters` 会原样透传两种 API 的扩展参数；Responses 模式下 `max_tokens` 会自动转换为 `max_output_tokens`。未知的新 output item 或事件不会丢失，可从 `RawResponse` 或 `StreamEvent.Raw` 读取。

> `poke2api.com` 是第三方兼容服务。客户端支持上述 OpenAI 协议结构，不代表第三方服务一定实现 `previous_response_id`、MCP、hosted tools、文件或全部事件类型，请以该服务的实际响应为准。

## 🔍 文档/多模态 OCR 与本地文件上传 (New 🚀)

库提供了对阿里云百炼的 `qwen3.5-ocr` 模型及其专用 Responses API 的完整封装，支持传入公网 URL 或直接加载并上传本地 PDF、图像文件，同时提供了完备的强类型版面分析/键值抽取结果结构体。

### 1. 支持的文件传入方式
通过 `spec` 提供的构建函数，您可以非常方便地传递各种文件输入：
* **公网 URL 文件**: `spec.NewInputFilePart("https://example.com/doc.pdf")`
* **本地路径文件**: `spec.NewInputFileLocalPart("path/to/doc.pdf", "application/pdf")` （自动进行 Base64 编码并组合为 Data URI）
* **内存二进制数据**: `spec.NewInputFileBytesPart("application/pdf", data)`

### 2. 调用示例

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/iEvan-lhr/go-llm-client/client"
	"github.com/iEvan-lhr/go-llm-client/llm"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

func main() {
	c, err := client.New(llm.Config{
		Provider: "dashscope",
		Model:    "qwen3.5-ocr", // 使用 OCR 模型
		APIKey:   os.Getenv("DASHSCOPE_API_KEY"),
	})
	if err != nil {
		panic(err)
	}

	// 1. 读取本地 PDF 文件
	localPart, err := spec.NewInputFileLocalPart("testing.pdf", "application/pdf")
	if err != nil {
		panic(err)
	}

	// 2. 调用并指定 OCR 任务参数 (如 document_parsing 或 advanced_recognition)
	resp, err := c.SendPartsNoHistory(
		context.Background(),
		localPart,
		spec.WithParameter("ocr_options", map[string]any{"task": "document_parsing"}),
	)
	if err != nil {
		panic(err)
	}

	// 3. 提取 Markdown 格式的文本内容
	fmt.Println("=== 解析结果 ===")
	fmt.Println(resp.Message.Content)

	// 4. 获取详细的结构化布局与样式元数据
	if resp.OCRResult != nil {
		fmt.Printf("解析到 %d 个版面块\n", len(resp.OCRResult.Layouts))
		for _, layout := range resp.OCRResult.Layouts {
			fmt.Printf("[%s] (第 %d 页): %s\n", layout.Type, layout.PageNum, layout.Text)
		}
	}
}
```

## 📚 API 方法速查

### `client.Client` 方法

| 方法 | 说明 | 适用场景 |
| --- | --- | --- |
| **`Send`** | 发送消息，**记录历史**，等待完整回复 | 常规多轮非流式对话 |
| **`SendStream`** | 发送消息，**记录历史**，流式回调 | 常规多轮流式对话 (打字机) |
| **`SendText2Image`** | 发送提示词，触发 **文生图** 任务 | AI 画图、视觉生成 (返回图片URL) |
| **`SendNoHistory`** | 发送消息，**携带**历史但不记录本次 | 基于上下文的临时追问 |
| **`SendStreamNoHistory`** | 发送消息，**不携带**且不记录历史 | 独立的一次性任务 (如翻译/搜索) |
| **`SendOCR`** | 传入文件 Part，发送 OCR 识别请求 | 细粒度控制的多模态文件 OCR |
| **`SendOCRURL`** | 传入公网 URL，发送 OCR 识别请求 | 快速公网文档 OCR 解析 |
| **`SendOCRLocal`** | 传入本地路径，自动编码并发送 OCR 请求 | 快速本地文档 OCR 解析 (自动 Base64) |
| **`SendOCRBytes`** | 传入二进制字节流 `[]byte`，发送 OCR 请求 | 内存/流式二进制文档 OCR 解析 |
| **`ResetHistory`** | 清空对话历史 | 重置会话 |

### `llm.Config` 配置项

| 字段 | 说明 |
| --- | --- |
| `Provider` | 厂商标识: `dashscope`, `openai`, `generic` |
| `Model` | 模型名称: `qwen-plus`, `gpt-4o`, `qwen-image-plus` 等 |
| `APIKey` | API 密钥 |
| `APIURL` | (可选) 自定义接口地址，用于代理或私有部署 |
| `Timeout` | (可选) 完整 HTTP 请求超时，默认 10 分钟，例如 `15*time.Minute` |
| `Thinking` | (可选) `llm.Thinking()` 开启思考模式适配 |
| `ReasoningEffort` | (可选) 思考等级，如 `llm.ReasoningEffortLow`、`llm.ReasoningEffortMedium`、`llm.ReasoningEffortHigh` |
| `SystemPrompt` | (可选) 系统预设人设 |

## 💡 高级用法

### 开启 "Thinking" (思考/推理) 模式

适配 DeepSeek R1 或 Qwen 等具备推理能力的模型。

```go
c, _ := client.New(llm.Config{
    Provider: "generic", // 或 dashscope
    Model:    "deepseek-r1",
    APIKey:   "...",
    Thinking:        llm.Thinking(),           // 开启思考模式适配
    ReasoningEffort: llm.ReasoningEffortHigh, // 设置思考等级
})

```

* **Dashscope**: 会自动传递 `enable_thinking` 参数。
* **Generic**: 会自动清洗返回内容中的 `<think>...</think>` 标签（视具体实现而定）。

### 无状态调用 (Stateless)

如果您不需要创建 Client 对象，也可以直接使用 `llm` 包提供的函数式接口：

```go
import "github.com/ievan-lhr/go-llm-client/llm"

// 单次直接调用
resp, err := llm.ChatText(context.Background(), "简单介绍一下 Go 语言", llm.Config{
    Provider: "openai",
    APIKey:   "sk-...",
    Model:    "gpt-4o",
})
fmt.Println(resp)

```

## License

MIT
