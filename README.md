# Go LLM Client

一个轻量、统一且易于扩展的 Go 语言大模型客户端库。旨在屏蔽不同大模型供应商（Dashscope/Qwen、OpenAI、DeepSeek、ZHIPU AI 等）的接口差异，提供统一的**有状态（Stateful）**和**无状态（Stateless）**调用体验。

## ✨ 核心特性

* **统一接口**：一套代码适配 Dashscope (阿里云百炼)、OpenAI、ZHIPU AI（智谱）及各类私有化部署模型（Generic）。
* **客户端模式 (Client)**：内置上下文记忆管理，像聊天一样简单地调用。
* **多模态输入与图像生成 (New 🚀)**：支持图片 URL/Base64/本地文件、音频、文档和文件 ID 输入，并支持 DashScope 异步文生图、OpenAI Responses `image_generation` 以及独立的 Images 生成/编辑 API。
* **DashScope 实时语音识别**：统一支持 Qwen-Audio 3.0、Fun-ASR、Qwen3-ASR-Realtime 与 Paraformer 的双向 WebSocket 音频流和实时转写事件。
* **流式响应 (Streaming)**：支持打字机效果，提供便捷的回调函数 (`StreamCallback`)。
* **灵活的对话控制**：支持带历史对话、不带历史对话 (`SendNoHistory`) 以及流式不记录 (`SendStreamNoHistory`) 等多种模式。
* **思考模式支持**：针对 DeepSeek R1 / Qwen 等推理模型，自动处理 `<think>` 标签或特定参数。
* **完整 OpenAI Responses 支持**：覆盖 SSE、WebSocket、Function Calling、Hosted Tools、结构化输出、后台任务、续流、Compaction、Conversations、输入 token 计数和原始字段保留。

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
    "github.com/iEvan-lhr/go-llm-client/client" // 核心客户端，管理会话
    "github.com/iEvan-lhr/go-llm-client/llm"    // 包含配置定义和通用类型
)

func main() {
    // 1. 初始化客户端
    // 注意：使用 client.New 而不是 llm.New
    c, err := client.New(llm.Config{
       Provider: "dashscope", // 支持 "dashscope", "openai", "zhipu", "generic"
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

## ZHIPU AI（智谱）

将 `Provider` 设置为 `zhipu` 即可使用智谱 Chat Completions。默认请求官方端点 `https://open.bigmodel.cn/api/paas/v4/chat/completions`，通常无需手动设置 `APIURL`：

```go
c, err := client.New(llm.Config{
    Provider:        "zhipu",
    Model:           "glm-5.3",
    APIKey:          os.Getenv("ZHIPU_API_KEY"),
    Thinking:        llm.Thinking(),
    ReasoningEffort: llm.ReasoningEffortHigh,
    ReasoningCallback: func(ctx context.Context, chunk string) error {
        fmt.Print(chunk) // 思考内容 reasoning_content
        return nil
    },
    // 智谱新增参数可以直接透传。
    Parameters: map[string]any{"do_sample": true},
})
if err != nil {
    panic(err)
}

resp, err := c.SendStreamNoHistory(context.Background(), "解释量子纠缠", func(ctx context.Context, chunk string) error {
    fmt.Print(chunk) // 最终回答 content
    return nil
})
```

文本、图片、视频、文件和音频输入可分别使用 `NewTextPart`、`NewImageURLPart`、`NewVideoURLPart`、`NewInputFilePart` 和 `NewInputAudioPart` 构造；Function Calling 可通过 `Parameters` 透传标准 `tools` 与 `tool_choice`。响应中的正文、`reasoning_content`、函数调用、结束原因和 token 用量会映射到统一 `spec.Response`，完整服务端响应仍保留在 `RawResponse`。

智谱支持两种联网搜索方式：`SendWebSearch` 让模型在 Chat Completions 中调用 `web_search` 工具；`SearchWeb` 直接调用智谱 `/paas/v4/web_search` 搜索接口并返回结构化结果：

```go
// 模型搜索并生成回答（Chat Completions）
answer, err := c.SendWebSearch(context.Background(), "今天有哪些重要 AI 新闻？", spec.WebSearchConfig{
    SearchEngine:   spec.WebSearchEngineStandard,
    SearchIntent:   spec.Bool(false),
    Count:          5,
    ContentSize:    spec.WebSearchContentSizeMedium,
    IncludeSources: true,
})

// 直接获取搜索结果（/paas/v4/web_search）
results, err := c.SearchWeb(context.Background(), spec.WebSearchRequest{
    SearchQuery:  "今天 AI 新闻",
    SearchEngine: spec.WebSearchEngineStandard,
    SearchIntent: false,
    Count:        5,
})
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

## DashScope 实时语音识别

外部应用推荐使用 `StreamRealtimeTranscription`：传入麦克风、FFmpeg stdout、网络连接等任意 `io.Reader`，客户端会自动完成建连、音频分块、并发接收、结束任务、等待最终结果和关闭会话：

```go
err := c.StreamRealtimeTranscription(
	ctx,
	microphoneReader,
	spec.RealtimeTranscriptionRequest{
		Format:        "pcm",
		SampleRate:    16000,
		LanguageHints: []string{"zh", "en"},
	},
	spec.RealtimeTranscriptionStreamOptions{
		ChunkSize: 3200, // 16kHz/16-bit/单声道 PCM 的 100ms 音频
		OnText: func(ctx context.Context, text string, final bool) error {
			fmt.Printf("text=%q final=%t\n", text, final)
			return nil
		},
		OnEvent: func(ctx context.Context, event spec.RealtimeTranscriptionEvent) error {
			// 可选：接收时间戳、情绪、用量和原始服务端事件。
			return nil
		},
	},
)
```

输入流返回 `io.EOF` 后，该方法会自动发送结束指令并等待 `Terminal` 事件。若音频源的 `Read` 会长期阻塞，应用在取消 `ctx` 时也应关闭该音频源。Qwen3 Manual 模式会把整个输入流作为一句话并自动 `Commit`；需要在一个连接内手动提交多句话时，请使用下面的底层会话接口。

`StartRealtimeTranscription` 会根据模型自动选择 DashScope 的 WebSocket 协议，并在服务端确认任务启动后返回。一个 goroutine 可以持续调用 `Receive`，另一个 goroutine 同时调用 `SendAudio`。以下底层示例适用于 `qwen-audio-3.0-asr-flash-streaming`、`fun-asr-realtime` 和 Paraformer 系列：

```go
c, err := client.New(llm.Config{
	Provider: "dashscope",
	Model:    "qwen-audio-3.0-asr-flash-streaming",
	APIKey:   os.Getenv("DASHSCOPE_API_KEY"),
	// 推荐填写业务空间专属地址；模型需要另一套协议时会自动切换路径。
	APIURL: "wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
})
if err != nil {
	return err
}

heartbeat := true
session, err := c.StartRealtimeTranscription(ctx, spec.RealtimeTranscriptionRequest{
	Format:        "pcm",
	SampleRate:    16000,
	LanguageHints: []string{"zh", "en"},
	Heartbeat:     &heartbeat,
	Vocabulary:    map[string]int{"DashScope": 5},
})
if err != nil {
	return err
}
defer session.Close()

receiveDone := make(chan error, 1)
go func() {
	for {
		event, err := session.Receive(ctx)
		if err != nil {
			receiveDone <- err
			return
		}
		if event.Error != nil {
			receiveDone <- event.Error
			return
		}
		if event.Transcript != "" {
			fmt.Printf("text=%q final=%t\n", event.Transcript, event.Final)
		}
		if event.Terminal {
			receiveDone <- nil
			return
		}
	}
}()

// microphoneChunks 中每个 []byte 是一帧实时音频，例如 100ms 的
// 16kHz/16-bit/单声道 PCM 为 3200 字节。
for chunk := range microphoneChunks {
	if err := session.SendAudio(ctx, chunk); err != nil {
		return err
	}
}
if err := session.Finish(ctx); err != nil {
	return err
}
if err := <-receiveDone; err != nil {
	return err
}
```

Qwen3-ASR-Realtime 使用同一个接口。VAD 模式可设置 `TurnDetection`；Manual 模式应在每段语音后调用 `Commit`：

```go
session, err := c.StartRealtimeTranscription(ctx, spec.RealtimeTranscriptionRequest{
	Model:      "qwen3-asr-flash-realtime",
	Format:     "pcm", // Qwen3 推荐 pcm 或 opus
	SampleRate: 16000,
	Language:   "zh", // 留空时自动检测
	Manual:     true,
})
if err != nil {
	return err
}
defer session.Close()

_ = session.SendAudio(ctx, audioChunk)
_ = session.Commit(ctx) // Manual 模式提交当前语句；VAD 模式不调用
_ = session.Finish(ctx)
```

`RealtimeTranscriptionEvent.Type` 保留服务端原始事件名，`Raw` 保留完整 JSON；`Transcript`/`Final` 提供统一文本视图，`Terminal` 标记整个任务或会话已经结束。任务协议还会解析 `Sentence` 的句级、字级时间戳及 `Usage`，Qwen3 事件会解析 `StableText`、`Stash`、`Language` 和 `Emotion`。`Parameters`、`Input`、`Session` 与 `TurnDetection.ExtraFields` 可透传后续新增字段。

## OpenAI Chat Completions 与 Responses API

普通 `Send`/`SendParts` 会继续按配置路径选择 Chat Completions 或 Responses。`SendResponse`、`CreateResponse` 以及 Responses 生命周期方法会显式调用 Responses API，因此 `APIURL` 可以填写 API 根路径、`/chat/completions` 或 `/responses`，客户端会安全派生出正确端点。下面使用 OpenAI 官方端点；兼容服务只需替换 `APIURL`：

```go
c, err := client.New(llm.Config{
	Provider: "openai",
	Model:    "gpt-5.6-sol",
	APIKey:   os.Getenv("OPENAI_API_KEY"),
	APIURL:   "https://api.openai.com/v1/responses",
	// 默认 10 分钟；长时间联网或推理任务可以自定义。
	Timeout: 15 * time.Minute,
	// 可选：none、minimal、low、medium、high、xhigh、max；具体可用等级由模型决定。
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

// 本地图片：读取文件并转换为 data:image/...;base64 URL。
// 不要把 "img.png" 直接传给 NewImageURLPartWithDetail；服务端无法访问本地路径。
localImage, err := spec.NewImageFilePart("img.png", "image/png")
if err != nil {
	return err
}
resp, err = c.SendPartsNoHistory(
	context.Background(),
	localImage,
	spec.NewTextPart("描述这张本地图片"),
)

// 流式文本输出
resp, err = c.SendStreamNoHistory(context.Background(), "写一首短诗", func(_ context.Context, chunk string) error {
	fmt.Print(chunk)
	return nil
})
```

图片流式理解可使用 `SendStreamParts`；最终完整文本仍会写入 `resp.Message.Content`。协议级完整结果分别位于 `resp.ChatCompletion` 和 `resp.Responses`。非流式原始 JSON 位于 `resp.RawResponse`；流式原始 chunk/事件位于 `StreamEvent.Raw`。

### 多模态输入构造器

`ContentPart` 会按目标协议自动转换为 Chat Completions content part 或 Responses input content。URL 构造器只接受服务端能够访问的完整 URL 或 data URL；本地路径请使用带 `File`/`Local` 的构造器。

| 输入 | 构造器 | 说明 |
| --- | --- | --- |
| 文本 | `NewTextPart` | 普通文本输入 |
| 图片 URL | `NewImageURLPart`、`NewImageURLPartWithDetail` | 公网 URL 或 `data:image/...;base64,...`；支持设置 detail |
| 图片 Base64/内存 | `NewImageBase64Part`、`NewImageBytesPart` | 自动生成图片 data URL |
| 本地图片 | `NewImageFilePart` | 读取本地文件并生成图片 data URL |
| 已上传图片 | `NewInputImageFileIDPart` | Responses `input_image.file_id` |
| 音频 | `NewInputAudioPart`、`NewInputAudioBytesPart` | Responses 内联 Base64 音频 |
| 文档 URL | `NewInputFilePart` | Responses `input_file.file_url` |
| 文档 Base64/内存 | `NewInputFileBase64Part`、`NewInputFileBytesPart` | 生成文档 data URL |
| 本地文档 | `NewInputFileLocalPart` | 读取本地文件并生成文档 data URL |
| 已上传文档 | `NewInputFileIDPart` | Responses `input_file.file_id` |

### OpenAI 联网搜索（Responses 与 Chat Completions）

`SendWebSearch` 会根据 `APIURL` 自动选择协议：配置 `/responses` 时追加 Responses `web_search` 工具；配置 `/v1/chat/completions`（或使用 OpenAI provider 默认地址）时发送 `web_search_options`：

```go
resp, err := c.SendWebSearch(
	context.Background(),
	"查询今天的重要 AI 新闻，并标注来源",
	spec.WebSearchConfig{
		SearchContextSize: spec.WebSearchContextSizeMedium,
		UserLocation: &spec.WebSearchUserLocation{
			Country:  "CN",
			City:     "Shanghai",
			Timezone: "Asia/Shanghai",
		},
	},
)

// 显式 Responses 多模态输入；即使 APIURL 配成 /chat/completions 也会调用 /responses。
resp, err = c.SendResponse(context.Background(), []spec.ContentPart{
	spec.NewTextPart("总结图片、音频和文档"),
	spec.NewInputImageFileIDPart("file_image"),
	spec.NewInputAudioBytesPart("wav", audioBytes),
	spec.NewInputFileIDPart("file_document"),
})
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

Responses 模式还支持 `ReturnTokenBudget`、`Filters`、`SearchContentTypes`、`ImageSettings`、`IncludeSources`、`IncludeResults` 和 `ToolChoice`；Chat Completions 模式使用 `SearchContextSize`、`UserLocation` 等 `web_search_options` 字段。两种模式返回的 `url_citation` 都会解析到 `resp.Citations`；Responses 搜索调用还会解析到 `resp.WebSearchCalls`。

`WithWebSearch` 会根据端点选择搜索协议：`/responses` 使用 Responses `web_search` 工具，`/v1/chat/completions` 使用 Chat Completions `web_search_options`。第三方兼容服务是否真正执行搜索仍取决于服务商；有些网关会把查询词和来源降级追加到文本中。通过 `OPENAI_TEST_API_KEY`、`OPENAI_TEST_BASE_URL` 和 `OPENAI_TEST_MODEL` 环境变量配置后，可运行对应集成测试实测。不要把真实 API Key 写入测试文件或提交到仓库。

### Responses 连续对话

`SendResponse` 直接接受 Responses API 的 `input`。`ContinueResponse` 会发送 `previous_response_id`，不会重复发送本地聊天历史：

```go
first, err := c.SendResponse(ctx, "我叫小明")
if err != nil {
	return err
}

next, err := c.ContinueResponse(ctx, first.ID, "我叫什么？")
```

完整请求可以直接使用 `spec.ResponseCreateRequest`。稳定字段提供强类型，`ExtraFields` 可立即使用 OpenAI 新增但库尚未命名的字段：

```go
resp, err := c.CreateResponse(ctx, spec.ResponseCreateRequest{
	Model: "gpt-5.6-sol",
	Input: "提取订单号和金额，并在需要时调用查询函数",
	Tools: []any{
		spec.NewFunctionTool("lookup_order", "查询订单", orderSchema, true),
		spec.NewWebSearchTool(),
	},
	Text: &spec.ResponseTextConfig{
		Format: spec.NewResponseJSONSchemaFormat("order", outputSchema),
	},
	Reasoning: &spec.ResponseReasoningConfig{
		Effort:  spec.ReasoningEffortHigh,
		Summary: "auto",
	},
	Background: spec.Bool(true),
	Store:      spec.Bool(true),
})
```

`ResponseCreateRequest` 已提供下列稳定字段；OpenAI 后续新增字段可先放入 `ExtraFields`：

| 分类 | 字段 |
| --- | --- |
| 模型与输入 | `Model`、`Input`、`Instructions` |
| 状态与会话 | `PreviousResponseID`、`Conversation`、`Store`、`Metadata` |
| 返回内容 | `Include`、`Text`、`Reasoning`、`MaxOutputTokens` |
| 工具 | `Tools`、`ToolChoice`、`ParallelToolCalls`、`MaxToolCalls` |
| 采样 | `Temperature`、`TopP`、`TopLogprobs` |
| 流式与后台 | `Stream`、`StreamOptions`、`Background` |
| 服务与缓存 | `ServiceTier`、`PromptCacheKey`、`PromptCacheOptions`、`PromptCacheRetention` |
| Prompt 模板 | `Prompt` |
| 上下文与截断 | `ContextManagement`、`Truncation` |
| 安全与审核 | `SafetyIdentifier`、`Moderation`、`User` |
| 前向兼容 | `ExtraFields` |

### Responses 工具与工具结果

库为当前已接入的 Responses 工具提供了构造器。工具是否可用仍取决于具体模型、账号和兼容服务：

| 工具 | 构造器 |
| --- | --- |
| Function Calling | `NewFunctionTool` |
| Web Search | `NewWebSearchTool` |
| File Search | `NewFileSearchTool` |
| Image Generation | `NewImageGenerationTool` |
| Code Interpreter | `NewCodeInterpreterTool` |
| Remote MCP | `NewMCPTool` |
| Computer Use | `NewComputerTool` |
| Custom Tool | `NewCustomTool` |
| Local Shell / Hosted Shell | `NewLocalShellTool`、`NewShellTool` |
| Apply Patch | `NewApplyPatchTool` |
| Tool Search | `NewToolSearchTool` |
| Programmatic Tool Calling | `NewProgrammaticToolCallingTool` |
| Namespace | `NewNamespaceTool` |

工具选择可以使用 `RequiredToolChoice`、`AutoToolChoice`、`NoneToolChoice`、`ToolChoice` 和 `NamedFunctionToolChoice`。不在强类型构造器中的新工具仍可通过 `spec.ResponseTool.ExtraFields` 或 `map[string]any` 立即使用。

工具执行结果和控制项可以通过以下 input item 回传：

| 类型 | 构造器 |
| --- | --- |
| Function 结果 | `NewFunctionCallOutput` |
| 任意 call output | `NewToolCallOutput` |
| Computer 结果 | `NewComputerCallOutput` |
| Custom Tool 结果 | `NewCustomToolCallOutput` |
| Shell 结果 | `NewShellCallOutput`、`NewLocalShellCallOutput` |
| Apply Patch 结果 | `NewApplyPatchCallOutput` |
| MCP 审批 | `NewMCPApprovalResponse` |
| 引用既有 item | `NewItemReference` |

### 图像生成

```go
tool := spec.NewImageGenerationTool()
tool.Size = "1536x1024"
tool.Quality = "high"

resp, err := c.GenerateImage(ctx, "雨夜中的未来上海", tool)
images := resp.Responses.ImageGenerationResults() // base64 图片列表
```

`SendText2Image` 在 OpenAI provider 下也会自动改用 Responses `image_generation` 工具，其他 provider 保持原行为。

### 独立 Images API：生成与编辑

`CreateImage` 和 `EditImage` 不经过 Responses，分别直连 `/v1/images/generations` 与 `/v1/images/edits`。`APIURL` 可以配置为 API 根路径、`/chat/completions`、`/responses` 或 Images 端点，客户端会派生出正确路径。

```go
c, err := client.New(llm.Config{
	Provider: "openai",
	Model:    "gpt-image-2",
	APIKey:   os.Getenv("POKE_API_KEY"),
	APIURL:   "https://www.poke2api.com/v1",
})
if err != nil {
	return err
}

generated, err := c.CreateImage(ctx, spec.ImageGenerationRequest{
	Prompt:         "一座位于雪山脚下的未来观测站，夜空可见银河，电影感写实摄影，无文字，无人物",
	Size:           "1024x1024",
	Quality:        "high",
	ResponseFormat: "b64_json",
})
if err != nil {
	return err
}
images := generated.Base64Images()

reference, err := spec.NewImageFile("reference.png", "image/png")
if err != nil {
	return err
}
edited, err := c.EditImage(ctx, spec.ImageEditRequest{
	Prompt:         "保留主体，把背景改为克制的黑白编辑风格，并加入少量深蓝信号色",
	Size:           "1024x1024",
	Quality:        "high",
	ResponseFormat: "b64_json",
	Image:          reference,
})
if err != nil {
	return err
}
editedImages := edited.Base64Images()
```

请求中省略 `Model` 时会使用 `llm.Config.Model`。已有内存图片可以通过 `spec.NewImageFileBytes` 传入；需要 mask 时设置 `ImageEditRequest.Mask`。底层无状态用法可使用 `openai.NewImagesClient` 获取 `spec.ImagesClient`。

#### 真实图片生成测试

根目录的 `openai_llm_test.go` 包含 `TestOpenAIImageGeneration`。该测试会真实调用 Images API，优先解码 `b64_json`；兼容网关若返回 `url`，测试会下载图片。解码后还会校验 MIME 类型，并把图片保留在本地。PowerShell 运行方式：

```powershell
$env:OPENAI_TEST_API_KEY = $env:POKE_API_KEY
$env:OPENAI_TEST_BASE_URL = "https://www.poke2api.com/v1"
$env:OPENAI_TEST_IMAGE_MODEL = "gpt-image-2"
$env:OPENAI_TEST_IMAGE_OUTPUT = "openai-image-generation-test.png"
go test -run '^TestOpenAIImageGeneration$' -count=1 -v .
```

还可以通过 `OPENAI_TEST_IMAGES_URL` 单独覆盖生成端点。测试使用无人物、无角色 IP 的雪山未来观测站提示词；带有受保护角色的提示词可能被兼容网关以 `content_policy_violation` 拦截。

2026-08-09 使用 POKE 兼容端点实测通过：耗时 23.17 秒，响应返回 `b64_json`，解码得到 2,835,660 字节的 PNG，输出到 `openai-image-generation-test.png`。请求指定 `1024x1024/high`，但该网关响应及实际文件为 `1536x1024`、`quality=auto`，说明兼容服务可能会归一化或忽略部分 Images 参数。

### 后台任务、续流与生命周期

```go
// 查询后台响应。
resp, err := c.RetrieveResponse(ctx, "resp_xxx", spec.ResponseRetrieveOptions{})

// 从最后收到的 sequence_number 继续 SSE。
resp, err = c.RetrieveResponse(ctx, "resp_xxx", spec.ResponseRetrieveOptions{
	Stream:        true,
	StartingAfter: 128,
}, spec.WithEventCallback(handleEvent))

// 取消、删除、查看原始输入、预估 token、压缩长上下文。
_, _ = c.CancelResponse(ctx, "resp_xxx")
_, _ = c.DeleteResponse(ctx, "resp_xxx")
items, _ := c.ListResponseInputItems(ctx, "resp_xxx", spec.ResponseInputItemsOptions{})
tokens, _ := c.CountResponseInputTokens(ctx, spec.ResponseInputTokenCountRequest{Input: "hello"})
compact, _ := c.CompactResponse(ctx, spec.ResponseCompactRequest{Input: priorItems})
```

同时支持 conversations 的创建、查询、更新、删除，以及 conversation items 的创建、分页、查询和删除。底层用户也可以使用 `openai.NewResponsesClient` 直接获得 `spec.ResponsesClient`。

需要在一条持久连接上连续创建多个响应时，可使用官方 Responses WebSocket：

```go
socket, err := c.ConnectResponseWebSocket(ctx)
if err != nil {
	return err
}
defer socket.Close()

err = socket.CreateResponse(ctx, spec.ResponseCreateRequest{
	Model: "gpt-5.6-sol",
	Input: "写一句问候语",
})
for {
	event, err := socket.Receive(ctx)
	if err != nil {
		return err
	}
	fmt.Println(event.Type, string(event.Raw))
	if event.Type == "response.completed" {
		break
	}
}
```

连接允许一个 reader 与一个 writer 并发；`SendEvent` 也可直接发送后续新增的客户端事件。

| WebSocket 方法 | 说明 |
| --- | --- |
| `CreateResponse` | 发送强类型 `response.create` 客户端事件 |
| `SendEvent` | 发送任意后续客户端事件，便于前向兼容 |
| `Receive` | 接收下一个服务端事件，并保留完整 `StreamEvent.Raw` |
| `Close` | 关闭连接 |

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

Chat Completions 可使用 `NewSystemMessage`、`NewUserMessage`、`NewAssistantMessage`、`NewUserPartsMessage` 构造消息，工具结果使用 `NewToolMessage` 追加到 `messages`。

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

`llm.Config.Parameters` 会原样透传两种 API 的扩展参数；Responses 模式下 `max_tokens` 会自动转换为 `max_output_tokens`。每个 `ResponseOutputItem`、内容 part 和 SSE 事件都保留 `Raw`，未知字段可以无损读取并重新发送；完整 HTTP JSON 仍位于 `RawResponse`。

### 请求 Options 与底层客户端配置

| 分类 | Options |
| --- | --- |
| 模型与采样 | `WithModel`、`WithTemperature`、`WithMaxTokens`、`WithTopP` |
| 推理 | `WithThinking`、`WithReasoningEffort` |
| 流式回调 | `WithStreaming`、`WithStreamCallback`、`WithReasoningCallback`、`WithEventCallback` |
| Responses 输入与状态 | `WithResponseInput`、`WithInstructions`、`WithPreviousResponseID` |
| 模型托管搜索 | `WithWebSearch`（Responses、Chat Completions、ZHIPU） |
| 扩展字段 | `WithParameters`、`WithParameter`、`WithProvider` |
| 翻译 | `WithTranslation` |
| 文生图请求 | `WithText2Image`、`WithText2ImageParameters` |
| 文生图参数 | `WithText2ImageSize`、`WithText2ImageWatermark`、`WithText2ImageNegativePrompt`、`WithText2ImagePromptExtend`、`WithText2ImageCount` |

直接创建底层 OpenAI client 时，可使用 `WithAPIKey`、`WithAPIURL`、`WithHTTPClient` 和 `WithTimeout`；`NewClientConfig` 与 `NewRequestConfig` 可分别创建底层客户端配置和单次请求配置。可选布尔字段可以使用 `Bool`，结构化输出可用 `NewResponseJSONSchemaFormat` 快速构造严格的 `json_schema` format。

> 第三方兼容服务（例如 `poke2api.com`）不一定实现 `previous_response_id`、MCP、hosted tools、文件或全部事件类型；本库支持上述 OpenAI 协议结构，但实际能力仍以服务商为准。

OpenAI 官方参考：[Responses API](https://developers.openai.com/api/docs/api-reference/responses)、[Images and vision](https://developers.openai.com/api/docs/guides/images-vision)、[GPT Image 2](https://developers.openai.com/api/docs/models/gpt-image-2)。

需要绕过有状态 `client.Client` 时，可以直接使用 `openai.NewClient` 获得通用 Chat client，使用 `openai.NewResponsesClient` 获得完整 `spec.ResponsesClient`，或使用 `openai.NewImagesClient` 获得 `spec.ImagesClient`。

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
	resp, err := c.SendOCR(
		context.Background(),
		localPart,
		"document_parsing",
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

使用 `client.New(llm.Config{...})` 创建客户端。下面列出 `Client` 当前全部公开方法。

#### 对话、历史与多模态

| 方法 | 说明 |
| --- | --- |
| `Send` | 发送文本、携带并记录历史，返回完整响应 |
| `SendStream` | 发送文本、携带并记录历史，通过回调接收文本增量 |
| `SendNoHistory` | 携带已有历史，但不记录本轮输入和输出 |
| `SendStreamNoHistory` | 不携带且不记录历史的流式单次请求 |
| `SendParts` | 发送并记录由文本、图片等组成的多模态 parts |
| `SendStreamParts` | 发送并记录多模态 parts，流式接收文本 |
| `SendPartsNoHistory` | 不携带且不记录历史的多模态请求 |
| `SendImageURL` | 图片 URL 问答快捷方法，记录历史 |
| `SendImageBase64` | Base64 图片问答快捷方法，记录历史 |
| `SendText` | `Send` 的文本快捷封装；失败时返回兼容性的错误文案 |
| `SendByMemory` | 将外部 memory JSON 与用户输入组合后发送 |
| `ResetHistory` | 清空历史并重新放入系统提示词 |
| `GetHistory` | 返回当前对话历史 |

#### Responses 创建与高级能力

| 方法 | 说明 |
| --- | --- |
| `SendResponse` | 直接发送 Responses `input`，不修改本地聊天历史 |
| `CreateResponse` | 使用完整 `ResponseCreateRequest` 创建响应 |
| `ContinueResponse` | 使用 `previous_response_id` 延续响应 |
| `SendWebSearch` | 按配置端点使用 Responses 或 Chat Completions 联网搜索 |
| `GenerateImage` | 使用 Responses `image_generation` 工具生成图片 |
| `ConnectResponseWebSocket` | 建立持久 Responses WebSocket 连接 |

#### Responses 生命周期

| 方法 | 说明 |
| --- | --- |
| `RetrieveResponse` | 获取响应；支持流式续传及 `starting_after` |
| `DeleteResponse` | 删除已存储响应 |
| `CancelResponse` | 取消后台响应 |
| `ListResponseInputItems` | 分页读取响应的原始输入 items |
| `CountResponseInputTokens` | 在创建响应前估算输入 token |
| `CompactResponse` | 压缩长上下文并返回 compaction items |

#### Images API

| 方法 | 说明 |
| --- | --- |
| `CreateImage` | 调用独立的 `/images/generations` JSON 接口 |
| `EditImage` | 调用独立的 `/images/edits` multipart 接口 |

#### Conversations 与 Conversation Items

| 方法 | 说明 |
| --- | --- |
| `CreateConversation` | 创建持久 conversation |
| `RetrieveConversation` | 查询 conversation |
| `UpdateConversation` | 更新 conversation metadata |
| `DeleteConversation` | 删除 conversation |
| `CreateConversationItems` | 向 conversation 追加 items |
| `ListConversationItems` | 分页读取 conversation items |
| `RetrieveConversationItem` | 查询单个 conversation item |
| `DeleteConversationItem` | 删除单个 conversation item |

#### 其他模型能力

| 方法 | 说明 |
| --- | --- |
| `StreamRealtimeTranscription` | 从 `io.Reader` 自动分块发送音频、回调转写结果并管理完整会话生命周期 |
| `StartRealtimeTranscription` | 建立 DashScope 双向实时语音识别会话并流式收发音频/转写事件 |
| `SendEmbedding` | 生成单条或批量文本向量；需要 provider 实现 Embedding |
| `SearchWeb` | 调用 provider 的独立联网搜索 API（目前支持 ZHIPU） |
| `SendText2Image` | DashScope 使用异步文生图；OpenAI 使用 Responses 图像生成工具 |
| `SendOCR` | 使用自定义 file part 和 OCR 任务参数识别文档 |
| `SendOCRURL` | OCR 公网文档 URL |
| `SendOCRLocal` | 读取本地文档并编码后执行 OCR |
| `SendOCRBytes` | 使用内存字节执行 OCR |

### `llm.Config` 配置项

| 字段 | 说明 |
| --- | --- |
| `Provider` | 厂商标识: `dashscope`, `openai`, `zhipu`, `deepseek`, `openrouter`, `generic` |
| `Model` | 模型名称: `qwen-plus`, `gpt-4o`, `qwen-image-plus` 等 |
| `APIKey` | API 密钥 |
| `APIURL` | (可选) 自定义接口地址，用于代理或私有部署 |
| `Timeout` | (可选) 完整 HTTP 请求超时，默认 10 分钟，例如 `15*time.Minute` |
| `SystemPrompt` | (可选) 系统预设人设；Responses 下可作为默认 instructions |
| `Thinking` | (可选) `llm.Thinking()` 开启思考模式适配 |
| `ReasoningEffort` | (可选) 思考等级，如 `llm.ReasoningEffortLow`、`llm.ReasoningEffortMedium`、`llm.ReasoningEffortHigh` |
| `Parameters` | 两种协议的扩展请求字段；Responses 自动把 `max_tokens` 映射为 `max_output_tokens` |
| `ResponseInput` | Responses 模式的默认 `input` |
| `Instructions` | Responses 模式的默认 instructions，可为字符串或结构化输入 |
| `PreviousResponseID` | Responses 连续对话的默认前一响应 ID |
| `WebSearch` | 模型托管联网搜索配置（Responses、Chat Completions、ZHIPU） |
| `Translation` | 翻译任务配置 |
| `StreamCallback` | 文本输出增量回调 |
| `ReasoningCallback` | 推理摘要增量回调 |
| `EventCallback` | 所有 Responses SSE 事件或 Chat stream chunk 回调 |
| `Text2Image` | 开启文生图模式 |
| `ImageEdit` | 开启图片编辑模式（由对应 provider 实现） |
| `WebExtractor` | 网页抓取、联网搜索和代码解释器组合配置 |
| `ProviderOpts` | provider 专属扩展配置 |

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

| 函数 | 说明 |
| --- | --- |
| `GetClient` | 根据 `llm.Config` 创建底层 provider client |
| `ChatMessages` | 使用完整 `[]spec.Message` 发起请求 |
| `Chat` | 使用单条用户文本发起请求并返回 `*spec.Response` |
| `ChatText` | 使用单条用户文本发起请求并只返回文本 |
| `Thinking` | 返回启用思考模式的 `*bool` |
| `NoThinking` | 返回关闭思考模式的 `*bool` |

```go
import "github.com/iEvan-lhr/go-llm-client/llm"

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
