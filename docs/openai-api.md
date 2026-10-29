# OpenAI API 使用指南

`Provider: "openai"` 支持下列模型接口。模型名由调用方传入，不维护会过期的模型白名单；具体参数、工具和流式能力取决于所选模型。

| 能力 | 入口 |
| --- | --- |
| Chat Completions | `Send`、`SendStream`、底层 `Model.Chat` |
| Responses | `CreateResponse`、`SendResponse`、`ContinueResponse` |
| Responses 生命周期 | `RetrieveResponse`、`DeleteResponse`、`CancelResponse`、`ListResponseInputItems` |
| Responses 辅助接口 | `CountResponseInputTokens`、`CompactResponse`、`ConnectResponseWebSocket` |
| Conversations | conversation 和 items 的创建、读取、修改、删除、分页 |
| Responses 生图工具 | `GenerateImage`，或 `ResponseCreateRequest.Tools` 中的 `image_generation` |
| Images | `CreateImage`、`EditImage`、`CreateImageVariation`、`CreateImageStream`、`EditImageStream` |
| Embeddings | `SendEmbedding`、`CreateEmbedding` |
| Models / Moderations | `ListModels`、`RetrieveModel`、`CreateModeration` |
| Audio | `CreateSpeech`、`CreateSpeechStream`、`CreateTranscription`、`CreateTranscriptionStream`、`CreateTranslation` |

这是模型调用接口集。Files、Batch、Vector Stores 和微调等平台管理接口不在本次接口集内。

## 创建客户端

```go
import (
    "context"
    "encoding/base64"
    "errors"
    "fmt"
    "os"

    "github.com/iEvan-lhr/go-llm-client/client"
    "github.com/iEvan-lhr/go-llm-client/llm"
    "github.com/iEvan-lhr/go-llm-client/providers/openai"
    "github.com/iEvan-lhr/go-llm-client/spec"
)

c, err := client.New(llm.Config{
    Provider: "openai", APIKey: os.Getenv("OPENAI_API_KEY"),
    Model: os.Getenv("OPENAI_MODEL"),
    APIURL: "https://api.openai.com/v1",
    // Organization: "org_...", Project: "proj_...",
})
if err != nil { return err }
ctx := context.Background()
```

已有 `client.Client` 接口保留。底层调用可以使用 `openai.NewAPIClient(spec.WithAPIKey(...))` 返回的 `spec.OpenAIClient`，一次获得所有上述接口；底层每次请求自行填写模型。专用构造器有 `NewResponsesClient`、`NewImagesAPIClient`，旧的 `NewImagesClient` 保持原接口。

APIURL 可为 `/v1` 根路径或已有的 `/chat/completions`、`/responses`、Images、Audio、Embeddings 端点；跨资源调用会替换端点后缀，保留代理前缀与查询参数。普通聊天默认采用 Chat Completions，显式 `/responses` 地址采用 Responses。

## Responses 与工具调用

```go
response, err := c.CreateResponse(ctx, spec.ResponseCreateRequest{
    Input: "查一下北京天气",
    Tools: []any{spec.NewFunctionTool("weather", "查询城市天气", map[string]any{
        "type": "object",
        "properties": map[string]any{"city": map[string]any{"type": "string"}},
        "required": []string{"city"}, "additionalProperties": false,
    }, true)},
})
if err != nil { return err }

var outputs []spec.ResponseInputItem
for _, call := range response.Message.ToolCalls {
    // 应用执行 call.Function.Name / Arguments 对应的函数。
    outputs = append(outputs, spec.NewFunctionCallOutput(call.ID, `{"temperature":20}`))
}
if len(outputs) > 0 {
    response, err = c.ContinueResponse(ctx, response.ID, outputs)
    if err != nil { return err }
}
```

普通消息历史中的 `ToolCalls` 和 `NewToolMessage` 也会转换成 `function_call` / `function_call_output`。从 Responses 收到的助手消息通过 `ResponsesOutput` 保存原生 reasoning、函数调用和 hosted tool items，下一轮原样回传；手动修改消息时先清空 `ResponsesOutput`。

`ResponseCreateRequest` 支持结构化输出、reasoning、include、background、store、prompt、conversation、上下文压缩及工具设置。新增字段可放在 `ExtraFields`，新增工具可直接用 `map[string]any`；未知输出字段保留在 `Raw` / `RawResponse`。

默认参数的优先级是：客户端 `Parameters` → 请求 `ExtraFields` → 请求中的已设置强类型字段 → 显式 `spec.Option`。`Tools: []any{}` 可清空默认工具；`Store: spec.Bool(false)` 可显式关闭存储。配置流式回调会启用流式请求。

后台请求可用 `Background: spec.Bool(true)` 创建，再按 ID 查询或取消。续传使用 `ResponseRetrieveOptions{Stream: true, StartingAfter: n}`；若要显式从序号 0 后续传，设置 `StartingAfterSequence: &zero`。最终响应保留完整文本，回调只收到续传部分。连接中断且没有终止事件时，可通过 `errors.Is(err, io.ErrUnexpectedEOF)` 识别。

## 两种生图入口

`GenerateImage` 使用文本模型的 Responses `image_generation` 工具；`CreateImage` 直接使用图像模型。两者的模型字段不要混用。

```go
// c 的默认模型应为支持图像工具的文本模型。
response, err := c.GenerateImage(ctx, "画一只橘猫", spec.ResponseTool{
    Size: "1024x1024", Quality: "high", OutputFormat: "png", PartialImages: 1,
}, spec.WithEventCallback(func(_ context.Context, event spec.StreamEvent) error {
    if event.PartialImageB64 != "" { fmt.Println("收到预览", event.PartialImageIndex) }
    return nil
}))
if err != nil { return err }
images := response.Responses.ImageGenerationResults()
```

`ResponseTool` 还支持 `Action`、`InputFidelity`、`InputImageMask`；可在 `CreateResponse` 中结合图像输入使用。若需要发送 `output_compression: 0`，现有工具结构的整数字段有 `omitempty`，请用 `ExtraFields: map[string]any{"output_compression": 0}`。

```go
images, err := c.CreateImageStream(ctx, spec.ImageGenerationRequest{
    Model: os.Getenv("OPENAI_IMAGE_MODEL"),
    Prompt: "雪山中的未来观测站", OutputFormat: "png", PartialImages: 2,
}, func(_ context.Context, event spec.ImageStreamEvent) error {
    fmt.Println(event.Type, event.PartialImageIndex)
    return nil
})
if err != nil { return err }
if len(images.Data) == 0 { return fmt.Errorf("no image returned") }
data, err := base64.StdEncoding.DecodeString(images.Data[0].Base64JSON)
if err != nil { return err }
if err := os.WriteFile("generated.png", data, 0600); err != nil { return err }
```

独立图片事件的 base64 字段为 `Base64JSON`，Responses 工具预览的字段为 `PartialImageB64`。流式方法返回的 `ImageResponse.Data` 只收集完成图片，`RawResponse` 是聚合结果的 JSON；每个原始事件通过回调的 `Raw` 获取。回调返回 error 会停止接收并关闭响应体。

编辑使用 `ImageEditRequest{Image: file}`；多图使用 `Images: []spec.ImageFile{a, b}`，两者互斥。`Mask` 可指定透明蒙版，`EditImageStream` 提供流式编辑。`NewImageFile` 读取磁盘文件，`NewImageFileBytes` 复制内存数据。独立 Images 请求的 `OutputCompression` 是指针，可显式传入零。

`CreateImageVariation` 面向支持 `/images/variations` 的模型，例如 `dall-e-2`。图像模型对尺寸、数量、透明背景、流式和编辑输入有不同要求，由服务端校验。GPT Image 模型使用 base64 输出，不需要 `ResponseFormat: "b64_json"`。

## 向量、模型与审核

```go
dimensions := 512
embedding, err := c.CreateEmbedding(ctx, spec.EmbeddingRequest{
    Model: "text-embedding-3-small", Input: []string{"你好", "hello"},
    Dimensions: &dimensions, EncodingFormat: "base64",
})
// 无论 float 或 base64，Data[i].Embedding 都是解码后的 []float32。
// base64 原文保存在 Base64Embedding，响应原文保存在 RawResponse。

models, err := c.ListModels(ctx)
model, err := c.RetrieveModel(ctx, "text-embedding-3-small")
moderation, err := c.CreateModeration(ctx, spec.ModerationRequest{
    Model: "omni-moderation-latest", Input: "待审核内容",
})
```

`SendEmbedding` 也已支持 OpenAI；其模型使用 `llm.Config.Model`。审核输入可以是字符串、字符串数组或 `[]spec.ModerationInput`（text/image_url）；分类和分数使用 map 保留新的审核类别。`CreateModeration` 省略模型时使用服务端默认审核模型，不继承聊天模型。

## 音频

```go
speech, err := c.CreateSpeech(ctx, spec.SpeechRequest{
    Model: "gpt-4o-mini-tts", Input: "你好，欢迎使用。", Voice: "alloy", ResponseFormat: "mp3",
})
if err != nil { return err }
if err := os.WriteFile("speech.mp3", speech.Data, 0600); err != nil { return err }

file, err := spec.NewAudioFile("meeting.wav", "audio/wav")
if err != nil { return err }
transcript, err := c.CreateTranscription(ctx, spec.TranscriptionRequest{
    Model: "gpt-4o-transcribe", File: file, Language: "zh",
})
translation, err := c.CreateTranslation(ctx, spec.AudioTranslationRequest{
    Model: "whisper-1", File: file, ResponseFormat: "srt",
})
```

`CreateSpeechStream` 返回原始音频流，调用方必须关闭 `Body`；如通过 `ExtraFields` 请求 SSE 格式，则 Body 为原始 SSE 字节。`CreateTranscriptionStream` 使用 `TranscriptionEventCallback` 接收 delta、segment、done 以及原始事件，返回聚合文本及 usage。转写支持时间戳、include、chunking_strategy 和说话人参数；它们的可用性由模型决定。JSON 响应解析为结构体，text/SRT/VTT 完整保存在 `Text` 和 `RawResponse`。

## 错误与验证

```go
var apiError *spec.HTTPError
if errors.As(err, &apiError) {
    fmt.Println(apiError.StatusCode, apiError.RequestID, apiError.Header.Get("Retry-After"))
    if apiError.APIError != nil { fmt.Println(apiError.APIError.Code, apiError.APIError.Message) }
}
```

HTTP 非 2xx 错误保留响应体、请求 ID 和头部；`errors.As` 也可匹配 `*spec.APIError`。context 取消、超时及回调错误会向上传递。库不自动重试生成请求，以免创建重复响应或图片。

本地协议测试使用 `httptest`，验证请求路径、认证头、JSON/multipart、流式事件、工具历史及错误行为：

```sh
go test ./providers/openai ./client ./spec ./llm ./internal/...
go test ./... -run '^$'
go vet ./...
```

仓库部分其他 provider 和根目录测试会真实调用外部服务，需要各自的密钥与网络。官方参考入口：[Responses](https://platform.openai.com/docs/api-reference/responses)、[Images](https://platform.openai.com/docs/api-reference/images)、[Audio](https://platform.openai.com/docs/api-reference/audio)、[Embeddings](https://platform.openai.com/docs/api-reference/embeddings)。
