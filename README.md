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
go get github.com/ievan-lhr/go-llm-client

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

## 📚 API 方法速查

### `client.Client` 方法

| 方法 | 说明 | 适用场景 |
| --- | --- | --- |
| **`Send`** | 发送消息，**记录历史**，等待完整回复 | 常规多轮非流式对话 |
| **`SendStream`** | 发送消息，**记录历史**，流式回调 | 常规多轮流式对话 (打字机) |
| **`SendText2Image`** | 发送提示词，触发 **文生图** 任务 | AI 画图、视觉生成 (返回图片URL) |
| **`SendNoHistory`** | 发送消息，**携带**历史但不记录本次 | 基于上下文的临时追问 |
| **`SendStreamNoHistory`** | 发送消息，**不携带**且不记录历史 | 独立的一次性任务 (如翻译/搜索) |
| **`ResetHistory`** | 清空对话历史 | 重置会话 |

### `llm.Config` 配置项

| 字段 | 说明 |
| --- | --- |
| `Provider` | 厂商标识: `dashscope`, `openai`, `generic` |
| `Model` | 模型名称: `qwen-plus`, `gpt-4o`, `qwen-image-plus` 等 |
| `APIKey` | API 密钥 |
| `APIURL` | (可选) 自定义接口地址，用于代理或私有部署 |
| `Thinking` | (可选) `llm.Thinking()` 开启思考模式适配 |
| `SystemPrompt` | (可选) 系统预设人设 |

## 💡 高级用法

### 开启 "Thinking" (思考/推理) 模式

适配 DeepSeek R1 或 Qwen 等具备推理能力的模型。

```go
c, _ := client.New(llm.Config{
    Provider: "generic", // 或 dashscope
    Model:    "deepseek-r1",
    APIKey:   "...",
    Thinking: llm.Thinking(), // 开启思考模式适配
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