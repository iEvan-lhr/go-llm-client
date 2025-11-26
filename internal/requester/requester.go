package requester

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Requester 封装了执行HTTP请求的通用逻辑。
type Requester struct {
	HTTPClient *http.Client
}

// Post 方法发送一个POST请求并返回原始响应体。
func (r *Requester) Post(ctx context.Context, url string, headers http.Header, requestBody any) ([]byte, error) {
	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("requester: failed to marshal request body: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("requester: failed to create request: %w", err)
	}

	// 设置请求头
	httpReq.Header = headers

	// 发送请求
	resp, err := r.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("requester: request failed: %w", err)
	}
	defer resp.Body.Close()

	// 读取响应体
	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("requester: failed to read response body: %w", err)
	}

	// 检查状态码
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("requester: API error (status %d): %s", resp.StatusCode, string(rawBody))
	}

	return rawBody, nil
}

// PostStream 发送请求并返回 http.Response，由调用方负责读取 Body 和关闭。
// 用于流式(SSE)场景。
func (r *Requester) PostStream(ctx context.Context, url string, headers http.Header, requestBody any) (*http.Response, error) {
	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("requester: failed to marshal request body: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("requester: failed to create request: %w", err)
	}

	httpReq.Header = headers

	resp, err := r.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("requester: request failed: %w", err)
	}

	// 注意：这里不读取也不关闭 Body，交给上层处理
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// 如果请求出错，尽力读取错误信息
		defer resp.Body.Close()
		rawBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("requester: API error (status %d): %s", resp.StatusCode, string(rawBody))
	}

	return resp, nil
}
