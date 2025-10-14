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
