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
	return r.Do(ctx, http.MethodPost, url, headers, requestBody)
}

// Get sends a GET request and returns the complete response body.
func (r *Requester) Get(ctx context.Context, url string, headers http.Header) ([]byte, error) {
	return r.Do(ctx, http.MethodGet, url, headers, nil)
}

// Delete sends a DELETE request and returns the complete response body.
func (r *Requester) Delete(ctx context.Context, url string, headers http.Header) ([]byte, error) {
	return r.Do(ctx, http.MethodDelete, url, headers, nil)
}

// Patch sends a PATCH request and returns the complete response body.
func (r *Requester) Patch(ctx context.Context, url string, headers http.Header, requestBody any) ([]byte, error) {
	return r.Do(ctx, http.MethodPatch, url, headers, requestBody)
}

// Do executes a JSON request. A nil requestBody produces an empty body.
func (r *Requester) Do(ctx context.Context, method, url string, headers http.Header, requestBody any) ([]byte, error) {
	var body io.Reader
	if requestBody != nil {
		jsonBody, err := json.Marshal(requestBody)
		if err != nil {
			return nil, fmt.Errorf("requester: failed to marshal request body: %w", err)
		}
		body = bytes.NewReader(jsonBody)
	}

	return r.DoBody(ctx, method, url, headers, body)
}

// DoBody executes a request whose body and content type have already been
// prepared by the caller, for example a multipart/form-data request.
func (r *Requester) DoBody(ctx context.Context, method, url string, headers http.Header, body io.Reader) ([]byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("requester: failed to create request: %w", err)
	}
	httpReq.Header = headers.Clone()

	resp, err := r.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("requester: request failed: %w", err)
	}
	defer resp.Body.Close()

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("requester: failed to read response body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("requester: API error (status %d): %s", resp.StatusCode, string(rawBody))
	}
	return rawBody, nil
}

// PostStream 发送请求并返回 http.Response，由调用方负责读取 Body 和关闭。
// 用于流式(SSE)场景。
func (r *Requester) PostStream(ctx context.Context, url string, headers http.Header, requestBody any) (*http.Response, error) {
	return r.DoStream(ctx, http.MethodPost, url, headers, requestBody)
}

// GetStream sends a streaming GET request. The caller owns the response body.
func (r *Requester) GetStream(ctx context.Context, url string, headers http.Header) (*http.Response, error) {
	return r.DoStream(ctx, http.MethodGet, url, headers, nil)
}

// DoStream executes a request and leaves a successful response body open for
// the caller. A nil requestBody produces an empty body.
func (r *Requester) DoStream(ctx context.Context, method, url string, headers http.Header, requestBody any) (*http.Response, error) {
	var body io.Reader
	if requestBody != nil {
		jsonBody, err := json.Marshal(requestBody)
		if err != nil {
			return nil, fmt.Errorf("requester: failed to marshal request body: %w", err)
		}
		body = bytes.NewReader(jsonBody)
	}

	httpReq, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("requester: failed to create request: %w", err)
	}
	httpReq.Header = headers.Clone()

	resp, err := r.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("requester: request failed: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		rawBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("requester: API error (status %d): %s", resp.StatusCode, string(rawBody))
	}
	return resp, nil
}
