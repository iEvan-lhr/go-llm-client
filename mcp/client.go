package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	URL, Token string
	HTTP       *http.Client
	Timeout    time.Duration
	MaxRetries int
	Logger     *log.Logger
}
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	return fmt.Sprintf("mcp json-rpc error (%d): %s", e.Code, e.Message)
}
func (c *Client) Call(ctx context.Context, method string, params any, result any) error {
	if c.URL == "" {
		return fmt.Errorf("mcp: URL is required")
	}
	if c.HTTP == nil {
		c.HTTP = &http.Client{}
	}
	attempts := c.MaxRetries + 1
	if attempts < 1 {
		attempts = 1
	}
	for i := 0; i < attempts; i++ {
		err := c.callOnce(ctx, method, params, result)
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if i+1 < attempts {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Duration(i+1) * 100 * time.Millisecond):
			}
		} else {
			return err
		}
	}
	return nil
}
func (c *Client) callOnce(ctx context.Context, method string, params any, result any) error {
	body, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	if c.Logger != nil {
		c.Logger.Printf("mcp request %s", method)
	}
	timeout := c.Timeout
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
		req = req.WithContext(ctx)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("mcp: HTTP status %d", resp.StatusCode)
	}
	var envelope struct {
		Result json.RawMessage `json:"result"`
		Error  *RPCError       `json:"error"`
	}
	if err = json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("mcp: invalid response: %w", err)
	}
	if envelope.Error != nil {
		return envelope.Error
	}
	if result != nil && len(envelope.Result) > 0 {
		if err = json.Unmarshal(envelope.Result, result); err != nil {
			return fmt.Errorf("mcp: invalid result: %w", err)
		}
	}
	return nil
}
func NormalizeURL(s string) string { return strings.TrimRight(strings.TrimSpace(s), "/") }
