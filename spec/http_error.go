package spec

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// UnmarshalJSON accepts numeric error codes from hosted MCP/JSON-RPC tools
// while retaining the string API used for OpenAI error codes.
func (e *APIError) UnmarshalJSON(data []byte) error {
	var wire struct {
		Code    json.RawMessage `json:"code"`
		Message string          `json:"message"`
		Param   string          `json:"param"`
		Type    string          `json:"type"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*e = APIError{Message: wire.Message, Param: wire.Param, Type: wire.Type}
	if len(wire.Code) > 0 && string(wire.Code) != "null" {
		if wire.Code[0] == '"' {
			return json.Unmarshal(wire.Code, &e.Code)
		}
		e.Code = string(wire.Code)
	}
	return nil
}

// HTTPError preserves the status, request ID, headers (including Retry-After),
// and error body. Use errors.As(err, &apiError) to inspect failed requests.
type HTTPError struct {
	StatusCode int
	RequestID  string
	Header     http.Header
	Body       []byte
	APIError   *APIError
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("requester: API error (status %d): %s", e.StatusCode, e.Body)
}

func (e *HTTPError) Unwrap() error {
	if e.APIError == nil {
		return nil
	}
	return e.APIError
}

func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("openai API: %s: %s", e.Code, e.Message)
	}
	return "openai API: " + e.Message
}
