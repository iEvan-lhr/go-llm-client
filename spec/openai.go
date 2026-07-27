package spec

import (
	"context"
	"encoding/json"
)

// Protocol identifies the OpenAI wire protocol used for a response or event.
type Protocol string

const (
	ProtocolChatCompletions Protocol = "chat_completions"
	ProtocolResponses       Protocol = "responses"
)

// FunctionCall is the function name and JSON-encoded arguments of a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall represents a Chat Completions tool call. Index is populated for
// streaming deltas.
type ToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// TokenDetails contains optional token categories returned by OpenAI APIs.
type TokenDetails struct {
	CachedTokens             int `json:"cached_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}

// Usage contains the union of Chat Completions and Responses token accounting.
type Usage struct {
	PromptTokens            int           `json:"prompt_tokens,omitempty"`
	CompletionTokens        int           `json:"completion_tokens,omitempty"`
	TotalTokens             int           `json:"total_tokens,omitempty"`
	InputTokens             int           `json:"input_tokens,omitempty"`
	OutputTokens            int           `json:"output_tokens,omitempty"`
	PromptTokensDetails     *TokenDetails `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *TokenDetails `json:"completion_tokens_details,omitempty"`
	InputTokensDetails      *TokenDetails `json:"input_tokens_details,omitempty"`
	OutputTokensDetails     *TokenDetails `json:"output_tokens_details,omitempty"`
}

// APIError is the structured error shape used in Responses terminal objects.
type APIError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Param   string `json:"param,omitempty"`
	Type    string `json:"type,omitempty"`
}

// ChatChoice is one choice returned by Chat Completions.
type ChatChoice struct {
	Index        int             `json:"index"`
	Message      Message         `json:"message"`
	Delta        Message         `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     json.RawMessage `json:"logprobs,omitempty"`
}

// ChatCompletionResponse preserves the protocol-level Chat Completions result.
type ChatCompletionResponse struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"`
	Created           int64        `json:"created"`
	Model             string       `json:"model"`
	Choices           []ChatChoice `json:"choices"`
	Usage             *Usage       `json:"usage,omitempty"`
	ServiceTier       string       `json:"service_tier,omitempty"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
}

// ResponseOutputContent represents a text, refusal, or reasoning summary part.
// Raw fields that evolve independently remain available in Response.RawResponse.
type ResponseOutputContent struct {
	Type        string            `json:"type"`
	Text        string            `json:"text,omitempty"`
	Refusal     string            `json:"refusal,omitempty"`
	Annotations []json.RawMessage `json:"annotations,omitempty"`
	Logprobs    []json.RawMessage `json:"logprobs,omitempty"`
}

// ResponseOutputItem represents the common fields across Responses output item
// variants, including messages, function calls, and hosted tool calls.
type ResponseOutputItem struct {
	ID        string                  `json:"id,omitempty"`
	Type      string                  `json:"type"`
	Role      Role                    `json:"role,omitempty"`
	Status    string                  `json:"status,omitempty"`
	Content   []ResponseOutputContent `json:"content,omitempty"`
	Summary   []ResponseOutputContent `json:"summary,omitempty"`
	Name      string                  `json:"name,omitempty"`
	CallID    string                  `json:"call_id,omitempty"`
	Arguments string                  `json:"arguments,omitempty"`
	Output    json.RawMessage         `json:"output,omitempty"`
	Action    json.RawMessage         `json:"action,omitempty"`
	Result    json.RawMessage         `json:"result,omitempty"`
}

// IncompleteDetails explains why a Responses request ended incomplete.
type IncompleteDetails struct {
	Reason string `json:"reason,omitempty"`
}

// ResponsesAPIResponse preserves the protocol-level Responses result.
type ResponsesAPIResponse struct {
	ID                   string               `json:"id"`
	Object               string               `json:"object"`
	CreatedAt            int64                `json:"created_at"`
	CompletedAt          int64                `json:"completed_at,omitempty"`
	Model                string               `json:"model"`
	Status               string               `json:"status"`
	Output               []ResponseOutputItem `json:"output"`
	OutputText           string               `json:"output_text,omitempty"`
	Usage                *Usage               `json:"usage,omitempty"`
	Error                *APIError            `json:"error,omitempty"`
	IncompleteDetails    *IncompleteDetails   `json:"incomplete_details,omitempty"`
	PreviousResponseID   string               `json:"previous_response_id,omitempty"`
	Instructions         json.RawMessage      `json:"instructions,omitempty"`
	Metadata             map[string]string    `json:"metadata,omitempty"`
	ParallelToolCalls    bool                 `json:"parallel_tool_calls,omitempty"`
	MaxOutputTokens      int                  `json:"max_output_tokens,omitempty"`
	Temperature          *float64             `json:"temperature,omitempty"`
	TopP                 *float64             `json:"top_p,omitempty"`
	Background           bool                 `json:"background,omitempty"`
	Store                bool                 `json:"store,omitempty"`
	ServiceTier          string               `json:"service_tier,omitempty"`
	ToolChoice           json.RawMessage      `json:"tool_choice,omitempty"`
	Tools                []json.RawMessage    `json:"tools,omitempty"`
	Text                 json.RawMessage      `json:"text,omitempty"`
	Reasoning            json.RawMessage      `json:"reasoning,omitempty"`
	Truncation           json.RawMessage      `json:"truncation,omitempty"`
	MaxToolCalls         int                  `json:"max_tool_calls,omitempty"`
	SafetyIdentifier     string               `json:"safety_identifier,omitempty"`
	PromptCacheKey       string               `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string               `json:"prompt_cache_retention,omitempty"`
}

// ResponseInputItem is a flexible Responses input item. It supports message
// items and function_call_output without preventing newer item types.
type ResponseInputItem struct {
	Type      string `json:"type,omitempty"`
	Role      Role   `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
	ID        string `json:"id,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    any    `json:"output,omitempty"`
}

// NewFunctionCallOutput creates the input item used to return tool output to a
// Responses function call.
func NewFunctionCallOutput(callID string, output any) ResponseInputItem {
	return ResponseInputItem{
		Type:   "function_call_output",
		CallID: callID,
		Output: output,
	}
}

// StreamEvent exposes every SSE payload while retaining common typed fields.
// Raw always contains the complete event JSON.
type StreamEvent struct {
	Protocol       Protocol        `json:"-"`
	Type           string          `json:"type,omitempty"`
	SequenceNumber int             `json:"sequence_number,omitempty"`
	Delta          string          `json:"delta,omitempty"`
	ItemID         string          `json:"item_id,omitempty"`
	OutputIndex    int             `json:"output_index,omitempty"`
	ContentIndex   int             `json:"content_index,omitempty"`
	SummaryIndex   int             `json:"summary_index,omitempty"`
	Response       json.RawMessage `json:"response,omitempty"`
	Item           json.RawMessage `json:"item,omitempty"`
	Part           json.RawMessage `json:"part,omitempty"`
	Raw            json.RawMessage `json:"-"`
}

// EventCallback receives each protocol event or Chat Completions chunk.
type EventCallback func(ctx context.Context, event StreamEvent) error
