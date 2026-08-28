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
	CacheWriteTokens         int `json:"cache_write_tokens,omitempty"`
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
	Data        string            `json:"data,omitempty"`
	Transcript  string            `json:"transcript,omitempty"`
	Annotations []json.RawMessage `json:"annotations,omitempty"`
	Logprobs    []json.RawMessage `json:"logprobs,omitempty"`
	Raw         json.RawMessage   `json:"-"`
}

// ResponseOutputItem represents the common fields across Responses output item
// variants, including messages, function calls, and hosted tool calls.
type ResponseOutputItem struct {
	ID                       string                  `json:"id,omitempty"`
	Type                     string                  `json:"type"`
	Role                     Role                    `json:"role,omitempty"`
	Status                   string                  `json:"status,omitempty"`
	Content                  []ResponseOutputContent `json:"content,omitempty"`
	Summary                  []ResponseOutputContent `json:"summary,omitempty"`
	Name                     string                  `json:"name,omitempty"`
	CallID                   string                  `json:"call_id,omitempty"`
	Arguments                string                  `json:"arguments,omitempty"`
	Input                    string                  `json:"input,omitempty"`
	Output                   json.RawMessage         `json:"output,omitempty"`
	Action                   json.RawMessage         `json:"action,omitempty"`
	Result                   json.RawMessage         `json:"result,omitempty"`
	Results                  json.RawMessage         `json:"results,omitempty"`
	Code                     string                  `json:"code,omitempty"`
	ContainerID              string                  `json:"container_id,omitempty"`
	Queries                  []string                `json:"queries,omitempty"`
	ServerLabel              string                  `json:"server_label,omitempty"`
	ApprovalRequestID        string                  `json:"approval_request_id,omitempty"`
	EncryptedContent         string                  `json:"encrypted_content,omitempty"`
	Operation                json.RawMessage         `json:"operation,omitempty"`
	PendingSafetyChecks      []json.RawMessage       `json:"pending_safety_checks,omitempty"`
	AcknowledgedSafetyChecks []json.RawMessage       `json:"acknowledged_safety_checks,omitempty"`
	Error                    *APIError               `json:"error,omitempty"`
	Phase                    string                  `json:"phase,omitempty"`
	Raw                      json.RawMessage         `json:"-"`
}

const (
	WebSearchContextSizeLow    = "low"
	WebSearchContextSizeMedium = "medium"
	WebSearchContextSizeHigh   = "high"

	WebSearchReturnTokenBudgetDefault   = "default"
	WebSearchReturnTokenBudgetUnlimited = "unlimited"
)

// WebSearchConfig configures model-hosted search. Providers translate the
// common fields to Responses tools, Chat Completions web_search_options, or a
// provider-specific web_search tool.
type WebSearchConfig struct {
	SearchContextSize  string
	ReturnTokenBudget  string
	ExternalWebAccess  *bool
	Filters            *WebSearchFilters
	UserLocation       *WebSearchUserLocation
	SearchContentTypes []string
	ImageSettings      *WebSearchImageSettings
	IncludeSources     bool
	IncludeResults     bool
	ToolChoice         any

	// The fields below configure ZHIPU's hosted web_search tool. ContentSize
	// falls back to SearchContextSize when it is not set.
	SearchEngine        string
	SearchQuery         string
	SearchIntent        *bool
	Count               int
	SearchDomainFilter  string
	SearchRecencyFilter string
	ContentSize         string
	ResultSequence      string
	RequireSearch       *bool
	SearchPrompt        string
}

// WebSearchFilters limits or excludes domains. Domains should not include a
// URL scheme.
type WebSearchFilters struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	BlockedDomains []string `json:"blocked_domains,omitempty"`
}

// WebSearchUserLocation supplies an approximate location for local results.
// Type defaults to "approximate" when omitted.
type WebSearchUserLocation struct {
	Type     string `json:"type,omitempty"`
	Country  string `json:"country,omitempty"`
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// WebSearchImageSettings controls image results when SearchContentTypes
// contains "image".
type WebSearchImageSettings struct {
	MaxResults int  `json:"max_results,omitempty"`
	Caption    bool `json:"caption,omitempty"`
}

// Bool returns a pointer suitable for optional boolean request fields.
func Bool(value bool) *bool {
	return &value
}

// URLCitation identifies a source cited by an output_text part.
type URLCitation struct {
	Type       string `json:"type,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
}

// WebSearchSource is a source consulted by a web search action.
type WebSearchSource struct {
	Type  string `json:"type,omitempty"`
	URL   string `json:"url,omitempty"`
	Title string `json:"title,omitempty"`
}

// WebSearchAction describes a search, page open, or in-page find action.
type WebSearchAction struct {
	Type    string            `json:"type,omitempty"`
	Query   string            `json:"query,omitempty"`
	Queries []string          `json:"queries,omitempty"`
	URL     string            `json:"url,omitempty"`
	Pattern string            `json:"pattern,omitempty"`
	Sources []WebSearchSource `json:"sources,omitempty"`
}

// WebSearchResult exposes hosted search results from Responses or compatible
// Chat Completions providers, including optional image and page metadata.
type WebSearchResult struct {
	Type             string `json:"type,omitempty"`
	URL              string `json:"url,omitempty"`
	Title            string `json:"title,omitempty"`
	ImageURL         string `json:"image_url,omitempty"`
	ThumbnailURL     string `json:"thumbnail_url,omitempty"`
	SourceWebsiteURL string `json:"source_website_url,omitempty"`
	Caption          string `json:"caption,omitempty"`
	Content          string `json:"content,omitempty"`
	Media            string `json:"media,omitempty"`
	Icon             string `json:"icon,omitempty"`
	Refer            string `json:"refer,omitempty"`
	PublishDate      string `json:"publish_date,omitempty"`
}

// WebSearchCall is the typed view of a hosted model search operation.
type WebSearchCall struct {
	ID      string            `json:"id,omitempty"`
	Status  string            `json:"status,omitempty"`
	Action  WebSearchAction   `json:"action,omitempty"`
	Results []WebSearchResult `json:"results,omitempty"`
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
	PromptCacheOptions   json.RawMessage      `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention string               `json:"prompt_cache_retention,omitempty"`
	Conversation         json.RawMessage      `json:"conversation,omitempty"`
	ContextManagement    []json.RawMessage    `json:"context_management,omitempty"`
	Moderation           json.RawMessage      `json:"moderation,omitempty"`
	TopLogprobs          int                  `json:"top_logprobs,omitempty"`
	User                 string               `json:"user,omitempty"`
}

// ResponseInputItem is a flexible Responses input item. It supports message
// items and function_call_output without preventing newer item types.
type ResponseInputItem struct {
	Type                     string            `json:"type,omitempty"`
	Role                     Role              `json:"role,omitempty"`
	Status                   string            `json:"status,omitempty"`
	Content                  any               `json:"content,omitempty"`
	ID                       string            `json:"id,omitempty"`
	CallID                   string            `json:"call_id,omitempty"`
	Name                     string            `json:"name,omitempty"`
	Arguments                string            `json:"arguments,omitempty"`
	Input                    string            `json:"input,omitempty"`
	Output                   any               `json:"output,omitempty"`
	ApprovalRequestID        string            `json:"approval_request_id,omitempty"`
	Approve                  *bool             `json:"approve,omitempty"`
	Reason                   string            `json:"reason,omitempty"`
	EncryptedContent         string            `json:"encrypted_content,omitempty"`
	Operation                any               `json:"operation,omitempty"`
	AcknowledgedSafetyChecks []json.RawMessage `json:"acknowledged_safety_checks,omitempty"`
	Raw                      json.RawMessage   `json:"-"`
	ExtraFields              map[string]any    `json:"-"`
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

// NewToolCallOutput creates an output item for a tool call type such as
// computer_call_output, custom_tool_call_output, local_shell_call_output,
// shell_call_output, or apply_patch_call_output.
func NewToolCallOutput(itemType, callID string, output any) ResponseInputItem {
	return ResponseInputItem{Type: itemType, CallID: callID, Output: output}
}

func NewComputerCallOutput(callID string, output any, acknowledgedSafetyChecks ...json.RawMessage) ResponseInputItem {
	return ResponseInputItem{
		Type:                     "computer_call_output",
		CallID:                   callID,
		Output:                   output,
		AcknowledgedSafetyChecks: acknowledgedSafetyChecks,
	}
}

func NewCustomToolCallOutput(callID string, output any) ResponseInputItem {
	return NewToolCallOutput("custom_tool_call_output", callID, output)
}

func NewShellCallOutput(callID string, output any) ResponseInputItem {
	return NewToolCallOutput("shell_call_output", callID, output)
}

func NewLocalShellCallOutput(callID string, output any) ResponseInputItem {
	return NewToolCallOutput("local_shell_call_output", callID, output)
}

func NewApplyPatchCallOutput(callID string, output any) ResponseInputItem {
	return NewToolCallOutput("apply_patch_call_output", callID, output)
}

func NewMCPApprovalResponse(approvalRequestID string, approve bool, reason string) ResponseInputItem {
	return ResponseInputItem{
		Type:              "mcp_approval_response",
		ApprovalRequestID: approvalRequestID,
		Approve:           Bool(approve),
		Reason:            reason,
	}
}

func NewItemReference(itemID string) ResponseInputItem {
	return ResponseInputItem{Type: "item_reference", ID: itemID}
}

// StreamEvent exposes every SSE payload while retaining common typed fields.
// Raw always contains the complete event JSON.
type StreamEvent struct {
	Protocol          Protocol        `json:"-"`
	Type              string          `json:"type,omitempty"`
	SequenceNumber    int             `json:"sequence_number,omitempty"`
	Delta             string          `json:"delta,omitempty"`
	ResponseID        string          `json:"response_id,omitempty"`
	ItemID            string          `json:"item_id,omitempty"`
	OutputIndex       int             `json:"output_index,omitempty"`
	ContentIndex      int             `json:"content_index,omitempty"`
	SummaryIndex      int             `json:"summary_index,omitempty"`
	AnnotationIndex   int             `json:"annotation_index,omitempty"`
	PartialImageIndex int             `json:"partial_image_index,omitempty"`
	Code              string          `json:"code,omitempty"`
	Message           string          `json:"message,omitempty"`
	Obfuscation       string          `json:"obfuscation,omitempty"`
	Agent             json.RawMessage `json:"agent,omitempty"`
	Response          json.RawMessage `json:"response,omitempty"`
	Item              json.RawMessage `json:"item,omitempty"`
	Part              json.RawMessage `json:"part,omitempty"`
	Annotation        json.RawMessage `json:"annotation,omitempty"`
	Raw               json.RawMessage `json:"-"`
}

// EventCallback receives each protocol event or Chat Completions chunk.
type EventCallback func(ctx context.Context, event StreamEvent) error
