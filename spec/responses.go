package spec

import (
	"context"
	"encoding/json"
	"fmt"
)

// ResponsesWebSocket is a persistent Responses API connection. One reader and
// one writer may be used concurrently. Receive returns every server event with
// its complete JSON in StreamEvent.Raw.
type ResponsesWebSocket interface {
	CreateResponse(ctx context.Context, request ResponseCreateRequest) error
	SendEvent(ctx context.Context, event any) error
	Receive(ctx context.Context) (StreamEvent, error)
	Close() error
}

// ResponseWebSocketCreateEvent is the client event sent by CreateResponse.
// Its request fields are flattened next to type, matching the wire protocol.
type ResponseWebSocketCreateEvent struct {
	Type    string                `json:"-"`
	Request ResponseCreateRequest `json:"-"`
}

func (e ResponseWebSocketCreateEvent) MarshalJSON() ([]byte, error) {
	body, err := json.Marshal(e.Request)
	if err != nil {
		return nil, err
	}
	var event map[string]any
	if err := json.Unmarshal(body, &event); err != nil {
		return nil, fmt.Errorf("responses websocket: build create event: %w", err)
	}
	event["type"] = e.Type
	return json.Marshal(event)
}

// ResponseCreateRequest is the complete, forward-compatible request envelope
// for POST /v1/responses. Input may be a string or a slice of input items.
// ExtraFields are merged into the JSON object so newly introduced API fields
// can be used before this package adds a dedicated field.
type ResponseCreateRequest struct {
	Model                string                   `json:"model"`
	Input                any                      `json:"input,omitempty"`
	Instructions         any                      `json:"instructions,omitempty"`
	PreviousResponseID   string                   `json:"previous_response_id,omitempty"`
	Conversation         any                      `json:"conversation,omitempty"`
	Metadata             map[string]string        `json:"metadata,omitempty"`
	Include              []string                 `json:"include,omitempty"`
	Tools                []any                    `json:"tools,omitempty"`
	ToolChoice           any                      `json:"tool_choice,omitempty"`
	ParallelToolCalls    *bool                    `json:"parallel_tool_calls,omitempty"`
	MaxToolCalls         *int                     `json:"max_tool_calls,omitempty"`
	Text                 *ResponseTextConfig      `json:"text,omitempty"`
	Reasoning            *ResponseReasoningConfig `json:"reasoning,omitempty"`
	MaxOutputTokens      *int                     `json:"max_output_tokens,omitempty"`
	Temperature          *float32                 `json:"temperature,omitempty"`
	TopP                 *float32                 `json:"top_p,omitempty"`
	TopLogprobs          *int                     `json:"top_logprobs,omitempty"`
	Stream               *bool                    `json:"stream,omitempty"`
	StreamOptions        *ResponseStreamOptions   `json:"stream_options,omitempty"`
	Background           *bool                    `json:"background,omitempty"`
	Store                *bool                    `json:"store,omitempty"`
	ServiceTier          string                   `json:"service_tier,omitempty"`
	SafetyIdentifier     string                   `json:"safety_identifier,omitempty"`
	PromptCacheKey       string                   `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   any                      `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention string                   `json:"prompt_cache_retention,omitempty"`
	Prompt               *ResponsePrompt          `json:"prompt,omitempty"`
	ContextManagement    []any                    `json:"context_management,omitempty"`
	Moderation           any                      `json:"moderation,omitempty"`
	Truncation           any                      `json:"truncation,omitempty"`
	User                 string                   `json:"user,omitempty"`
	ExtraFields          map[string]any           `json:"-"`
}

// MarshalJSON merges ExtraFields first, then lets typed fields take
// precedence. This keeps the request extensible without sacrificing type
// safety for stable fields.
func (r ResponseCreateRequest) MarshalJSON() ([]byte, error) {
	type alias ResponseCreateRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

// ResponseInputContent is a typed content part accepted inside a Responses
// message input item.
type ResponseInputContent struct {
	Type       string              `json:"type"`
	Text       string              `json:"text,omitempty"`
	ImageURL   string              `json:"image_url,omitempty"`
	FileID     string              `json:"file_id,omitempty"`
	Detail     string              `json:"detail,omitempty"`
	FileURL    string              `json:"file_url,omitempty"`
	FileData   string              `json:"file_data,omitempty"`
	Filename   string              `json:"filename,omitempty"`
	InputAudio *ResponseInputAudio `json:"input_audio,omitempty"`
}

// ResponseInputAudio is inline input audio. Format is normally "mp3" or
// "wav" and Data is base64 without a data URL prefix.
type ResponseInputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

// ResponseTextConfig controls text output, including structured outputs.
type ResponseTextConfig struct {
	Format    any    `json:"format,omitempty"`
	Verbosity string `json:"verbosity,omitempty"`
}

// ResponseJSONSchemaFormat selects strict JSON Schema structured output.
type ResponseJSONSchemaFormat struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
}

// NewResponseJSONSchemaFormat builds a strict json_schema text format.
func NewResponseJSONSchemaFormat(name string, schema any) ResponseJSONSchemaFormat {
	return ResponseJSONSchemaFormat{
		Type:   "json_schema",
		Name:   name,
		Schema: schema,
		Strict: Bool(true),
	}
}

// ResponseReasoningConfig controls reasoning-capable models.
type ResponseReasoningConfig struct {
	Effort          ReasoningEffort `json:"effort,omitempty"`
	Summary         string          `json:"summary,omitempty"`
	GenerateSummary string          `json:"generate_summary,omitempty"`
	Mode            string          `json:"mode,omitempty"`
	Context         string          `json:"context,omitempty"`
}

// ResponsePrompt references a reusable prompt template.
type ResponsePrompt struct {
	ID        string         `json:"id"`
	Version   string         `json:"version,omitempty"`
	Variables map[string]any `json:"variables,omitempty"`
}

// ResponseStreamOptions controls SSE details.
type ResponseStreamOptions struct {
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

// ResponseTool is a superset of the fields used by built-in and custom
// Responses tools. For uncommon or newly introduced tools callers may put a
// map[string]any directly in ResponseCreateRequest.Tools.
type ResponseTool struct {
	Type              string         `json:"type"`
	Name              string         `json:"name,omitempty"`
	Description       string         `json:"description,omitempty"`
	Parameters        any            `json:"parameters,omitempty"`
	Strict            *bool          `json:"strict,omitempty"`
	Format            any            `json:"format,omitempty"`
	VectorStoreIDs    []string       `json:"vector_store_ids,omitempty"`
	MaxNumResults     int            `json:"max_num_results,omitempty"`
	RankingOptions    any            `json:"ranking_options,omitempty"`
	Filters           any            `json:"filters,omitempty"`
	SearchContextSize string         `json:"search_context_size,omitempty"`
	UserLocation      any            `json:"user_location,omitempty"`
	ExternalWebAccess *bool          `json:"external_web_access,omitempty"`
	ServerLabel       string         `json:"server_label,omitempty"`
	ServerURL         string         `json:"server_url,omitempty"`
	ServerDescription string         `json:"server_description,omitempty"`
	Headers           map[string]any `json:"headers,omitempty"`
	AllowedTools      any            `json:"allowed_tools,omitempty"`
	RequireApproval   any            `json:"require_approval,omitempty"`
	Container         any            `json:"container,omitempty"`
	DisplayWidth      int            `json:"display_width,omitempty"`
	DisplayHeight     int            `json:"display_height,omitempty"`
	Environment       string         `json:"environment,omitempty"`
	Size              string         `json:"size,omitempty"`
	Quality           string         `json:"quality,omitempty"`
	OutputFormat      string         `json:"output_format,omitempty"`
	Background        string         `json:"background,omitempty"`
	OutputCompression int            `json:"output_compression,omitempty"`
	PartialImages     int            `json:"partial_images,omitempty"`
	Moderation        string         `json:"moderation,omitempty"`
	Tools             []any          `json:"tools,omitempty"`
	Execution         any            `json:"execution,omitempty"`
	AllowedCallers    []string       `json:"allowed_callers,omitempty"`
	DeferLoading      *bool          `json:"defer_loading,omitempty"`
	ExtraFields       map[string]any `json:"-"`
}

func (t ResponseTool) MarshalJSON() ([]byte, error) {
	type alias ResponseTool
	return marshalWithExtra(alias(t), t.ExtraFields)
}

func NewFunctionTool(name, description string, parameters any, strict bool) ResponseTool {
	return ResponseTool{Type: "function", Name: name, Description: description, Parameters: parameters, Strict: Bool(strict)}
}

func NewWebSearchTool() ResponseTool { return ResponseTool{Type: "web_search"} }

func NewFileSearchTool(vectorStoreIDs ...string) ResponseTool {
	return ResponseTool{Type: "file_search", VectorStoreIDs: vectorStoreIDs}
}

func NewImageGenerationTool() ResponseTool { return ResponseTool{Type: "image_generation"} }

func NewCodeInterpreterTool(container any) ResponseTool {
	return ResponseTool{Type: "code_interpreter", Container: container}
}

func NewMCPTool(label, serverURL string) ResponseTool {
	return ResponseTool{Type: "mcp", ServerLabel: label, ServerURL: serverURL}
}

func NewComputerTool(displayWidth, displayHeight int, environment string) ResponseTool {
	return ResponseTool{Type: "computer", DisplayWidth: displayWidth, DisplayHeight: displayHeight, Environment: environment}
}

func NewCustomTool(name, description string, format any) ResponseTool {
	return ResponseTool{Type: "custom", Name: name, Description: description, Format: format}
}

func NewLocalShellTool() ResponseTool { return ResponseTool{Type: "local_shell"} }

func NewShellTool() ResponseTool { return ResponseTool{Type: "shell"} }

func NewApplyPatchTool() ResponseTool { return ResponseTool{Type: "apply_patch"} }

func NewToolSearchTool() ResponseTool { return ResponseTool{Type: "tool_search"} }

func NewProgrammaticToolCallingTool() ResponseTool {
	return ResponseTool{Type: "programmatic_tool_calling"}
}

func NewNamespaceTool(name, description string, tools ...any) ResponseTool {
	return ResponseTool{Type: "namespace", Name: name, Description: description, Tools: tools}
}

// Common tool-choice helpers.
func RequiredToolChoice() any { return "required" }
func AutoToolChoice() any     { return "auto" }
func NoneToolChoice() any     { return "none" }
func ToolChoice(toolType string) any {
	return map[string]any{"type": toolType}
}
func NamedFunctionToolChoice(name string) any {
	return map[string]any{"type": "function", "name": name}
}

// ResponseRetrieveOptions controls GET /responses/{id}. Stream resumes the
// SSE event stream; StartingAfter is the last received sequence number.
type ResponseRetrieveOptions struct {
	Include            []string
	Stream             bool
	StartingAfter      int
	IncludeObfuscation *bool
}

// ResponseInputItemsOptions controls response input-item pagination.
type ResponseInputItemsOptions struct {
	After   string
	Limit   int
	Order   string
	Include []string
}

type ResponseInputItemsPage struct {
	Object  string              `json:"object,omitempty"`
	Data    []ResponseInputItem `json:"data"`
	FirstID string              `json:"first_id,omitempty"`
	LastID  string              `json:"last_id,omitempty"`
	HasMore bool                `json:"has_more"`
	Raw     json.RawMessage     `json:"-"`
}

type ResponseDeleted struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Deleted bool            `json:"deleted"`
	Raw     json.RawMessage `json:"-"`
}

type ResponseInputTokenCountRequest struct {
	Model             string                   `json:"model"`
	Input             any                      `json:"input,omitempty"`
	Instructions      any                      `json:"instructions,omitempty"`
	Tools             []any                    `json:"tools,omitempty"`
	Conversation      any                      `json:"conversation,omitempty"`
	ParallelToolCalls *bool                    `json:"parallel_tool_calls,omitempty"`
	Text              *ResponseTextConfig      `json:"text,omitempty"`
	Reasoning         *ResponseReasoningConfig `json:"reasoning,omitempty"`
	ExtraFields       map[string]any           `json:"-"`
}

func (r ResponseInputTokenCountRequest) MarshalJSON() ([]byte, error) {
	type alias ResponseInputTokenCountRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

type ResponseInputTokenCount struct {
	Object      string          `json:"object,omitempty"`
	InputTokens int             `json:"input_tokens"`
	Raw         json.RawMessage `json:"-"`
}

// ResponseCompactRequest is accepted by POST /responses/compact.
type ResponseCompactRequest struct {
	Model              string         `json:"model"`
	Input              any            `json:"input"`
	Instructions       any            `json:"instructions,omitempty"`
	PreviousResponseID string         `json:"previous_response_id,omitempty"`
	ExtraFields        map[string]any `json:"-"`
}

func (r ResponseCompactRequest) MarshalJSON() ([]byte, error) {
	type alias ResponseCompactRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

// ResponseCompaction preserves the compact endpoint's complete output.
type ResponseCompaction struct {
	ID        string               `json:"id,omitempty"`
	Object    string               `json:"object,omitempty"`
	CreatedAt int64                `json:"created_at,omitempty"`
	Output    []ResponseOutputItem `json:"output,omitempty"`
	Usage     *Usage               `json:"usage,omitempty"`
	Raw       json.RawMessage      `json:"-"`
}

// Conversation types cover the stateful Responses conversation endpoints.
type Conversation struct {
	ID        string            `json:"id"`
	Object    string            `json:"object"`
	CreatedAt int64             `json:"created_at,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Raw       json.RawMessage   `json:"-"`
}

type ConversationCreateRequest struct {
	Items    []ResponseInputItem `json:"items,omitempty"`
	Metadata map[string]string   `json:"metadata,omitempty"`
}

type ConversationUpdateRequest struct {
	Metadata map[string]string `json:"metadata"`
}

type ConversationItemsRequest struct {
	Items []ResponseInputItem `json:"items"`
}

type ConversationItemsOptions struct {
	After   string
	Limit   int
	Order   string
	Include []string
}

type ConversationItemsPage struct {
	Object  string              `json:"object,omitempty"`
	Data    []ResponseInputItem `json:"data"`
	FirstID string              `json:"first_id,omitempty"`
	LastID  string              `json:"last_id,omitempty"`
	HasMore bool                `json:"has_more"`
	Raw     json.RawMessage     `json:"-"`
}

func (c *ResponseOutputContent) UnmarshalJSON(data []byte) error {
	type alias ResponseOutputContent
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*c = ResponseOutputContent(decoded)
	c.Raw = append(c.Raw[:0], data...)
	return nil
}

func (c ResponseOutputContent) MarshalJSON() ([]byte, error) {
	type alias ResponseOutputContent
	return marshalWithRaw(alias(c), c.Raw, nil)
}

func (i *ResponseOutputItem) UnmarshalJSON(data []byte) error {
	type alias ResponseOutputItem
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*i = ResponseOutputItem(decoded)
	i.Raw = append(i.Raw[:0], data...)
	return nil
}

func (i ResponseOutputItem) MarshalJSON() ([]byte, error) {
	type alias ResponseOutputItem
	return marshalWithRaw(alias(i), i.Raw, nil)
}

func (i *ResponseInputItem) UnmarshalJSON(data []byte) error {
	type alias ResponseInputItem
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*i = ResponseInputItem(decoded)
	i.Raw = append(i.Raw[:0], data...)
	return nil
}

func (i ResponseInputItem) MarshalJSON() ([]byte, error) {
	type alias ResponseInputItem
	return marshalWithRaw(alias(i), i.Raw, i.ExtraFields)
}

// ItemsByType returns all output items with the requested wire type.
func (r *ResponsesAPIResponse) ItemsByType(itemType string) []ResponseOutputItem {
	if r == nil {
		return nil
	}
	items := make([]ResponseOutputItem, 0)
	for _, item := range r.Output {
		if item.Type == itemType {
			items = append(items, item)
		}
	}
	return items
}

// ImageGenerationResults returns base64 image payloads from all
// image_generation_call output items.
func (r *ResponsesAPIResponse) ImageGenerationResults() []string {
	var results []string
	for _, item := range r.ItemsByType("image_generation_call") {
		var value string
		if len(item.Result) > 0 && json.Unmarshal(item.Result, &value) == nil && value != "" {
			results = append(results, value)
		}
	}
	return results
}

func marshalWithExtra(value any, extra map[string]any) ([]byte, error) {
	return marshalWithRaw(value, nil, extra)
}

func marshalWithRaw(value any, raw json.RawMessage, extra map[string]any) ([]byte, error) {
	typedJSON, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if len(extra) == 0 && len(raw) == 0 {
		return typedJSON, nil
	}
	merged := make(map[string]any, len(extra)+8)
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &merged); err != nil {
			return nil, fmt.Errorf("responses: merge raw fields: %w", err)
		}
	}
	for key, item := range extra {
		merged[key] = item
	}
	var typed map[string]any
	if err := json.Unmarshal(typedJSON, &typed); err != nil {
		return nil, fmt.Errorf("responses: merge request fields: %w", err)
	}
	for key, item := range typed {
		merged[key] = item
	}
	return json.Marshal(merged)
}
