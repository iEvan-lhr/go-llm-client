package opencodego

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

const chatReply = `{"id":"chat_1","model":"glm-5.3","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_content":"plan"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10,"prompt_cache_hit_tokens":3}}`
const responsesReply = `{"id":"resp_1","model":"gpt-6-luna","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":8,"output_tokens":2,"total_tokens":10}}`
const messagesReply = `{"id":"msg_1","type":"message","role":"assistant","model":"minimax-m3","content":[{"type":"thinking","thinking":"plan","signature":"signed"},{"type":"text","text":"ok"},{"type":"tool_use","id":"tool_1","name":"lookup","input":{"city":"Shanghai","ticket":9007199254740993}}],"stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":0}}`

func TestOpenCodeGoRoutesAndIdentifiesRequests(t *testing.T) {
	for _, test := range []struct {
		model, path string
		protocol    spec.Protocol
	}{
		{"glm-5.3", "/chat/completions", spec.ProtocolChatCompletions},
		{"opencode-go/kimi-k3", "/chat/completions", spec.ProtocolChatCompletions},
		{"deepseek-v4.1-flash", "/chat/completions", spec.ProtocolChatCompletions},
		{"mimo-v2.6-flash", "/chat/completions", spec.ProtocolChatCompletions},
		{"longcat-2.5-preview-free", "/chat/completions", spec.ProtocolChatCompletions},
		{"future-model", "/chat/completions", spec.ProtocolChatCompletions},
		{"gpt-6-luna", "/responses", spec.ProtocolResponses},
		{"grok-4.7", "/responses", spec.ProtocolResponses},
		{"muse-spark-1.3-contributor", "/responses", spec.ProtocolResponses},
		{"minimax-m3", "/messages", spec.ProtocolAnthropicMessages},
		{"minimax-m2.5", "/messages", spec.ProtocolAnthropicMessages},
		{"claude-haiku-5-5", "/messages", spec.ProtocolAnthropicMessages},
		{"qwen3.8-max", "/messages", spec.ProtocolAnthropicMessages},
	} {
		t.Run(test.model, func(t *testing.T) {
			transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.Method != http.MethodPost || r.URL.String() != DefaultAPIURL+test.path {
					t.Errorf("request = %s %s", r.Method, r.URL)
				}
				if r.Header.Get("Authorization") != "Bearer test-key" || r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("headers = %#v", r.Header)
				}
				if r.Header.Get("User-Agent") != "my-agent/2.0" || r.Header.Get("x-opencode-session") != "conversation-1" {
					t.Errorf("identity headers = %#v", r.Header)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if body["model"] != strings.TrimPrefix(test.model, "opencode-go/") {
					t.Errorf("model = %v", body["model"])
				}
				reply := chatReply
				if test.protocol == spec.ProtocolResponses {
					reply = responsesReply
					if body["input"] == nil || body["messages"] != nil {
						t.Errorf("Responses body = %#v", body)
					}
				} else if test.protocol == spec.ProtocolAnthropicMessages {
					reply = messagesReply
					if body["max_tokens"] != float64(4096) || r.Header.Get("anthropic-version") != "2023-06-01" || r.Header.Get("x-api-key") != "test-key" {
						t.Errorf("Messages request = %#v, headers = %#v", body, r.Header)
					}
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(reply))}, nil
			})
			httpClient := &http.Client{Transport: transport, Timeout: 17 * time.Second}
			provider, err := NewClient(spec.WithAPIKey("test-key"), spec.WithSessionID("conversation-1"), spec.WithUserAgent("my-agent/2.0"), spec.WithHTTPClient(httpClient))
			if err != nil {
				t.Fatal(err)
			}
			response, err := provider.Model(test.model).Chat(context.Background(), []spec.Message{spec.NewUserMessage("review code")})
			if err != nil {
				t.Fatal(err)
			}
			if response.Protocol != test.protocol || response.Message.Content != "ok" {
				t.Fatalf("response = %#v", response)
			}
			if _, ok := httpClient.Transport.(*sessionTransport); ok {
				t.Fatal("caller HTTP transport was modified")
			}
			if provider.(*clientImpl).config.HTTPClient.Timeout != 17*time.Second {
				t.Fatal("caller timeout was lost")
			}
		})
	}
}

func TestOpenCodeGoExplicitEndpointAndModelOverride(t *testing.T) {
	for _, endpoint := range []string{"http://localhost/v1/chat/completions?proxy=1", "http://localhost/v1/responses/", "http://localhost/v1/messages"} {
		provider, err := NewClient(spec.WithAPIKey("test-key"), spec.WithAPIURL(endpoint), spec.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body["model"] != "gpt-6-luna" {
				t.Errorf("model override = %v", body["model"])
			}
			if r.URL.String() != strings.TrimSuffix(endpoint, "/") {
				t.Errorf("endpoint override = %s", r.URL)
			}
			reply := chatReply
			if strings.Contains(endpoint, "responses") {
				reply = responsesReply
			}
			if strings.Contains(endpoint, "messages") {
				reply = messagesReply
			}
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(reply))}, nil
		})}))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := provider.Model("glm-5.3").Chat(context.Background(), []spec.Message{spec.NewUserMessage("review")}, spec.WithModel("gpt-6-luna")); err != nil {
			t.Fatal(err)
		}
	}
}

func TestOpenCodeGoCompatibleChatAndStream(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if body["thinking"].(map[string]any)["type"] != "disabled" || body["reasoning_effort"] != "high" || body["max_tokens"] != float64(50) || body["temperature"] != float64(0.5) {
					t.Errorf("body = %#v", body)
				}
				if !stream {
					_, _ = fmt.Fprint(w, chatReply)
					return
				}
				if body["stream_options"].(map[string]any)["include_usage"] != true {
					t.Errorf("stream options = %#v", body)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				for _, event := range []string{
					`{"id":"chat_1","model":"glm-5.3","choices":[{"index":0,"delta":{"reasoning_content":"plan"}}]}`,
					`{"choices":[{"index":0,"delta":{"content":"ok","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
					`{"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10,"prompt_cache_hit_tokens":3}}`,
				} {
					_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
				}
				_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			}))
			defer server.Close()
			parameters := map[string]any{"thinking": map[string]any{"type": "enabled", "extra": true}}
			var text, reasoning strings.Builder
			opts := []spec.Option{spec.WithParameters(parameters), spec.WithThinking(false), spec.WithReasoningEffort(spec.ReasoningEffortHigh), spec.WithMaxTokens(50), spec.WithTemperature(0.5)}
			if stream {
				opts = append(opts, spec.WithStreamCallback(func(_ context.Context, s string) error { text.WriteString(s); return nil }), spec.WithReasoningCallback(func(_ context.Context, s string) error { reasoning.WriteString(s); return nil }))
			}
			provider, err := NewClient(spec.WithAPIKey("test-key"), spec.WithAPIURL(server.URL+"/v1"))
			if err != nil {
				t.Fatal(err)
			}
			response, err := provider.Model("glm-5.3").Chat(context.Background(), []spec.Message{spec.NewUserMessage("review")}, opts...)
			if err != nil {
				t.Fatal(err)
			}
			if response.Message.Content != "ok" || response.Message.ReasoningContent != "plan" || response.CachedTokens() != 3 || response.InputTokens() != 8 {
				t.Errorf("response = %#v", response)
			}
			if parameters["thinking"].(map[string]any)["type"] != "enabled" {
				t.Fatal("caller parameters were mutated")
			}
			if stream && (text.String() != "ok" || reasoning.String() != "plan" || len(response.Message.ToolCalls) != 1) {
				t.Errorf("stream result = %#v", response)
			}
		})
	}
}

func TestOpenCodeGoMessagesConvertsInputAndReplaysSignedThinking(t *testing.T) {
	requestNumber := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		decoder := json.NewDecoder(r.Body)
		decoder.UseNumber()
		if err := decoder.Decode(&body); err != nil {
			t.Error(err)
			return
		}
		requestNumber++
		messages := body["messages"].([]any)
		if requestNumber == 1 {
			if len(body["system"].([]any)) != 2 || len(messages) != 3 {
				t.Errorf("messages body = %#v", body)
			}
			content := messages[0].(map[string]any)["content"].([]any)
			if content[1].(map[string]any)["source"].(map[string]any)["type"] != "base64" || content[2].(map[string]any)["type"] != "document" {
				t.Errorf("multimodal content = %#v", content)
			}
			tool := body["tools"].([]any)[0].(map[string]any)
			if tool["name"] != "lookup" || tool["input_schema"] == nil || tool["function"] != nil {
				t.Errorf("tool = %#v", tool)
			}
			properties := tool["input_schema"].(map[string]any)["properties"].(map[string]any)
			if properties["ticket"].(map[string]any)["const"] != json.Number("9007199254740993") {
				t.Errorf("tool schema lost number precision: %#v", properties)
			}
			if body["tool_choice"].(map[string]any)["type"] != "any" || body["thinking"].(map[string]any)["type"] != "adaptive" {
				t.Errorf("body = %#v", body)
			}
			results := messages[2].(map[string]any)["content"].([]any)
			if len(results) != 2 || results[1].(map[string]any)["tool_use_id"] != "previous_2" {
				t.Errorf("tool results = %#v", results)
			}
		} else {
			content := messages[1].(map[string]any)["content"].([]any)
			if content[0].(map[string]any)["signature"] != "signed" || content[2].(map[string]any)["id"] != "tool_1" {
				t.Errorf("native replay = %#v", content)
			}
			if content[2].(map[string]any)["input"].(map[string]any)["ticket"] != json.Number("9007199254740993") {
				t.Errorf("native arguments lost number precision: %#v", content)
			}
		}
		_, _ = fmt.Fprint(w, messagesReply)
	}))
	defer server.Close()
	provider, err := NewClient(spec.WithAPIKey("test-key"), spec.WithAPIURL(server.URL+"/v1"))
	if err != nil {
		t.Fatal(err)
	}
	parameters := map[string]any{"tools": []any{map[string]any{"type": "function", "function": map[string]any{"name": "lookup", "parameters": map[string]any{"type": "object", "properties": map[string]any{"ticket": map[string]any{"const": json.Number("9007199254740993")}}}}}}, "tool_choice": "required", "thinking": map[string]any{"type": "adaptive"}}
	response, err := provider.Model("minimax-m3").Chat(context.Background(), []spec.Message{
		spec.NewSystemMessage("coding agent"), {Role: spec.RoleDeveloper, Content: "be precise"},
		spec.NewUserPartsMessage(spec.NewTextPart("review"), spec.NewImageBase64Part("image/png", "aW1hZ2U="), spec.NewInputFilePart("https://example.com/report.pdf")),
		{Role: spec.RoleAssistant, ToolCalls: []spec.ToolCall{{ID: "previous_1", Type: "function", Function: spec.FunctionCall{Name: "lookup", Arguments: `{}`}}, {ID: "previous_2", Type: "function", Function: spec.FunctionCall{Name: "lookup", Arguments: `{}`}}}},
		spec.NewToolMessage("previous_1", "one"), spec.NewToolMessage("previous_2", "two"),
	}, spec.WithParameters(parameters), spec.WithThinking(true))
	if err != nil {
		t.Fatal(err)
	}
	if response.Message.ReasoningContent != "plan" || len(response.Message.ToolCalls) != 1 || response.InputTokens() != 8 || response.Usage.TotalTokens != 10 {
		t.Fatalf("response = %#v", response)
	}
	cache := response.CacheUsage()
	if cache.ReadTokens != 3 || !cache.WriteReported || cache.WriteTokens != 0 {
		t.Fatalf("cache usage = %#v", cache)
	}
	clone := response.Message.Clone()
	response.Message.AnthropicContent[0] = '!'
	if !json.Valid(clone.AnthropicContent) {
		t.Fatal("native content was not copied")
	}
	if _, err := provider.Model("minimax-m3").Chat(context.Background(), []spec.Message{spec.NewUserMessage("review"), clone, spec.NewToolMessage("tool_1", "found")}); err != nil {
		t.Fatal(err)
	}
	if parameters["tools"].([]any)[0].(map[string]any)["type"] != "function" {
		t.Fatal("tools were mutated")
	}
}

var messageStreamEvents = []string{
	`{"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","model":"minimax-m3","content":[],"usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":3,"cache_creation_input_tokens":0}}}`,
	`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}`,
	`{"type":"content_block_stop","index":0}`,
	`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
	`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}`,
	`{"type":"content_block_stop","index":1}`,
	`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}`,
	`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
	`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Shanghai\"}"}}`,
	`{"type":"content_block_stop","index":2}`,
	`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`,
	`{"type":"message_stop"}`,
}

func TestOpenCodeGoMessagesStreamPreservesEventsToolsAndUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["stream"] != true {
			t.Error("stream was not enabled")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range messageStreamEvents {
			_, _ = fmt.Fprintf(w, "event: ignored\ndata: %s\n\n", event)
		}
	}))
	defer server.Close()
	provider, err := NewClient(spec.WithAPIKey("test-key"), spec.WithAPIURL(server.URL+"/v1"))
	if err != nil {
		t.Fatal(err)
	}
	var text, reasoning strings.Builder
	var types []string
	response, err := provider.Model("minimax-m3").Chat(context.Background(), []spec.Message{spec.NewUserMessage("review")},
		spec.WithStreamCallback(func(_ context.Context, s string) error { text.WriteString(s); return nil }),
		spec.WithReasoningCallback(func(_ context.Context, s string) error { reasoning.WriteString(s); return nil }),
		spec.WithEventCallback(func(_ context.Context, event spec.StreamEvent) error {
			if event.Protocol != spec.ProtocolAnthropicMessages || !json.Valid(event.Raw) {
				t.Errorf("event = %#v", event)
			}
			types = append(types, event.Type)
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if text.String() != "ok" || reasoning.String() != "plan" || response.Status != "tool_use" || len(types) != len(messageStreamEvents) {
		t.Fatalf("stream response = %#v, events = %v", response, types)
	}
	if response.Usage.TotalTokens != 10 || response.InputTokens() != 8 || response.CachedTokens() != 3 || !response.CacheUsage().WriteReported {
		t.Errorf("stream usage = %#v", response.Usage)
	}
	if calls := response.Message.ToolCalls; len(calls) != 1 || calls[0].Function.Arguments != `{"city":"Shanghai"}` {
		t.Errorf("stream tools = %#v", calls)
	}
	var content []contentBlock
	if err := json.Unmarshal(response.Message.AnthropicContent, &content); err != nil {
		t.Fatal(err)
	}
	if content[0].Signature != "signed" || content[0].Thinking != "plan" {
		t.Errorf("native content = %#v", content)
	}
}

func TestOpenCodeGoStreamErrors(t *testing.T) {
	callbackError := errors.New("stop callback")
	for _, test := range []struct {
		name, body, want string
		callback         spec.StreamCallback
	}{
		{"provider error", "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n", "busy", nil},
		{"malformed event", "data: {invalid}\n\n", "decode stream event", nil},
		{"missing start", "data: {\"type\":\"message_stop\"}\n\n", "before message_start", nil},
		{"truncated stream", "data: " + messageStreamEvents[0] + "\n\n", "before message_stop", nil},
		{"callback", "data: " + messageStreamEvents[0] + "\n\ndata: " + messageStreamEvents[5] + "\n\ndata: " + messageStreamEvents[6] + "\n\n", "stop callback", func(context.Context, string) error { return callbackError }},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(w, test.body)
			}))
			defer server.Close()
			provider, err := NewClient(spec.WithAPIKey("test-key"), spec.WithAPIURL(server.URL+"/v1/messages"))
			if err != nil {
				t.Fatal(err)
			}
			_, err = provider.Model("minimax-m3").Chat(context.Background(), []spec.Message{spec.NewUserMessage("review")}, spec.WithStreaming(), spec.WithStreamCallback(test.callback))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %s", err, test.want)
			}
			if test.callback != nil && !errors.Is(err, callbackError) {
				t.Fatal("callback error was not propagated")
			}
		})
	}
}

func TestOpenCodeGoSSEMultilineAndEOF(t *testing.T) {
	response := &http.Response{Body: io.NopCloser(strings.NewReader(": heartbeat\ndata: {\"type\":\n" + "data: \"ping\"}\n\n" + "data: {\"type\":\"message_stop\"}"))}
	var types []string
	err := scanMessagesSSE(response, func(raw []byte) (bool, error) {
		var event struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			return false, err
		}
		types = append(types, event.Type)
		return false, nil
	})
	if err != nil || !reflect.DeepEqual(types, []string{"ping", "message_stop"}) {
		t.Fatalf("events = %v, error = %v", types, err)
	}
}

func TestOpenCodeGoValidationAndHTTPError(t *testing.T) {
	if _, err := NewClient(); err == nil {
		t.Fatal("missing API key accepted")
	}
	for _, opts := range [][]spec.ClientOption{
		{spec.WithAPIURL(":bad")}, {spec.WithAPIURL("file:///tmp/data")}, {spec.WithHTTPClient(nil)}, {spec.WithSessionID("bad\nheader")},
	} {
		if _, err := NewClient(append([]spec.ClientOption{spec.WithAPIKey("test-key")}, opts...)...); err == nil {
			t.Fatal("invalid config accepted")
		}
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"quota exceeded"}}`)
	}))
	defer server.Close()
	provider, err := NewClient(spec.WithAPIKey("test-key"), spec.WithAPIURL(server.URL+"/v1/messages"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := provider.Model("").Chat(context.Background(), nil); err == nil {
		t.Fatal("missing model accepted")
	}
	if _, err := provider.Model("minimax-m3").Chat(context.Background(), []spec.Message{spec.NewUserMessage("review")}); err == nil || !strings.Contains(err.Error(), "429") {
		t.Fatalf("HTTP error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := provider.Model("minimax-m3").Chat(ctx, []spec.Message{spec.NewUserMessage("review")}); !errors.Is(err, context.Canceled) {
		t.Fatalf("context error = %v", err)
	}
}
