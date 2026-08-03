package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// NewResponsesClient creates an OpenAI client with the complete Responses REST
// surface available without a type assertion.
func NewResponsesClient(opts ...spec.ClientOption) (spec.ResponsesClient, error) {
	client, err := NewClient(opts...)
	if err != nil {
		return nil, err
	}
	responsesClient, ok := client.(spec.ResponsesClient)
	if !ok {
		return nil, fmt.Errorf("openai provider: Responses API is unavailable")
	}
	return responsesClient, nil
}

// CreateResponse explicitly uses POST /responses regardless of whether the
// configured URL is an API root, /chat/completions, or /responses endpoint.
func (c *clientImpl) CreateResponse(ctx context.Context, request spec.ResponseCreateRequest, opts ...spec.Option) (*spec.Response, error) {
	if request.Model == "" {
		return nil, fmt.Errorf("openai responses: model is required")
	}
	normalizedInput, err := normalizeResponseInput(request.Input)
	if err != nil {
		return nil, err
	}
	request.Input = normalizedInput
	body, err := requestMap(request)
	if err != nil {
		return nil, err
	}
	delete(body, "model")

	endpoint, err := responsesEndpointURL(c.config.APIURL)
	if err != nil {
		return nil, err
	}
	clientCopy := *c
	clientCopy.config = c.config
	clientCopy.config.APIURL = endpoint

	requestOptions := make([]spec.Option, 0, len(opts)+2)
	requestOptions = append(requestOptions, spec.WithParameters(body))
	if request.Stream != nil && *request.Stream {
		requestOptions = append(requestOptions, spec.WithStreaming())
	}
	requestOptions = append(requestOptions, opts...)
	config := applyRequestOptions(requestOptions...)
	modelName := request.Model
	if config.Model != "" {
		modelName = config.Model
	}
	model := &modelImpl{client: &clientCopy, name: modelName}
	return model.responses(ctx, nil, config)
}

func (c *clientImpl) RetrieveResponse(ctx context.Context, responseID string, options spec.ResponseRetrieveOptions, opts ...spec.Option) (*spec.Response, error) {
	endpoint, err := c.responseResourceURL(responseID)
	if err != nil {
		return nil, err
	}
	query := url.Values{}
	addIncludes(query, options.Include)
	if options.Stream {
		query.Set("stream", "true")
		if options.StartingAfter > 0 {
			query.Set("starting_after", strconv.Itoa(options.StartingAfter))
		}
		if options.IncludeObfuscation != nil {
			query.Set("include_obfuscation", strconv.FormatBool(*options.IncludeObfuscation))
		}
	}
	endpoint = withQuery(endpoint, query)
	config := applyRequestOptions(opts...)
	if options.Stream {
		config.Streaming = true
		resp, requestErr := c.requester.GetStream(ctx, endpoint, c.headers())
		if requestErr != nil {
			return nil, requestErr
		}
		return (&modelImpl{client: c}).consumeResponsesStream(ctx, resp, map[string]any{}, config)
	}

	raw, err := c.requester.Get(ctx, endpoint, c.headers())
	if err != nil {
		return nil, err
	}
	parsed, message, err := parseResponses(raw)
	if err != nil {
		return nil, err
	}
	return newResponsesResult(parsed, message, raw), nil
}

func (c *clientImpl) DeleteResponse(ctx context.Context, responseID string) (*spec.ResponseDeleted, error) {
	endpoint, err := c.responseResourceURL(responseID)
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Delete(ctx, endpoint, c.headers())
	if err != nil {
		return nil, err
	}
	var result spec.ResponseDeleted
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("openai responses: decode delete result: %w", err)
	}
	result.Raw = append(result.Raw[:0], raw...)
	return &result, nil
}

func (c *clientImpl) CancelResponse(ctx context.Context, responseID string) (*spec.Response, error) {
	endpoint, err := c.responseResourceURL(responseID)
	if err != nil {
		return nil, err
	}
	endpoint, err = url.JoinPath(endpoint, "cancel")
	if err != nil {
		return nil, fmt.Errorf("openai responses: build cancel URL: %w", err)
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), nil)
	if err != nil {
		return nil, err
	}
	parsed, message, err := parseResponses(raw)
	if err != nil {
		return nil, err
	}
	return newResponsesResult(parsed, message, raw), nil
}

func (c *clientImpl) ListResponseInputItems(ctx context.Context, responseID string, options spec.ResponseInputItemsOptions) (*spec.ResponseInputItemsPage, error) {
	endpoint, err := c.responseResourceURL(responseID)
	if err != nil {
		return nil, err
	}
	query := url.Values{}
	setIfNotEmpty(query, "after", options.After)
	setIfNotEmpty(query, "order", options.Order)
	if options.Limit > 0 {
		query.Set("limit", strconv.Itoa(options.Limit))
	}
	addIncludes(query, options.Include)
	endpoint, err = url.JoinPath(endpoint, "input_items")
	if err != nil {
		return nil, fmt.Errorf("openai responses: build input items URL: %w", err)
	}
	raw, err := c.requester.Get(ctx, withQuery(endpoint, query), c.headers())
	if err != nil {
		return nil, err
	}
	var page spec.ResponseInputItemsPage
	if err := json.Unmarshal(raw, &page); err != nil {
		return nil, fmt.Errorf("openai responses: decode input items: %w", err)
	}
	page.Raw = append(page.Raw[:0], raw...)
	return &page, nil
}

func (c *clientImpl) CountResponseInputTokens(ctx context.Context, request spec.ResponseInputTokenCountRequest) (*spec.ResponseInputTokenCount, error) {
	normalizedInput, err := normalizeResponseInput(request.Input)
	if err != nil {
		return nil, err
	}
	request.Input = normalizedInput
	endpoint, err := c.responsesRootURL("input_tokens")
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	var result spec.ResponseInputTokenCount
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("openai responses: decode input token count: %w", err)
	}
	result.Raw = append(result.Raw[:0], raw...)
	return &result, nil
}

func (c *clientImpl) CompactResponse(ctx context.Context, request spec.ResponseCompactRequest) (*spec.ResponseCompaction, error) {
	normalizedInput, err := normalizeResponseInput(request.Input)
	if err != nil {
		return nil, err
	}
	request.Input = normalizedInput
	endpoint, err := c.responsesRootURL("compact")
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	var result spec.ResponseCompaction
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("openai responses: decode compaction: %w", err)
	}
	result.Raw = append(result.Raw[:0], raw...)
	return &result, nil
}

func (c *clientImpl) CreateConversation(ctx context.Context, request spec.ConversationCreateRequest) (*spec.Conversation, error) {
	endpoint, err := c.conversationsURL("")
	if err != nil {
		return nil, err
	}
	return c.writeConversation(ctx, http.MethodPost, endpoint, request)
}

func (c *clientImpl) RetrieveConversation(ctx context.Context, conversationID string) (*spec.Conversation, error) {
	endpoint, err := c.conversationsURL(conversationID)
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Get(ctx, endpoint, c.headers())
	if err != nil {
		return nil, err
	}
	return decodeConversation(raw)
}

func (c *clientImpl) UpdateConversation(ctx context.Context, conversationID string, request spec.ConversationUpdateRequest) (*spec.Conversation, error) {
	endpoint, err := c.conversationsURL(conversationID)
	if err != nil {
		return nil, err
	}
	return c.writeConversation(ctx, http.MethodPost, endpoint, request)
}

func (c *clientImpl) DeleteConversation(ctx context.Context, conversationID string) (*spec.ResponseDeleted, error) {
	endpoint, err := c.conversationsURL(conversationID)
	if err != nil {
		return nil, err
	}
	return c.deleteResource(ctx, endpoint)
}

func (c *clientImpl) CreateConversationItems(ctx context.Context, conversationID string, request spec.ConversationItemsRequest) (*spec.ConversationItemsPage, error) {
	endpoint, err := c.conversationItemsURL(conversationID, "")
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	return decodeConversationItems(raw)
}

func (c *clientImpl) ListConversationItems(ctx context.Context, conversationID string, options spec.ConversationItemsOptions) (*spec.ConversationItemsPage, error) {
	endpoint, err := c.conversationItemsURL(conversationID, "")
	if err != nil {
		return nil, err
	}
	query := url.Values{}
	setIfNotEmpty(query, "after", options.After)
	setIfNotEmpty(query, "order", options.Order)
	if options.Limit > 0 {
		query.Set("limit", strconv.Itoa(options.Limit))
	}
	addIncludes(query, options.Include)
	raw, err := c.requester.Get(ctx, withQuery(endpoint, query), c.headers())
	if err != nil {
		return nil, err
	}
	return decodeConversationItems(raw)
}

func (c *clientImpl) RetrieveConversationItem(ctx context.Context, conversationID, itemID string, include []string) (*spec.ResponseInputItem, error) {
	endpoint, err := c.conversationItemsURL(conversationID, itemID)
	if err != nil {
		return nil, err
	}
	query := url.Values{}
	addIncludes(query, include)
	raw, err := c.requester.Get(ctx, withQuery(endpoint, query), c.headers())
	if err != nil {
		return nil, err
	}
	var item spec.ResponseInputItem
	if err := json.Unmarshal(raw, &item); err != nil {
		return nil, fmt.Errorf("openai responses: decode conversation item: %w", err)
	}
	return &item, nil
}

func (c *clientImpl) DeleteConversationItem(ctx context.Context, conversationID, itemID string) (*spec.ResponseDeleted, error) {
	endpoint, err := c.conversationItemsURL(conversationID, itemID)
	if err != nil {
		return nil, err
	}
	return c.deleteResource(ctx, endpoint)
}

func (c *clientImpl) headers() http.Header {
	return (&modelImpl{client: c}).headers()
}

func (c *clientImpl) responseResourceURL(responseID string) (string, error) {
	if responseID == "" {
		return "", fmt.Errorf("openai responses: response ID is required")
	}
	root, err := responsesEndpointURL(c.config.APIURL)
	if err != nil {
		return "", err
	}
	return url.JoinPath(root, responseID)
}

func (c *clientImpl) responsesRootURL(child string) (string, error) {
	root, err := responsesEndpointURL(c.config.APIURL)
	if err != nil {
		return "", err
	}
	return url.JoinPath(root, child)
}

func (c *clientImpl) conversationsURL(conversationID string) (string, error) {
	root, err := apiV1RootURL(c.config.APIURL)
	if err != nil {
		return "", err
	}
	endpoint, err := url.JoinPath(root, "conversations")
	if err != nil {
		return "", fmt.Errorf("openai responses: build conversations URL: %w", err)
	}
	if conversationID != "" {
		endpoint, err = url.JoinPath(endpoint, conversationID)
		if err != nil {
			return "", fmt.Errorf("openai responses: build conversation URL: %w", err)
		}
	}
	return endpoint, nil
}

func (c *clientImpl) conversationItemsURL(conversationID, itemID string) (string, error) {
	if conversationID == "" {
		return "", fmt.Errorf("openai responses: conversation ID is required")
	}
	endpoint, err := c.conversationsURL(conversationID)
	if err != nil {
		return "", err
	}
	endpoint, err = url.JoinPath(endpoint, "items")
	if err != nil {
		return "", fmt.Errorf("openai responses: build conversation items URL: %w", err)
	}
	if itemID != "" {
		endpoint, err = url.JoinPath(endpoint, itemID)
		if err != nil {
			return "", fmt.Errorf("openai responses: build conversation item URL: %w", err)
		}
	}
	return endpoint, nil
}

func (c *clientImpl) writeConversation(ctx context.Context, method, endpoint string, request any) (*spec.Conversation, error) {
	raw, err := c.requester.Do(ctx, method, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	return decodeConversation(raw)
}

func (c *clientImpl) deleteResource(ctx context.Context, endpoint string) (*spec.ResponseDeleted, error) {
	raw, err := c.requester.Delete(ctx, endpoint, c.headers())
	if err != nil {
		return nil, err
	}
	var result spec.ResponseDeleted
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("openai responses: decode delete result: %w", err)
	}
	result.Raw = append(result.Raw[:0], raw...)
	return &result, nil
}

func decodeConversation(raw []byte) (*spec.Conversation, error) {
	var result spec.Conversation
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("openai responses: decode conversation: %w", err)
	}
	result.Raw = append(result.Raw[:0], raw...)
	return &result, nil
}

func decodeConversationItems(raw []byte) (*spec.ConversationItemsPage, error) {
	var result spec.ConversationItemsPage
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("openai responses: decode conversation items: %w", err)
	}
	result.Raw = append(result.Raw[:0], raw...)
	return &result, nil
}

func applyRequestOptions(opts ...spec.Option) *spec.RequestConfig {
	config := spec.NewRequestConfig()
	for _, option := range opts {
		option(config)
	}
	return config
}

func requestMap(request any) (map[string]any, error) {
	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("openai responses: encode request: %w", err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		return nil, fmt.Errorf("openai responses: build request: %w", err)
	}
	return body, nil
}

func responsesEndpointURL(configured string) (string, error) {
	parsed, err := url.Parse(configured)
	if err != nil {
		return "", fmt.Errorf("openai responses: invalid API URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("openai responses: API URL must be absolute")
	}
	path := strings.TrimRight(parsed.Path, "/")
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		path = strings.TrimSuffix(path, "/chat/completions") + "/responses"
	case strings.HasSuffix(path, "/responses"):
	case strings.HasSuffix(path, "/v1"):
		path += "/responses"
	default:
		path += "/responses"
	}
	parsed.Path = path
	parsed.RawPath = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func apiV1RootURL(configured string) (string, error) {
	responsesURL, err := responsesEndpointURL(configured)
	if err != nil {
		return "", err
	}
	parsed, _ := url.Parse(responsesURL)
	parsed.Path = strings.TrimSuffix(strings.TrimRight(parsed.Path, "/"), "/responses")
	return parsed.String(), nil
}

func withQuery(endpoint string, values url.Values) string {
	if len(values) == 0 {
		return endpoint
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return endpoint
	}
	query := parsed.Query()
	for key, items := range values {
		query.Del(key)
		for _, item := range items {
			query.Add(key, item)
		}
	}
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func addIncludes(values url.Values, includes []string) {
	for _, include := range includes {
		if include != "" {
			values.Add("include", include)
		}
	}
}

func setIfNotEmpty(values url.Values, key, value string) {
	if value != "" {
		values.Set(key, value)
	}
}
