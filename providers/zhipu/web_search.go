package zhipu

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// SearchWeb implements ZHIPU's standalone /paas/v4/web_search endpoint.
func (c *clientImpl) SearchWeb(ctx context.Context, request spec.WebSearchRequest) (*spec.WebSearchResponse, error) {
	if err := validateWebSearchRequest(&request); err != nil {
		return nil, err
	}
	endpoint, err := webSearchEndpoint(c.config.APIURL)
	if err != nil {
		return nil, err
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer "+c.config.APIKey)
	rawBody, err := c.requester.Post(ctx, endpoint, headers, request)
	if err != nil {
		return nil, err
	}

	var envelope struct {
		Error *spec.APIError `json:"error"`
	}
	if err := json.Unmarshal(rawBody, &envelope); err == nil && envelope.Error != nil {
		return nil, apiError(envelope.Error)
	}

	var response spec.WebSearchResponse
	if err := json.Unmarshal(rawBody, &response); err != nil {
		return nil, fmt.Errorf("zhipu web search: failed to unmarshal response: %w", err)
	}
	response.RawResponse = append([]byte(nil), rawBody...)
	return &response, nil
}

func validateWebSearchRequest(request *spec.WebSearchRequest) error {
	request.SearchQuery = strings.TrimSpace(request.SearchQuery)
	if request.SearchQuery == "" {
		return fmt.Errorf("zhipu web search: search query is required")
	}
	if utf8.RuneCountInString(request.SearchQuery) > 70 {
		return fmt.Errorf("zhipu web search: search query must not exceed 70 characters")
	}
	if request.SearchEngine == "" {
		request.SearchEngine = spec.WebSearchEngineStandard
	}
	switch request.SearchEngine {
	case spec.WebSearchEngineStandard, spec.WebSearchEnginePro, spec.WebSearchEngineSogou, spec.WebSearchEngineQuark:
	default:
		return fmt.Errorf("zhipu web search: unsupported search engine %q", request.SearchEngine)
	}
	if request.Count < 0 || request.Count > 50 {
		return fmt.Errorf("zhipu web search: count must be between 1 and 50")
	}
	if request.SearchRecencyFilter != "" {
		switch request.SearchRecencyFilter {
		case spec.WebSearchRecencyOneDay, spec.WebSearchRecencyOneWeek, spec.WebSearchRecencyOneMonth, spec.WebSearchRecencyOneYear, spec.WebSearchRecencyNoLimit:
		default:
			return fmt.Errorf("zhipu web search: unsupported recency filter %q", request.SearchRecencyFilter)
		}
	}
	if request.ContentSize != "" && request.ContentSize != spec.WebSearchContentSizeMedium && request.ContentSize != spec.WebSearchContentSizeHigh {
		return fmt.Errorf("zhipu web search: unsupported content size %q", request.ContentSize)
	}
	return nil
}

func webSearchEndpoint(apiURL string) (string, error) {
	parsed, err := url.Parse(apiURL)
	if err != nil {
		return "", fmt.Errorf("zhipu web search: invalid API URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("zhipu web search: API URL must be absolute")
	}

	path := strings.TrimRight(parsed.Path, "/")
	switch {
	case strings.HasSuffix(path, "/paas/v4/web_search"):
		// The configured URL already points at the standalone search API.
	case strings.HasSuffix(path, "/paas/v4/chat/completions"):
		path = strings.TrimSuffix(path, "/paas/v4/chat/completions") + "/paas/v4/web_search"
	case strings.HasSuffix(path, "/v1/chat/completions"):
		path = strings.TrimSuffix(path, "/v1/chat/completions") + "/paas/v4/web_search"
	case path == "" || path == "/":
		path = "/paas/v4/web_search"
	case strings.HasSuffix(path, "/api"):
		path += "/paas/v4/web_search"
	default:
		path += "/paas/v4/web_search"
	}
	parsed.Path = path
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func applyHostedWebSearch(requestBody map[string]any, config spec.WebSearchConfig) error {
	options := map[string]any{
		"enable":        true,
		"search_engine": config.SearchEngine,
	}
	if options["search_engine"] == "" {
		options["search_engine"] = spec.WebSearchEngineStandard
	}
	if config.SearchQuery != "" {
		options["search_query"] = config.SearchQuery
	}
	if config.SearchIntent != nil {
		options["search_intent"] = *config.SearchIntent
	}
	if config.Count > 0 {
		options["count"] = config.Count
	}
	domainFilter := config.SearchDomainFilter
	if domainFilter == "" && config.Filters != nil && len(config.Filters.AllowedDomains) == 1 {
		domainFilter = config.Filters.AllowedDomains[0]
	}
	if domainFilter != "" {
		options["search_domain_filter"] = domainFilter
	}
	if config.SearchRecencyFilter != "" {
		options["search_recency_filter"] = config.SearchRecencyFilter
	}
	contentSize := config.ContentSize
	if contentSize == "" && (config.SearchContextSize == spec.WebSearchContextSizeMedium || config.SearchContextSize == spec.WebSearchContextSizeHigh) {
		contentSize = config.SearchContextSize
	}
	if contentSize != "" {
		options["content_size"] = contentSize
	}
	if config.ResultSequence != "" {
		options["result_sequence"] = config.ResultSequence
	}
	if config.IncludeResults || config.IncludeSources {
		options["search_result"] = true
	}
	if config.RequireSearch != nil {
		options["require_search"] = *config.RequireSearch
	}
	if config.SearchPrompt != "" {
		options["search_prompt"] = config.SearchPrompt
	}

	tools, err := normalizeTools(requestBody["tools"])
	if err != nil {
		return err
	}
	replaced := false
	for index, tool := range tools {
		object, ok := tool.(map[string]any)
		if !ok || object["type"] != "web_search" {
			continue
		}
		merged := map[string]any{}
		if existing, ok := object["web_search"].(map[string]any); ok {
			for key, value := range existing {
				merged[key] = value
			}
		}
		for key, value := range options {
			merged[key] = value
		}
		object["web_search"] = merged
		tools[index] = object
		replaced = true
		break
	}
	if !replaced {
		tools = append(tools, map[string]any{"type": "web_search", "web_search": options})
	}
	requestBody["tools"] = tools
	return nil
}

func normalizeTools(value any) ([]any, error) {
	if value == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("zhipu provider: encode tools: %w", err)
	}
	var tools []any
	if err := json.Unmarshal(encoded, &tools); err != nil {
		return nil, fmt.Errorf("zhipu provider: tools must be an array: %w", err)
	}
	return tools, nil
}

func zhipuSearchMetadata(rawBody []byte) ([]spec.WebSearchCall, []spec.URLCitation) {
	var envelope struct {
		WebSearch []spec.WebSearchDocument `json:"web_search"`
	}
	if err := json.Unmarshal(rawBody, &envelope); err != nil || len(envelope.WebSearch) == 0 {
		return nil, nil
	}

	sources := make([]spec.WebSearchSource, 0, len(envelope.WebSearch))
	results := make([]spec.WebSearchResult, 0, len(envelope.WebSearch))
	citations := make([]spec.URLCitation, 0, len(envelope.WebSearch))
	for _, item := range envelope.WebSearch {
		sources = append(sources, spec.WebSearchSource{Type: "url", URL: item.Link, Title: item.Title})
		results = append(results, spec.WebSearchResult{
			Type:        "url_result",
			URL:         item.Link,
			Title:       item.Title,
			Content:     item.Content,
			Media:       item.Media,
			Icon:        item.Icon,
			Refer:       item.Refer,
			PublishDate: item.PublishDate,
		})
		citations = append(citations, spec.URLCitation{Type: "url_citation", URL: item.Link, Title: item.Title})
	}
	return []spec.WebSearchCall{{Status: "completed", Action: spec.WebSearchAction{Type: "search", Sources: sources}, Results: results}}, citations
}

var _ spec.WebSearchClient = (*clientImpl)(nil)
