package spec

// Web search engine identifiers supported by ZHIPU's standalone and hosted
// search tools.
const (
	WebSearchEngineStandard = "search_std"
	WebSearchEnginePro      = "search_pro"
	WebSearchEngineSogou    = "search_pro_sogou"
	WebSearchEngineQuark    = "search_pro_quark"

	WebSearchRecencyOneDay   = "oneDay"
	WebSearchRecencyOneWeek  = "oneWeek"
	WebSearchRecencyOneMonth = "oneMonth"
	WebSearchRecencyOneYear  = "oneYear"
	WebSearchRecencyNoLimit  = "noLimit"

	WebSearchContentSizeMedium = "medium"
	WebSearchContentSizeHigh   = "high"
)

// WebSearchRequest is the provider-neutral request used by a standalone web
// search endpoint. ZHIPU implements it through /paas/v4/web_search.
type WebSearchRequest struct {
	SearchQuery         string         `json:"search_query"`
	SearchEngine        string         `json:"search_engine"`
	SearchIntent        bool           `json:"search_intent"`
	Count               int            `json:"count,omitempty"`
	SearchDomainFilter  string         `json:"search_domain_filter,omitempty"`
	SearchRecencyFilter string         `json:"search_recency_filter,omitempty"`
	ContentSize         string         `json:"content_size,omitempty"`
	RequestID           string         `json:"request_id,omitempty"`
	UserID              string         `json:"user_id,omitempty"`
	ExtraFields         map[string]any `json:"-"`
}

func (r WebSearchRequest) MarshalJSON() ([]byte, error) {
	type alias WebSearchRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

// WebSearchIntentResult describes ZHIPU's interpretation of a standalone
// search query.
type WebSearchIntentResult struct {
	Query    string `json:"query,omitempty"`
	Intent   string `json:"intent,omitempty"`
	Keywords string `json:"keywords,omitempty"`
}

// WebSearchDocument is one result from a standalone web search endpoint.
type WebSearchDocument struct {
	Title       string `json:"title,omitempty"`
	Content     string `json:"content,omitempty"`
	Link        string `json:"link,omitempty"`
	Media       string `json:"media,omitempty"`
	Icon        string `json:"icon,omitempty"`
	Refer       string `json:"refer,omitempty"`
	PublishDate string `json:"publish_date,omitempty"`
}

// WebSearchResponse is returned by a provider's standalone search API.
type WebSearchResponse struct {
	ID           string                  `json:"id,omitempty"`
	Created      int64                   `json:"created,omitempty"`
	RequestID    string                  `json:"request_id,omitempty"`
	SearchIntent []WebSearchIntentResult `json:"search_intent,omitempty"`
	SearchResult []WebSearchDocument     `json:"search_result,omitempty"`
	RawResponse  []byte                  `json:"-"`
}
