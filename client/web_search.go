package client

import (
	"context"
	"fmt"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// SearchWeb calls a provider's standalone web search endpoint. Use
// SendWebSearch when the selected language model should search and synthesize
// an answer itself.
func (c *Client) SearchWeb(ctx context.Context, request spec.WebSearchRequest) (*spec.WebSearchResponse, error) {
	api, ok := c.client.(spec.WebSearchClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support the standalone Web Search API", c.config.Provider)
	}
	return api.SearchWeb(ctx, request)
}
