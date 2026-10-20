package client

import (
	"context"
	"fmt"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// SystemOne evaluates structured decisions without reading or changing chat
// history. Request.Model overrides the configured model. Empty uses the
// configured model, or the provider's default free Jev model.
func (c *Client) SystemOne(ctx context.Context, request spec.SystemOneRequest) (*spec.SystemOneResponse, error) {
	api, ok := c.client.(spec.SystemOneClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support System One; use provider opencode for Jev", c.config.Provider)
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.SystemOne(ctx, request)
}
