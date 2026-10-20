package llm

import (
	"context"
	"fmt"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// SystemOne evaluates structured decisions. Request.Model takes precedence over
// Config.Model; when both are empty, the provider selects its default decision
// model (jev-1.13-free for OpenCode, decision-model-preview for DashScope). Chat settings
// (history, system prompt, tools, sampling, and streaming) do not apply.
func SystemOne(ctx context.Context, request spec.SystemOneRequest, cfg Config) (*spec.SystemOneResponse, error) {
	provider, err := GetClient(cfg)
	if err != nil {
		return nil, err
	}
	api, ok := provider.(spec.SystemOneClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support System One; use opencode for Jev or dashscope for decision models", cfg.Provider)
	}
	if request.Model == "" {
		request.Model = cfg.Model
	}
	return api.SystemOne(ctx, request)
}
