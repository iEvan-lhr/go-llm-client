package llm

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// Snapshot copies mutable request configuration, including nested tool schemas.
// Callbacks remain shared functions. Do not mutate cfg while it is being copied.
// JSON parameter objects are normalized to maps/slices, preserving number
// precision; provider-specific message input retains its typed representation.
func (cfg Config) Snapshot() (Config, error) {
	result := cfg
	var err error
	if result.Parameters, err = spec.SnapshotParameters(cfg.Parameters); err != nil {
		return Config{}, fmt.Errorf("config parameters: %w", err)
	}
	if result.ProviderOpts, err = spec.SnapshotParameters(cfg.ProviderOpts); err != nil {
		return Config{}, fmt.Errorf("config provider options: %w", err)
	}
	if result.ResponseInput, err = spec.SnapshotInput(cfg.ResponseInput); err != nil {
		return Config{}, fmt.Errorf("config response input: %w", err)
	}
	if result.Instructions, err = spec.SnapshotInput(cfg.Instructions); err != nil {
		return Config{}, fmt.Errorf("config instructions: %w", err)
	}
	if cfg.Thinking != nil {
		value := *cfg.Thinking
		result.Thinking = &value
	}
	if cfg.Translation != nil {
		value := *cfg.Translation
		result.Translation = &value
	}
	if cfg.WebExtractor != nil {
		value := *cfg.WebExtractor
		result.WebExtractor = &value
	}
	if cfg.PromptCache != nil {
		value := *cfg.PromptCache
		if cfg.PromptCache.Options != nil {
			value.Options, err = spec.SnapshotJSON(cfg.PromptCache.Options)
			if err != nil {
				return Config{}, fmt.Errorf("config prompt cache options: %w", err)
			}
		}
		result.PromptCache = &value
	}
	if cfg.WebSearch != nil {
		data, err := spec.SnapshotJSON(cfg.WebSearch)
		if err != nil {
			return Config{}, fmt.Errorf("config web search: %w", err)
		}
		var value spec.WebSearchConfig
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.UseNumber()
		if err := decoder.Decode(&value); err != nil {
			return Config{}, fmt.Errorf("config web search: %w", err)
		}
		result.WebSearch = &value
	}
	return result, nil
}
