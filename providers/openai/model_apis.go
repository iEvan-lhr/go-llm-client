package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

func (c *clientImpl) CreateEmbedding(ctx context.Context, request spec.EmbeddingRequest) (*spec.EmbeddingResponse, error) {
	if strings.TrimSpace(request.Model) == "" {
		return nil, fmt.Errorf("openai embeddings: model is required")
	}
	if request.Input == nil {
		return nil, fmt.Errorf("openai embeddings: input is required")
	}
	endpoint, err := apiEndpointURL(c.config.APIURL, "embeddings")
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	var result spec.EmbeddingResponse
	if err := decodeAPIResponse(raw, &result); err != nil {
		return nil, err
	}
	result.RawResponse = raw
	return &result, nil
}

// Embed makes the existing Client.SendEmbedding entry point work for OpenAI.
func (m *modelImpl) Embed(ctx context.Context, input any) (*spec.EmbeddingResponse, error) {
	return m.client.CreateEmbedding(ctx, spec.EmbeddingRequest{Model: m.name, Input: input})
}

func (c *clientImpl) ListModels(ctx context.Context) (*spec.ModelList, error) {
	endpoint, err := apiEndpointURL(c.config.APIURL, "models")
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Get(ctx, endpoint, c.headers())
	if err != nil {
		return nil, err
	}
	var result spec.ModelList
	if err := decodeAPIResponse(raw, &result); err != nil {
		return nil, err
	}
	result.RawResponse = raw
	return &result, nil
}

func (c *clientImpl) RetrieveModel(ctx context.Context, modelID string) (*spec.ModelInfo, error) {
	if err := validateResourceID("model", modelID); err != nil {
		return nil, err
	}
	endpoint, err := apiEndpointURL(c.config.APIURL, "models")
	if err != nil {
		return nil, err
	}
	endpoint, err = url.JoinPath(endpoint, modelID)
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Get(ctx, endpoint, c.headers())
	if err != nil {
		return nil, err
	}
	var result spec.ModelInfo
	if err := decodeAPIResponse(raw, &result); err != nil {
		return nil, err
	}
	result.RawResponse = raw
	return &result, nil
}

func (c *clientImpl) CreateModeration(ctx context.Context, request spec.ModerationRequest) (*spec.ModerationResponse, error) {
	if request.Input == nil {
		return nil, fmt.Errorf("openai moderations: input is required")
	}
	endpoint, err := apiEndpointURL(c.config.APIURL, "moderations")
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	var result spec.ModerationResponse
	if err := decodeAPIResponse(raw, &result); err != nil {
		return nil, err
	}
	result.RawResponse = raw
	return &result, nil
}

func decodeAPIResponse(raw []byte, target any) error {
	var envelope struct {
		Error *spec.APIError `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return fmt.Errorf("openai: decode response: %w", err)
	}
	if envelope.Error != nil {
		return envelope.Error
	}
	if err := json.Unmarshal(raw, target); err != nil {
		return fmt.Errorf("openai: decode response: %w", err)
	}
	return nil
}

var _ spec.Embedded = (*modelImpl)(nil)
var _ spec.OpenAIClient = (*clientImpl)(nil)
