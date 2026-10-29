package client

import (
	"context"
	"fmt"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// CreateEmbedding adds dimensions/encoding options to SendEmbedding.
func (c *Client) CreateEmbedding(ctx context.Context, request spec.EmbeddingRequest) (*spec.EmbeddingResponse, error) {
	api, ok := c.client.(spec.EmbeddingsClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support typed embeddings", c.config.Provider)
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateEmbedding(ctx, request)
}

func (c *Client) ListModels(ctx context.Context) (*spec.ModelList, error) {
	api, ok := c.client.(spec.ModelsClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support model listing", c.config.Provider)
	}
	return api.ListModels(ctx)
}

func (c *Client) RetrieveModel(ctx context.Context, modelID string) (*spec.ModelInfo, error) {
	api, ok := c.client.(spec.ModelsClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support model retrieval", c.config.Provider)
	}
	return api.RetrieveModel(ctx, modelID)
}

// CreateModeration leaves an omitted model to the API's moderation default;
// a configured chat model is not a moderation model.
func (c *Client) CreateModeration(ctx context.Context, request spec.ModerationRequest) (*spec.ModerationResponse, error) {
	api, ok := c.client.(spec.ModerationsClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support moderations", c.config.Provider)
	}
	return api.CreateModeration(ctx, request)
}

func (c *Client) audioAPI() (spec.AudioClient, error) {
	api, ok := c.client.(spec.AudioClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support the Audio API", c.config.Provider)
	}
	return api, nil
}

func (c *Client) CreateSpeech(ctx context.Context, request spec.SpeechRequest) (*spec.SpeechResponse, error) {
	api, err := c.audioAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateSpeech(ctx, request)
}

func (c *Client) CreateSpeechStream(ctx context.Context, request spec.SpeechRequest) (*spec.SpeechStream, error) {
	api, err := c.audioAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateSpeechStream(ctx, request)
}

func (c *Client) CreateTranscription(ctx context.Context, request spec.TranscriptionRequest) (*spec.TranscriptionResponse, error) {
	api, err := c.audioAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateTranscription(ctx, request)
}

func (c *Client) CreateTranscriptionStream(ctx context.Context, request spec.TranscriptionRequest, callback spec.TranscriptionEventCallback) (*spec.TranscriptionResponse, error) {
	api, err := c.audioAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateTranscriptionStream(ctx, request, callback)
}

func (c *Client) CreateTranslation(ctx context.Context, request spec.AudioTranslationRequest) (*spec.TranscriptionResponse, error) {
	api, err := c.audioAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateTranslation(ctx, request)
}
