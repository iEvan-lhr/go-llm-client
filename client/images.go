package client

import (
	"context"
	"fmt"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// CreateImage uses the standalone Images API rather than the Responses API.
func (c *Client) CreateImage(ctx context.Context, request spec.ImageGenerationRequest) (*spec.ImageResponse, error) {
	api, err := c.imagesAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateImage(ctx, request)
}

// EditImage uses the standalone multipart Images edit API.
func (c *Client) EditImage(ctx context.Context, request spec.ImageEditRequest) (*spec.ImageResponse, error) {
	api, err := c.imagesAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.EditImage(ctx, request)
}

func (c *Client) imagesAPI() (spec.ImagesClient, error) {
	api, ok := c.client.(spec.ImagesClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support the Images API", c.config.Provider)
	}
	return api, nil
}

func (c *Client) CreateImageStream(ctx context.Context, request spec.ImageGenerationRequest, callback spec.ImageEventCallback) (*spec.ImageResponse, error) {
	api, ok := c.client.(spec.ImageStreamingClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support streaming images", c.config.Provider)
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateImageStream(ctx, request, callback)
}

func (c *Client) EditImageStream(ctx context.Context, request spec.ImageEditRequest, callback spec.ImageEventCallback) (*spec.ImageResponse, error) {
	api, ok := c.client.(spec.ImageStreamingClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support streaming images", c.config.Provider)
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.EditImageStream(ctx, request, callback)
}

func (c *Client) CreateImageVariation(ctx context.Context, request spec.ImageVariationRequest) (*spec.ImageResponse, error) {
	api, ok := c.client.(spec.ImageVariationsClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support image variations", c.config.Provider)
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CreateImageVariation(ctx, request)
}
