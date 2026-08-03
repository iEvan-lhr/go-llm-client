package client

import (
	"context"
	"fmt"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// CreateResponse exposes the complete typed Responses request while applying
// the callbacks and defaults configured on Client.
func (c *Client) CreateResponse(ctx context.Context, request spec.ResponseCreateRequest, opts ...spec.Option) (*spec.Response, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	if request.Instructions == nil {
		if c.config.Instructions != nil {
			request.Instructions = c.config.Instructions
		} else if c.config.SystemPrompt != "" {
			request.Instructions = c.config.SystemPrompt
		}
	}
	if request.PreviousResponseID == "" {
		request.PreviousResponseID = c.config.PreviousResponseID
	}
	return api.CreateResponse(ctx, request, c.responseOptions(opts...)...)
}

// ConnectResponseWebSocket opens a persistent Responses API connection.
func (c *Client) ConnectResponseWebSocket(ctx context.Context) (spec.ResponsesWebSocket, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.ConnectResponseWebSocket(ctx)
}

func (c *Client) RetrieveResponse(ctx context.Context, responseID string, options spec.ResponseRetrieveOptions, opts ...spec.Option) (*spec.Response, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.RetrieveResponse(ctx, responseID, options, c.responseCallbackOptions(opts...)...)
}

func (c *Client) DeleteResponse(ctx context.Context, responseID string) (*spec.ResponseDeleted, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.DeleteResponse(ctx, responseID)
}

func (c *Client) CancelResponse(ctx context.Context, responseID string) (*spec.Response, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.CancelResponse(ctx, responseID)
}

func (c *Client) ListResponseInputItems(ctx context.Context, responseID string, options spec.ResponseInputItemsOptions) (*spec.ResponseInputItemsPage, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.ListResponseInputItems(ctx, responseID, options)
}

func (c *Client) CountResponseInputTokens(ctx context.Context, request spec.ResponseInputTokenCountRequest) (*spec.ResponseInputTokenCount, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CountResponseInputTokens(ctx, request)
}

func (c *Client) CompactResponse(ctx context.Context, request spec.ResponseCompactRequest) (*spec.ResponseCompaction, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.CompactResponse(ctx, request)
}

func (c *Client) CreateConversation(ctx context.Context, request spec.ConversationCreateRequest) (*spec.Conversation, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.CreateConversation(ctx, request)
}

func (c *Client) RetrieveConversation(ctx context.Context, conversationID string) (*spec.Conversation, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.RetrieveConversation(ctx, conversationID)
}

func (c *Client) UpdateConversation(ctx context.Context, conversationID string, request spec.ConversationUpdateRequest) (*spec.Conversation, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.UpdateConversation(ctx, conversationID, request)
}

func (c *Client) DeleteConversation(ctx context.Context, conversationID string) (*spec.ResponseDeleted, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.DeleteConversation(ctx, conversationID)
}

func (c *Client) CreateConversationItems(ctx context.Context, conversationID string, request spec.ConversationItemsRequest) (*spec.ConversationItemsPage, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.CreateConversationItems(ctx, conversationID, request)
}

func (c *Client) ListConversationItems(ctx context.Context, conversationID string, options spec.ConversationItemsOptions) (*spec.ConversationItemsPage, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.ListConversationItems(ctx, conversationID, options)
}

func (c *Client) RetrieveConversationItem(ctx context.Context, conversationID, itemID string, include []string) (*spec.ResponseInputItem, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.RetrieveConversationItem(ctx, conversationID, itemID, include)
}

func (c *Client) DeleteConversationItem(ctx context.Context, conversationID, itemID string) (*spec.ResponseDeleted, error) {
	api, err := c.responsesAPI()
	if err != nil {
		return nil, err
	}
	return api.DeleteConversationItem(ctx, conversationID, itemID)
}

// GenerateImage uses the Responses image_generation tool and returns the
// complete response. Use resp.Responses.ImageGenerationResults() to access the
// base64 image payloads.
func (c *Client) GenerateImage(ctx context.Context, prompt string, tool spec.ResponseTool, opts ...spec.Option) (*spec.Response, error) {
	if tool.Type == "" {
		tool = spec.NewImageGenerationTool()
	}
	if tool.Type != "image_generation" {
		return nil, fmt.Errorf("image generation tool type must be image_generation")
	}
	request := spec.ResponseCreateRequest{
		Model:      c.config.Model,
		Input:      prompt,
		Tools:      []any{tool},
		ToolChoice: spec.ToolChoice("image_generation"),
	}
	return c.CreateResponse(ctx, request, opts...)
}

func (c *Client) responsesAPI() (spec.ResponsesClient, error) {
	api, ok := c.client.(spec.ResponsesClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support the Responses API", c.config.Provider)
	}
	return api, nil
}

func (c *Client) responseOptions(opts ...spec.Option) []spec.Option {
	result := make([]spec.Option, 0, len(opts)+8)
	if c.config.Parameters != nil {
		result = append(result, spec.WithParameters(c.config.Parameters))
	}
	if c.config.Thinking != nil {
		result = append(result, spec.WithThinking(*c.config.Thinking))
	}
	if c.config.ReasoningEffort != "" {
		result = append(result, spec.WithReasoningEffort(c.config.ReasoningEffort))
	}
	if c.config.WebSearch != nil {
		result = append(result, spec.WithWebSearch(*c.config.WebSearch))
	}
	result = append(result, c.responseCallbackOptions()...)
	return append(result, opts...)
}

func (c *Client) responseCallbackOptions(opts ...spec.Option) []spec.Option {
	result := make([]spec.Option, 0, len(opts)+3)
	if c.config.StreamCallback != nil {
		result = append(result, spec.WithStreamCallback(c.config.StreamCallback))
	}
	if c.config.ReasoningCallback != nil {
		result = append(result, spec.WithReasoningCallback(c.config.ReasoningCallback))
	}
	if c.config.EventCallback != nil {
		result = append(result, spec.WithEventCallback(c.config.EventCallback))
	}
	return append(result, opts...)
}
