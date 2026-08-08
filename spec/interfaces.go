package spec

import "context"

// Model 是一个具体LLM模型的抽象接口。
type Model interface {
	Chat(ctx context.Context, messages []Message, opts ...Option) (*Response, error)
}

// Client 是与特定LLM提供商交互的顶层客户端。
type Client interface {
	Model(name string) Model
}

// ImagesClient is implemented by providers that expose OpenAI-compatible
// image generation and image editing endpoints.
type ImagesClient interface {
	Client
	CreateImage(ctx context.Context, request ImageGenerationRequest) (*ImageResponse, error)
	EditImage(ctx context.Context, request ImageEditRequest) (*ImageResponse, error)
}

// ResponsesClient is implemented by providers that expose OpenAI's Responses
// REST API in addition to the generic Model/Chat abstraction.
type ResponsesClient interface {
	Client
	ConnectResponseWebSocket(ctx context.Context) (ResponsesWebSocket, error)
	CreateResponse(ctx context.Context, request ResponseCreateRequest, opts ...Option) (*Response, error)
	RetrieveResponse(ctx context.Context, responseID string, options ResponseRetrieveOptions, opts ...Option) (*Response, error)
	DeleteResponse(ctx context.Context, responseID string) (*ResponseDeleted, error)
	CancelResponse(ctx context.Context, responseID string) (*Response, error)
	ListResponseInputItems(ctx context.Context, responseID string, options ResponseInputItemsOptions) (*ResponseInputItemsPage, error)
	CountResponseInputTokens(ctx context.Context, request ResponseInputTokenCountRequest) (*ResponseInputTokenCount, error)
	CompactResponse(ctx context.Context, request ResponseCompactRequest) (*ResponseCompaction, error)

	CreateConversation(ctx context.Context, request ConversationCreateRequest) (*Conversation, error)
	RetrieveConversation(ctx context.Context, conversationID string) (*Conversation, error)
	UpdateConversation(ctx context.Context, conversationID string, request ConversationUpdateRequest) (*Conversation, error)
	DeleteConversation(ctx context.Context, conversationID string) (*ResponseDeleted, error)
	CreateConversationItems(ctx context.Context, conversationID string, request ConversationItemsRequest) (*ConversationItemsPage, error)
	ListConversationItems(ctx context.Context, conversationID string, options ConversationItemsOptions) (*ConversationItemsPage, error)
	RetrieveConversationItem(ctx context.Context, conversationID, itemID string, include []string) (*ResponseInputItem, error)
	DeleteConversationItem(ctx context.Context, conversationID, itemID string) (*ResponseDeleted, error)
}
