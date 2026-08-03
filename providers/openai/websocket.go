package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sync"

	"github.com/coder/websocket"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

const responsesWebSocketReadLimit = 64 * 1024 * 1024

type responseWebSocket struct {
	connection *websocket.Conn
	readMu     sync.Mutex
	writeMu    sync.Mutex
	closeOnce  sync.Once
	closeErr   error
}

// ConnectResponseWebSocket opens the persistent Responses connection
// documented by OpenAI. APIURL is normalized in the same way as REST calls.
func (c *clientImpl) ConnectResponseWebSocket(ctx context.Context) (spec.ResponsesWebSocket, error) {
	endpoint, err := responsesEndpointURL(c.config.APIURL)
	if err != nil {
		return nil, err
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("openai responses websocket: invalid URL: %w", err)
	}
	switch parsed.Scheme {
	case "http":
		parsed.Scheme = "ws"
	case "https":
		parsed.Scheme = "wss"
	case "ws", "wss":
	default:
		return nil, fmt.Errorf("openai responses websocket: unsupported URL scheme %q", parsed.Scheme)
	}

	connection, response, err := websocket.Dial(ctx, parsed.String(), &websocket.DialOptions{
		HTTPClient: c.config.HTTPClient,
		HTTPHeader: c.headers(),
	})
	if err != nil {
		if response != nil {
			return nil, fmt.Errorf("openai responses websocket: handshake failed (status %d): %w", response.StatusCode, err)
		}
		return nil, fmt.Errorf("openai responses websocket: connect: %w", err)
	}
	connection.SetReadLimit(responsesWebSocketReadLimit)
	return &responseWebSocket{connection: connection}, nil
}

func (c *responseWebSocket) CreateResponse(ctx context.Context, request spec.ResponseCreateRequest) error {
	if request.Model == "" {
		return fmt.Errorf("openai responses websocket: model is required")
	}
	if request.Background != nil && *request.Background {
		return fmt.Errorf("openai responses websocket: background responses are not supported")
	}
	if _, exists := request.ExtraFields["background"]; exists {
		return fmt.Errorf("openai responses websocket: background is not supported")
	}
	if _, exists := request.ExtraFields["stream"]; exists {
		return fmt.Errorf("openai responses websocket: stream is implicit and must not be sent")
	}
	input, err := normalizeResponseInput(request.Input)
	if err != nil {
		return err
	}
	request.Input = input
	request.Stream = nil
	request.Background = nil
	return c.SendEvent(ctx, spec.ResponseWebSocketCreateEvent{
		Type:    "response.create",
		Request: request,
	})
}

func (c *responseWebSocket) SendEvent(ctx context.Context, event any) error {
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("openai responses websocket: encode client event: %w", err)
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if err := c.connection.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("openai responses websocket: write event: %w", err)
	}
	return nil
}

func (c *responseWebSocket) Receive(ctx context.Context) (spec.StreamEvent, error) {
	c.readMu.Lock()
	defer c.readMu.Unlock()
	_, data, err := c.connection.Read(ctx)
	if err != nil {
		return spec.StreamEvent{}, fmt.Errorf("openai responses websocket: read event: %w", err)
	}
	var event spec.StreamEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return spec.StreamEvent{}, fmt.Errorf("openai responses websocket: decode server event: %w", err)
	}
	event.Protocol = spec.ProtocolResponses
	event.Raw = append(event.Raw[:0], data...)
	return event, nil
}

func (c *responseWebSocket) Close() error {
	c.closeOnce.Do(func() {
		c.closeErr = c.connection.Close(websocket.StatusNormalClosure, "")
	})
	return c.closeErr
}

var _ spec.ResponsesWebSocket = (*responseWebSocket)(nil)
