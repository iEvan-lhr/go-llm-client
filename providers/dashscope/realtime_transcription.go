package dashscope

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

const realtimeTranscriptionReadLimit = 16 * 1024 * 1024

type realtimeTranscriptionSession struct {
	connection *websocket.Conn
	protocol   spec.RealtimeTranscriptionProtocol
	taskID     string
	manual     bool

	readMu    sync.Mutex
	writeMu   sync.Mutex
	stateMu   sync.Mutex
	pending   []spec.RealtimeTranscriptionEvent
	finished  bool
	closed    bool
	closeOnce sync.Once
	closeErr  error
}

// StartRealtimeTranscription opens and initializes a DashScope realtime ASR
// session. It waits for task-started or session.updated before returning, so
// audio can be sent immediately. Initialization events remain available from
// Receive in their original order.
func (c *clientImpl) StartRealtimeTranscription(ctx context.Context, request spec.RealtimeTranscriptionRequest) (spec.RealtimeTranscriptionSession, error) {
	normalized, err := normalizeRealtimeTranscriptionRequest(request)
	if err != nil {
		return nil, err
	}

	endpoint, err := realtimeTranscriptionEndpoint(c.config.APIURL, normalized.Protocol, normalized.Model)
	if err != nil {
		return nil, err
	}
	headers := http.Header{}
	for name, value := range normalized.Headers {
		headers.Set(name, value)
	}
	headers.Set("Authorization", "Bearer "+c.config.APIKey)
	if headers.Get("User-Agent") == "" {
		headers.Set("User-Agent", "go-llm-client")
	}
	if normalized.Protocol == spec.RealtimeTranscriptionProtocolRealtime {
		headers.Set("OpenAI-Beta", "realtime=v1")
	}

	connection, response, err := websocket.Dial(ctx, endpoint, &websocket.DialOptions{
		HTTPClient: c.config.HTTPClient,
		HTTPHeader: headers,
	})
	if err != nil {
		if response != nil {
			return nil, fmt.Errorf("dashscope realtime transcription: handshake failed (status %d): %w", response.StatusCode, err)
		}
		return nil, fmt.Errorf("dashscope realtime transcription: connect: %w", err)
	}
	connection.SetReadLimit(realtimeTranscriptionReadLimit)

	session := &realtimeTranscriptionSession{
		connection: connection,
		protocol:   normalized.Protocol,
		taskID:     normalized.TaskID,
		manual:     normalized.Manual,
	}
	if err := session.initialize(ctx, normalized); err != nil {
		_ = connection.Close(websocket.StatusInternalError, "initialization failed")
		return nil, err
	}
	return session, nil
}

func normalizeRealtimeTranscriptionRequest(request spec.RealtimeTranscriptionRequest) (spec.RealtimeTranscriptionRequest, error) {
	request.Model = strings.TrimSpace(request.Model)
	if request.Model == "" {
		return request, fmt.Errorf("dashscope realtime transcription: model is required")
	}
	if request.Protocol == spec.RealtimeTranscriptionProtocolAuto {
		request.Protocol = realtimeTranscriptionProtocolForModel(request.Model)
	}
	if request.Protocol != spec.RealtimeTranscriptionProtocolTask && request.Protocol != spec.RealtimeTranscriptionProtocolRealtime {
		return request, fmt.Errorf("dashscope realtime transcription: cannot infer protocol for model %q; set Protocol explicitly", request.Model)
	}

	request.Format = strings.ToLower(strings.TrimSpace(request.Format))
	if request.Format == "" {
		request.Format = "pcm"
	}
	if request.SampleRate == 0 {
		request.SampleRate = 16000
		if request.Protocol == spec.RealtimeTranscriptionProtocolTask && strings.Contains(strings.ToLower(request.Model), "8k") {
			request.SampleRate = 8000
		}
	}
	if request.SampleRate < 1 {
		return request, fmt.Errorf("dashscope realtime transcription: sample rate must be positive")
	}
	if request.Manual && request.TurnDetection != nil {
		return request, fmt.Errorf("dashscope realtime transcription: Manual and TurnDetection cannot both be set")
	}

	if request.Protocol == spec.RealtimeTranscriptionProtocolRealtime {
		if request.Format != "pcm" && request.Format != "opus" {
			return request, fmt.Errorf("dashscope realtime transcription: realtime protocol supports pcm or opus, got %q", request.Format)
		}
		if request.SampleRate != 8000 && request.SampleRate != 16000 {
			return request, fmt.Errorf("dashscope realtime transcription: realtime protocol supports sample rates 8000 or 16000, got %d", request.SampleRate)
		}
	} else {
		switch request.Format {
		case "pcm", "wav", "mp3", "opus", "speex", "aac", "amr":
		default:
			return request, fmt.Errorf("dashscope realtime transcription: unsupported task-protocol audio format %q", request.Format)
		}
	}
	if request.Protocol == spec.RealtimeTranscriptionProtocolTask && strings.Contains(strings.ToLower(request.Model), "8k") && request.SampleRate != 8000 {
		return request, fmt.Errorf("dashscope realtime transcription: 8k model %q requires sample rate 8000", request.Model)
	}
	if request.MaxSentenceSilence != 0 && (request.MaxSentenceSilence < 200 || request.MaxSentenceSilence > 6000) {
		return request, fmt.Errorf("dashscope realtime transcription: MaxSentenceSilence must be between 200 and 6000 ms")
	}
	if request.TurnDetection != nil {
		if request.TurnDetection.Threshold != nil && (*request.TurnDetection.Threshold < -1 || *request.TurnDetection.Threshold > 1) {
			return request, fmt.Errorf("dashscope realtime transcription: VAD threshold must be between -1 and 1")
		}
		if request.TurnDetection.SilenceDurationMS != 0 && (request.TurnDetection.SilenceDurationMS < 200 || request.TurnDetection.SilenceDurationMS > 6000) {
			return request, fmt.Errorf("dashscope realtime transcription: VAD silence duration must be between 200 and 6000 ms")
		}
	}
	for _, message := range request.Context {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		if role != "user" && role != "assistant" {
			return request, fmt.Errorf("dashscope realtime transcription: context role must be user or assistant, got %q", message.Role)
		}
		if message.Text == "" {
			return request, fmt.Errorf("dashscope realtime transcription: context text cannot be empty")
		}
	}
	if request.TaskID == "" && request.Protocol == spec.RealtimeTranscriptionProtocolTask {
		var err error
		request.TaskID, err = newRealtimeTranscriptionID("")
		if err != nil {
			return request, fmt.Errorf("dashscope realtime transcription: generate task ID: %w", err)
		}
	}
	return request, nil
}

func realtimeTranscriptionProtocolForModel(model string) spec.RealtimeTranscriptionProtocol {
	model = strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.HasPrefix(model, "qwen3-asr-flash-realtime"):
		return spec.RealtimeTranscriptionProtocolRealtime
	case strings.HasPrefix(model, "qwen-audio-3.0-asr-flash-streaming"),
		strings.HasPrefix(model, "fun-asr-realtime"),
		strings.HasPrefix(model, "fun-asr-flash-8k-realtime"),
		strings.HasPrefix(model, "paraformer-realtime"):
		return spec.RealtimeTranscriptionProtocolTask
	default:
		return spec.RealtimeTranscriptionProtocolAuto
	}
}

func realtimeTranscriptionEndpoint(apiURL string, protocol spec.RealtimeTranscriptionProtocol, model string) (string, error) {
	parsed, err := url.Parse(apiURL)
	if err != nil {
		return "", fmt.Errorf("dashscope realtime transcription: invalid API URL: %w", err)
	}
	switch parsed.Scheme {
	case "http":
		parsed.Scheme = "ws"
	case "https":
		parsed.Scheme = "wss"
	case "ws", "wss":
	default:
		return "", fmt.Errorf("dashscope realtime transcription: unsupported API URL scheme %q", parsed.Scheme)
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("dashscope realtime transcription: API URL host is required")
	}

	desiredPath := "/api-ws/v1/inference"
	if protocol == spec.RealtimeTranscriptionProtocolRealtime {
		desiredPath = "/api-ws/v1/realtime"
	}
	cleanPath := strings.TrimRight(parsed.Path, "/")
	switch {
	case cleanPath == "",
		cleanPath == "/compatible-mode/v1",
		cleanPath == "/compatible-mode/v1/chat/completions",
		cleanPath == "/api/v1",
		cleanPath == "/v1",
		cleanPath == "/api-ws/v1/inference",
		cleanPath == "/api-ws/v1/realtime":
		parsed.Path = desiredPath
	}
	if protocol == spec.RealtimeTranscriptionProtocolRealtime {
		query := parsed.Query()
		query.Set("model", model)
		parsed.RawQuery = query.Encode()
	}
	return parsed.String(), nil
}

func (s *realtimeTranscriptionSession) initialize(ctx context.Context, request spec.RealtimeTranscriptionRequest) error {
	var event any
	var readyType string
	if s.protocol == spec.RealtimeTranscriptionProtocolTask {
		event = buildTaskStartEvent(request)
		readyType = "task-started"
	} else {
		built, err := buildRealtimeSessionUpdateEvent(request)
		if err != nil {
			return err
		}
		event = built
		readyType = "session.updated"
	}
	if err := s.writeJSON(ctx, event); err != nil {
		return fmt.Errorf("dashscope realtime transcription: initialize: %w", err)
	}

	for {
		received, err := s.receiveOne(ctx)
		if err != nil {
			return fmt.Errorf("dashscope realtime transcription: wait for %s: %w", readyType, err)
		}
		s.pending = append(s.pending, received)
		if received.Type == readyType {
			return nil
		}
		if received.Error != nil {
			return fmt.Errorf("dashscope realtime transcription: session initialization failed: %w", received.Error)
		}
	}
}

func buildTaskStartEvent(request spec.RealtimeTranscriptionRequest) map[string]any {
	parameters := cloneAnyMap(request.Parameters)
	parameters["format"] = request.Format
	parameters["sample_rate"] = request.SampleRate
	if len(request.LanguageHints) > 0 {
		parameters["language_hints"] = append([]string(nil), request.LanguageHints...)
	}
	if request.VocabularyID != "" {
		parameters["vocabulary_id"] = request.VocabularyID
	}
	if len(request.Vocabulary) > 0 {
		parameters["vocabulary"] = request.Vocabulary
	}
	if request.SemanticPunctuationEnabled != nil {
		parameters["semantic_punctuation_enabled"] = *request.SemanticPunctuationEnabled
	}
	if request.MaxSentenceSilence != 0 {
		parameters["max_sentence_silence"] = request.MaxSentenceSilence
	}
	if request.MultiThresholdModeEnabled != nil {
		parameters["multi_threshold_mode_enabled"] = *request.MultiThresholdModeEnabled
	}
	if request.Heartbeat != nil {
		parameters["heartbeat"] = *request.Heartbeat
	}
	if request.SpeechNoiseThreshold != nil {
		parameters["speech_noise_threshold"] = *request.SpeechNoiseThreshold
	}
	if request.SpecialWordFilter != nil {
		parameters["special_word_filter"] = request.SpecialWordFilter
	}

	input := cloneAnyMap(request.Input)
	if len(request.Context) > 0 {
		input["context"] = buildRealtimeTranscriptionContext(request.Context)
	}
	return map[string]any{
		"header": map[string]any{
			"action":    "run-task",
			"task_id":   request.TaskID,
			"streaming": "duplex",
		},
		"payload": map[string]any{
			"task_group": "audio",
			"task":       "asr",
			"function":   "recognition",
			"model":      request.Model,
			"parameters": parameters,
			"input":      input,
		},
	}
}

func buildRealtimeSessionUpdateEvent(request spec.RealtimeTranscriptionRequest) (map[string]any, error) {
	eventID, err := newRealtimeTranscriptionID("event_")
	if err != nil {
		return nil, fmt.Errorf("dashscope realtime transcription: generate event ID: %w", err)
	}
	session := cloneAnyMap(request.Session)
	session["input_audio_format"] = request.Format
	session["sample_rate"] = request.SampleRate
	if len(request.Modalities) > 0 {
		session["modalities"] = append([]string(nil), request.Modalities...)
	} else if _, exists := session["modalities"]; !exists {
		session["modalities"] = []string{"text"}
	}
	if request.Language != "" {
		transcription := map[string]any{}
		if configured, ok := session["input_audio_transcription"].(map[string]any); ok {
			transcription = cloneAnyMap(configured)
		}
		transcription["language"] = request.Language
		session["input_audio_transcription"] = transcription
	}
	if request.Manual {
		session["turn_detection"] = nil
	} else if request.TurnDetection != nil {
		turnDetection := cloneAnyMap(request.TurnDetection.ExtraFields)
		turnDetection["type"] = request.TurnDetection.Type
		if request.TurnDetection.Type == "" {
			turnDetection["type"] = "server_vad"
		}
		if request.TurnDetection.Threshold != nil {
			turnDetection["threshold"] = *request.TurnDetection.Threshold
		}
		if request.TurnDetection.SilenceDurationMS != 0 {
			turnDetection["silence_duration_ms"] = request.TurnDetection.SilenceDurationMS
		}
		session["turn_detection"] = turnDetection
	}
	return map[string]any{
		"event_id": eventID,
		"type":     "session.update",
		"session":  session,
	}, nil
}

func buildRealtimeTranscriptionContext(messages []spec.RealtimeTranscriptionContextMessage) []map[string]any {
	contextMessages := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		contentType := "input_text"
		if role == "assistant" {
			contentType = "text"
		}
		contextMessages = append(contextMessages, map[string]any{
			"role": role,
			"content": []map[string]any{{
				"type": contentType,
				"text": message.Text,
			}},
		})
	}
	return contextMessages
}

func cloneAnyMap(source map[string]any) map[string]any {
	result := make(map[string]any, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func (s *realtimeTranscriptionSession) SendAudio(ctx context.Context, audio []byte) error {
	if len(audio) == 0 {
		return fmt.Errorf("dashscope realtime transcription: audio chunk cannot be empty")
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.ensureWritable(); err != nil {
		return err
	}
	if s.protocol == spec.RealtimeTranscriptionProtocolTask {
		if err := s.connection.Write(ctx, websocket.MessageBinary, audio); err != nil {
			return fmt.Errorf("dashscope realtime transcription: send audio: %w", err)
		}
		return nil
	}
	eventID, err := newRealtimeTranscriptionID("event_")
	if err != nil {
		return fmt.Errorf("dashscope realtime transcription: generate event ID: %w", err)
	}
	return s.writeJSONLocked(ctx, map[string]any{
		"event_id": eventID,
		"type":     "input_audio_buffer.append",
		"audio":    base64.StdEncoding.EncodeToString(audio),
	})
}

func (s *realtimeTranscriptionSession) Commit(ctx context.Context) error {
	if s.protocol != spec.RealtimeTranscriptionProtocolRealtime {
		return fmt.Errorf("dashscope realtime transcription: Commit is only supported by the realtime protocol")
	}
	if !s.manual {
		return fmt.Errorf("dashscope realtime transcription: Commit is disabled while server VAD is enabled")
	}
	eventID, err := newRealtimeTranscriptionID("event_")
	if err != nil {
		return fmt.Errorf("dashscope realtime transcription: generate event ID: %w", err)
	}
	return s.writeJSON(ctx, map[string]any{
		"event_id": eventID,
		"type":     "input_audio_buffer.commit",
	})
}

func (s *realtimeTranscriptionSession) UpdateContext(ctx context.Context, messages []spec.RealtimeTranscriptionContextMessage) error {
	if s.protocol != spec.RealtimeTranscriptionProtocolTask {
		return fmt.Errorf("dashscope realtime transcription: UpdateContext is only supported by the task protocol")
	}
	for _, message := range messages {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		if role != "user" && role != "assistant" {
			return fmt.Errorf("dashscope realtime transcription: context role must be user or assistant, got %q", message.Role)
		}
		if message.Text == "" {
			return fmt.Errorf("dashscope realtime transcription: context text cannot be empty")
		}
	}
	return s.writeJSON(ctx, map[string]any{
		"header": map[string]any{
			"action":    "continue-task",
			"task_id":   s.taskID,
			"streaming": "duplex",
		},
		"payload": map[string]any{
			"input": map[string]any{
				"context": buildRealtimeTranscriptionContext(messages),
			},
		},
	})
}

func (s *realtimeTranscriptionSession) Finish(ctx context.Context) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.ensureWritable(); err != nil {
		return err
	}
	var event any
	if s.protocol == spec.RealtimeTranscriptionProtocolTask {
		event = map[string]any{
			"header": map[string]any{
				"action":    "finish-task",
				"task_id":   s.taskID,
				"streaming": "duplex",
			},
			"payload": map[string]any{"input": map[string]any{}},
		}
	} else {
		eventID, err := newRealtimeTranscriptionID("event_")
		if err != nil {
			return fmt.Errorf("dashscope realtime transcription: generate event ID: %w", err)
		}
		event = map[string]any{"event_id": eventID, "type": "session.finish"}
	}
	if err := s.writeJSONLocked(ctx, event); err != nil {
		return err
	}
	s.stateMu.Lock()
	s.finished = true
	s.stateMu.Unlock()
	return nil
}

func (s *realtimeTranscriptionSession) Receive(ctx context.Context) (spec.RealtimeTranscriptionEvent, error) {
	s.readMu.Lock()
	defer s.readMu.Unlock()
	if len(s.pending) > 0 {
		event := s.pending[0]
		s.pending = s.pending[1:]
		return event, nil
	}
	return s.receiveOne(ctx)
}

func (s *realtimeTranscriptionSession) receiveOne(ctx context.Context) (spec.RealtimeTranscriptionEvent, error) {
	messageType, data, err := s.connection.Read(ctx)
	if err != nil {
		return spec.RealtimeTranscriptionEvent{}, fmt.Errorf("dashscope realtime transcription: receive event: %w", err)
	}
	if messageType != websocket.MessageText {
		return spec.RealtimeTranscriptionEvent{}, fmt.Errorf("dashscope realtime transcription: expected text event, got WebSocket message type %d", messageType)
	}
	if s.protocol == spec.RealtimeTranscriptionProtocolTask {
		return decodeTaskTranscriptionEvent(data)
	}
	return decodeRealtimeTranscriptionEvent(data)
}

func decodeTaskTranscriptionEvent(data []byte) (spec.RealtimeTranscriptionEvent, error) {
	var wire struct {
		Header struct {
			TaskID       string `json:"task_id"`
			Event        string `json:"event"`
			ErrorCode    string `json:"error_code"`
			ErrorMessage string `json:"error_message"`
		} `json:"header"`
		Payload struct {
			Output struct {
				Sentence *spec.RealtimeTranscriptionSentence `json:"sentence"`
			} `json:"output"`
			Usage *spec.RealtimeTranscriptionUsage `json:"usage"`
		} `json:"payload"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return spec.RealtimeTranscriptionEvent{}, fmt.Errorf("dashscope realtime transcription: decode task event: %w", err)
	}
	event := spec.RealtimeTranscriptionEvent{
		Type:     wire.Header.Event,
		TaskID:   wire.Header.TaskID,
		Sentence: wire.Payload.Output.Sentence,
		Usage:    wire.Payload.Usage,
		Raw:      append(json.RawMessage(nil), data...),
	}
	if event.Sentence != nil {
		event.Transcript = event.Sentence.Text
		event.Final = event.Sentence.SentenceEnd
		event.Emotion = event.Sentence.Emotion
	}
	event.Terminal = wire.Header.Event == spec.RealtimeTranscriptionEventTaskFinished || wire.Header.Event == spec.RealtimeTranscriptionEventTaskFailed
	if wire.Header.Event == "task-failed" || wire.Header.ErrorCode != "" || wire.Header.ErrorMessage != "" {
		event.Error = &spec.RealtimeTranscriptionError{Code: wire.Header.ErrorCode, Message: wire.Header.ErrorMessage}
		if event.Error.Message == "" {
			event.Error.Message = "DashScope realtime transcription task failed"
		}
	}
	return event, nil
}

func decodeRealtimeTranscriptionEvent(data []byte) (spec.RealtimeTranscriptionEvent, error) {
	var wire struct {
		Type           string                           `json:"type"`
		EventID        string                           `json:"event_id"`
		ItemID         string                           `json:"item_id"`
		PreviousItemID string                           `json:"previous_item_id"`
		ContentIndex   int                              `json:"content_index"`
		Language       string                           `json:"language"`
		Emotion        string                           `json:"emotion"`
		AudioStartMS   *int64                           `json:"audio_start_ms"`
		AudioEndMS     *int64                           `json:"audio_end_ms"`
		Text           string                           `json:"text"`
		Stash          string                           `json:"stash"`
		Transcript     string                           `json:"transcript"`
		Error          *spec.RealtimeTranscriptionError `json:"error"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return spec.RealtimeTranscriptionEvent{}, fmt.Errorf("dashscope realtime transcription: decode realtime event: %w", err)
	}
	event := spec.RealtimeTranscriptionEvent{
		Type:           wire.Type,
		EventID:        wire.EventID,
		ItemID:         wire.ItemID,
		PreviousItemID: wire.PreviousItemID,
		ContentIndex:   wire.ContentIndex,
		Language:       wire.Language,
		Emotion:        wire.Emotion,
		AudioStartMS:   wire.AudioStartMS,
		AudioEndMS:     wire.AudioEndMS,
		StableText:     wire.Text,
		Stash:          wire.Stash,
		Error:          wire.Error,
		Raw:            append(json.RawMessage(nil), data...),
	}
	switch wire.Type {
	case spec.RealtimeTranscriptionEventTranscriptionText:
		event.Transcript = wire.Text + wire.Stash
	case spec.RealtimeTranscriptionEventTranscriptionCompleted:
		event.Transcript = wire.Transcript
		event.Final = true
	case spec.RealtimeTranscriptionEventTranscriptionFailed, "error":
		if event.Error == nil {
			event.Error = &spec.RealtimeTranscriptionError{Message: "DashScope realtime transcription failed"}
		}
	}
	event.Terminal = wire.Type == spec.RealtimeTranscriptionEventSessionFinished
	return event, nil
}

func (s *realtimeTranscriptionSession) writeJSON(ctx context.Context, event any) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.ensureWritable(); err != nil {
		return err
	}
	return s.writeJSONLocked(ctx, event)
}

func (s *realtimeTranscriptionSession) writeJSONLocked(ctx context.Context, event any) error {
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("dashscope realtime transcription: encode event: %w", err)
	}
	if err := s.connection.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("dashscope realtime transcription: send event: %w", err)
	}
	return nil
}

func (s *realtimeTranscriptionSession) ensureWritable() error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.closed {
		return fmt.Errorf("dashscope realtime transcription: session is closed")
	}
	if s.finished {
		return fmt.Errorf("dashscope realtime transcription: session has already been finished")
	}
	return nil
}

func (s *realtimeTranscriptionSession) Close() error {
	s.closeOnce.Do(func() {
		s.writeMu.Lock()
		defer s.writeMu.Unlock()
		s.stateMu.Lock()
		s.closed = true
		s.stateMu.Unlock()
		s.closeErr = s.connection.Close(websocket.StatusNormalClosure, "")
	})
	return s.closeErr
}

func newRealtimeTranscriptionID(prefix string) (string, error) {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	if prefix != "" {
		return prefix + hex.EncodeToString(bytes), nil
	}
	bytes[6] = (bytes[6] & 0x0f) | 0x40
	bytes[8] = (bytes[8] & 0x3f) | 0x80
	return fmt.Sprintf("%s-%s-%s-%s-%s",
		hex.EncodeToString(bytes[0:4]),
		hex.EncodeToString(bytes[4:6]),
		hex.EncodeToString(bytes[6:8]),
		hex.EncodeToString(bytes[8:10]),
		hex.EncodeToString(bytes[10:16])), nil
}

var _ spec.RealtimeTranscriptionClient = (*clientImpl)(nil)
var _ spec.RealtimeTranscriptionSession = (*realtimeTranscriptionSession)(nil)
