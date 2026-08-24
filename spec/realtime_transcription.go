package spec

import (
	"context"
	"encoding/json"
	"fmt"
)

// RealtimeTranscriptionProtocol identifies the DashScope WebSocket protocol
// used by a realtime speech recognition model. Leave it empty to infer the
// protocol from the model name.
type RealtimeTranscriptionProtocol string

const (
	RealtimeTranscriptionProtocolAuto     RealtimeTranscriptionProtocol = ""
	RealtimeTranscriptionProtocolTask     RealtimeTranscriptionProtocol = "task"
	RealtimeTranscriptionProtocolRealtime RealtimeTranscriptionProtocol = "realtime"
)

const (
	RealtimeTranscriptionEventTaskStarted            = "task-started"
	RealtimeTranscriptionEventResultGenerated        = "result-generated"
	RealtimeTranscriptionEventTaskFinished           = "task-finished"
	RealtimeTranscriptionEventTaskFailed             = "task-failed"
	RealtimeTranscriptionEventSessionCreated         = "session.created"
	RealtimeTranscriptionEventSessionUpdated         = "session.updated"
	RealtimeTranscriptionEventSpeechStarted          = "input_audio_buffer.speech_started"
	RealtimeTranscriptionEventSpeechStopped          = "input_audio_buffer.speech_stopped"
	RealtimeTranscriptionEventAudioCommitted         = "input_audio_buffer.committed"
	RealtimeTranscriptionEventItemCreated            = "conversation.item.created"
	RealtimeTranscriptionEventTranscriptionText      = "conversation.item.input_audio_transcription.text"
	RealtimeTranscriptionEventTranscriptionCompleted = "conversation.item.input_audio_transcription.completed"
	RealtimeTranscriptionEventTranscriptionFailed    = "conversation.item.input_audio_transcription.failed"
	RealtimeTranscriptionEventSessionFinished        = "session.finished"
)

// RealtimeTranscriptionSession is one active bidirectional ASR session. One
// goroutine may call Receive while another sends audio or control events.
// Call Finish after the final audio chunk, keep receiving until the provider's
// terminal event, and then call Close.
type RealtimeTranscriptionSession interface {
	SendAudio(ctx context.Context, audio []byte) error
	Commit(ctx context.Context) error
	UpdateContext(ctx context.Context, messages []RealtimeTranscriptionContextMessage) error
	Finish(ctx context.Context) error
	Receive(ctx context.Context) (RealtimeTranscriptionEvent, error)
	Close() error
}

// RealtimeTranscriptionEventCallback receives every provider event in order.
// Returning an error stops the high-level streaming helper.
type RealtimeTranscriptionEventCallback func(ctx context.Context, event RealtimeTranscriptionEvent) error

// RealtimeTranscriptionTextCallback receives the best text for each result
// event. Final is true when the text is the final result for one utterance.
type RealtimeTranscriptionTextCallback func(ctx context.Context, text string, final bool) error

// RealtimeTranscriptionStreamOptions configures Client.StreamRealtimeTranscription.
type RealtimeTranscriptionStreamOptions struct {
	// ChunkSize is the number of audio bytes sent per frame. The default is
	// 3200, which represents 100ms of 16kHz, 16-bit, mono PCM audio.
	ChunkSize int
	OnEvent   RealtimeTranscriptionEventCallback
	OnText    RealtimeTranscriptionTextCallback
}

// RealtimeTranscriptionRequest configures a DashScope realtime ASR session.
// Parameters and Session provide forward-compatible protocol-specific fields;
// dedicated typed fields take precedence over entries with the same name.
type RealtimeTranscriptionRequest struct {
	Model      string
	Protocol   RealtimeTranscriptionProtocol
	Format     string
	SampleRate int
	// Headers adds optional handshake headers such as X-DashScope-WorkSpace.
	// Authorization and protocol-required headers are always set by the provider.
	Headers map[string]string

	// Task-protocol options used by Qwen-Audio 3.0, Fun-ASR, and Paraformer.
	LanguageHints              []string
	VocabularyID               string
	Vocabulary                 map[string]int
	Context                    []RealtimeTranscriptionContextMessage
	SemanticPunctuationEnabled *bool
	MaxSentenceSilence         int
	MultiThresholdModeEnabled  *bool
	Heartbeat                  *bool
	SpeechNoiseThreshold       *float64
	SpecialWordFilter          *RealtimeTranscriptionSpecialWordFilter
	Input                      map[string]any
	Parameters                 map[string]any

	// Realtime-protocol options used by Qwen3-ASR-Flash-Realtime.
	Language      string
	Manual        bool
	TurnDetection *RealtimeTranscriptionTurnDetection
	Modalities    []string
	Session       map[string]any

	// TaskID is optional and applies only to the task protocol. When empty, the
	// provider generates a UUID.
	TaskID string
}

// RealtimeTranscriptionContextMessage is one user/assistant context message
// used to improve recognition of prior conversation and domain terminology.
type RealtimeTranscriptionContextMessage struct {
	Role string
	Text string
}

// RealtimeTranscriptionTurnDetection configures Qwen3 server-side VAD. Set
// Request.Manual instead when utterance boundaries are committed by the client.
type RealtimeTranscriptionTurnDetection struct {
	Type              string
	Threshold         *float64
	SilenceDurationMS int
	ExtraFields       map[string]any
}

// RealtimeTranscriptionSpecialWordFilter controls replacement or removal of
// sensitive words for models that support this feature.
type RealtimeTranscriptionSpecialWordFilter struct {
	FilterWithSigned     *RealtimeTranscriptionWordList `json:"filter_with_signed,omitempty"`
	FilterWithEmpty      *RealtimeTranscriptionWordList `json:"filter_with_empty,omitempty"`
	SystemReservedFilter bool                           `json:"system_reserved_filter,omitempty"`
}

type RealtimeTranscriptionWordList struct {
	Words []string `json:"word_list"`
}

// RealtimeTranscriptionEvent contains a normalized view of either DashScope
// ASR protocol. Type remains the provider's original event name and Raw keeps
// the complete server JSON for fields introduced in the future.
type RealtimeTranscriptionEvent struct {
	Type           string
	EventID        string
	TaskID         string
	ItemID         string
	PreviousItemID string
	ContentIndex   int
	Language       string
	Emotion        string
	AudioStartMS   *int64
	AudioEndMS     *int64

	// Transcript is the best complete text for this event. For Qwen3 partial
	// events it is StableText + Stash; for final events it is the final result.
	Transcript string
	StableText string
	Stash      string
	Final      bool
	Terminal   bool

	Sentence *RealtimeTranscriptionSentence
	Usage    *RealtimeTranscriptionUsage
	Error    *RealtimeTranscriptionError
	Raw      json.RawMessage
}

type RealtimeTranscriptionSentence struct {
	BeginTime         int64                       `json:"begin_time"`
	EndTime           *int64                      `json:"end_time"`
	Text              string                      `json:"text"`
	SentenceBegin     bool                        `json:"sentence_begin"`
	SentenceEnd       bool                        `json:"sentence_end"`
	SentenceID        int64                       `json:"sentence_id"`
	Words             []RealtimeTranscriptionWord `json:"words"`
	Emotion           string                      `json:"emo_tag,omitempty"`
	EmotionConfidence *float64                    `json:"emo_confidence,omitempty"`
}

type RealtimeTranscriptionWord struct {
	BeginTime   int64  `json:"begin_time"`
	EndTime     *int64 `json:"end_time"`
	Text        string `json:"text"`
	Punctuation string `json:"punctuation"`
}

type RealtimeTranscriptionUsage struct {
	Duration float64 `json:"duration"`
}

// RealtimeTranscriptionError is returned inside task-failed, transcription
// failure, and generic error events.
type RealtimeTranscriptionError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Param   string `json:"param,omitempty"`
}

func (e *RealtimeTranscriptionError) Error() string {
	if e == nil {
		return ""
	}
	if e.Code == "" {
		return e.Message
	}
	if e.Message == "" {
		return e.Code
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}
