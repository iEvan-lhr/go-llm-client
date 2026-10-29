package spec

import (
	"context"
	"encoding/json"
	"io"
)

// AudioFile is an in-memory upload with a filename and MIME type.
type AudioFile = ImageFile

func NewAudioFile(path, contentType string) (AudioFile, error) {
	return NewImageFile(path, contentType)
}

func NewAudioFileBytes(filename, contentType string, data []byte) AudioFile {
	return NewImageFileBytes(filename, contentType, data)
}

type SpeechRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
	// Voice can be a built-in voice name or a custom voice reference.
	Voice          any            `json:"voice"`
	Instructions   string         `json:"instructions,omitempty"`
	ResponseFormat string         `json:"response_format,omitempty"`
	Speed          *float64       `json:"speed,omitempty"`
	ExtraFields    map[string]any `json:"-"`
}

func (r SpeechRequest) MarshalJSON() ([]byte, error) {
	type alias SpeechRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

type SpeechResponse struct {
	Data        []byte
	ContentType string
	RequestID   string
}

// SpeechStream streams the raw response bytes. The caller must close Body.
// The default response is encoded audio; stream_format=sse in ExtraFields
// returns the raw SSE bytes instead.
type SpeechStream struct {
	Body        io.ReadCloser
	ContentType string
	RequestID   string
}

type TranscriptionRequest struct {
	Model                  string
	File                   AudioFile
	Language               string
	Prompt                 string
	ResponseFormat         string
	Temperature            *float64
	TimestampGranularities []string
	Include                []string
	ChunkingStrategy       any
	KnownSpeakerNames      []string
	KnownSpeakerReferences []string
	// ExtraFields supports strings, numbers, arrays (sent as key[]), and
	// JSON objects for newly introduced multipart parameters.
	ExtraFields map[string]any
}

type AudioTranslationRequest struct {
	Model          string
	File           AudioFile
	Prompt         string
	ResponseFormat string
	Temperature    *float64
	ExtraFields    map[string]any
}

// TranscriptionResponse also represents audio translations and plain text,
// SRT, or VTT results. Text contains the complete text for non-JSON formats.
type TranscriptionResponse struct {
	Task        string                 `json:"task,omitempty"`
	Language    string                 `json:"language,omitempty"`
	Duration    float64                `json:"duration,omitempty"`
	Text        string                 `json:"text"`
	Words       []TranscriptionWord    `json:"words,omitempty"`
	Segments    []TranscriptionSegment `json:"segments,omitempty"`
	Logprobs    []json.RawMessage      `json:"logprobs,omitempty"`
	Usage       json.RawMessage        `json:"usage,omitempty"`
	RawResponse []byte                 `json:"-"`
}

type TranscriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type TranscriptionSegment struct {
	ID               any     `json:"id"`
	Seek             int     `json:"seek,omitempty"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Speaker          string  `json:"speaker,omitempty"`
	Tokens           []int   `json:"tokens,omitempty"`
	Temperature      float64 `json:"temperature,omitempty"`
	AvgLogprob       float64 `json:"avg_logprob,omitempty"`
	CompressionRatio float64 `json:"compression_ratio,omitempty"`
	NoSpeechProb     float64 `json:"no_speech_prob,omitempty"`
}

type TranscriptionEvent struct {
	Type      string            `json:"type"`
	Delta     string            `json:"delta,omitempty"`
	Text      string            `json:"text,omitempty"`
	SegmentID string            `json:"segment_id,omitempty"`
	Speaker   string            `json:"speaker,omitempty"`
	Start     float64           `json:"start,omitempty"`
	End       float64           `json:"end,omitempty"`
	Usage     json.RawMessage   `json:"usage,omitempty"`
	Logprobs  []json.RawMessage `json:"logprobs,omitempty"`
	Error     *APIError         `json:"error,omitempty"`
	Code      string            `json:"code,omitempty"`
	Message   string            `json:"message,omitempty"`
	Raw       json.RawMessage   `json:"-"`
}

type TranscriptionEventCallback func(context.Context, TranscriptionEvent) error
