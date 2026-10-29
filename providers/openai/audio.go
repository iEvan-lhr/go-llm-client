package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

func (c *clientImpl) CreateSpeechStream(ctx context.Context, request spec.SpeechRequest) (*spec.SpeechStream, error) {
	if strings.TrimSpace(request.Model) == "" || strings.TrimSpace(request.Input) == "" || request.Voice == nil || request.Voice == "" {
		return nil, fmt.Errorf("openai audio: speech requires model, input, and voice")
	}
	endpoint, err := apiEndpointURL(c.config.APIURL, "audio/speech")
	if err != nil {
		return nil, err
	}
	response, err := c.requester.PostStream(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	return &spec.SpeechStream{Body: response.Body, ContentType: response.Header.Get("Content-Type"), RequestID: response.Header.Get("X-Request-ID")}, nil
}

func (c *clientImpl) CreateSpeech(ctx context.Context, request spec.SpeechRequest) (*spec.SpeechResponse, error) {
	response, err := c.CreateSpeechStream(ctx, request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("openai audio: read speech: %w", err)
	}
	return &spec.SpeechResponse{Data: data, ContentType: response.ContentType, RequestID: response.RequestID}, nil
}

func (c *clientImpl) CreateTranscription(ctx context.Context, request spec.TranscriptionRequest) (*spec.TranscriptionResponse, error) {
	return c.transcribe(ctx, request, false, nil)
}

func (c *clientImpl) CreateTranscriptionStream(ctx context.Context, request spec.TranscriptionRequest, callback spec.TranscriptionEventCallback) (*spec.TranscriptionResponse, error) {
	return c.transcribe(ctx, request, true, callback)
}

func (c *clientImpl) transcribe(ctx context.Context, request spec.TranscriptionRequest, stream bool, callback spec.TranscriptionEventCallback) (*spec.TranscriptionResponse, error) {
	fields := cloneParameters(request.ExtraFields)
	fields["model"] = request.Model
	setAudioField(fields, "language", request.Language)
	setAudioField(fields, "prompt", request.Prompt)
	setAudioField(fields, "response_format", request.ResponseFormat)
	if request.Temperature != nil {
		fields["temperature"] = *request.Temperature
	}
	if request.ChunkingStrategy != nil {
		fields["chunking_strategy"] = request.ChunkingStrategy
	}
	if len(request.TimestampGranularities) > 0 {
		fields["timestamp_granularities"] = request.TimestampGranularities
	}
	if len(request.Include) > 0 {
		fields["include"] = request.Include
	}
	if len(request.KnownSpeakerNames) > 0 {
		fields["known_speaker_names"] = request.KnownSpeakerNames
	}
	if len(request.KnownSpeakerReferences) > 0 {
		fields["known_speaker_references"] = request.KnownSpeakerReferences
	}
	// Explicit streaming methods determine the response parser.
	fields["stream"] = stream
	response, err := c.audioMultipart(ctx, "transcriptions", request.File, fields)
	if err != nil {
		return nil, err
	}
	if stream {
		return consumeTranscriptionStream(ctx, response, callback)
	}
	return decodeTranscription(response, fmt.Sprint(fields["response_format"]))
}

func (c *clientImpl) CreateTranslation(ctx context.Context, request spec.AudioTranslationRequest) (*spec.TranscriptionResponse, error) {
	fields := cloneParameters(request.ExtraFields)
	fields["model"] = request.Model
	setAudioField(fields, "prompt", request.Prompt)
	setAudioField(fields, "response_format", request.ResponseFormat)
	if request.Temperature != nil {
		fields["temperature"] = *request.Temperature
	}
	response, err := c.audioMultipart(ctx, "translations", request.File, fields)
	if err != nil {
		return nil, err
	}
	return decodeTranscription(response, fmt.Sprint(fields["response_format"]))
}

func setAudioField(fields map[string]any, key, value string) {
	if value != "" {
		fields[key] = value
	}
}

func (c *clientImpl) audioMultipart(ctx context.Context, operation string, file spec.AudioFile, fields map[string]any) (*http.Response, error) {
	model, _ := fields["model"].(string)
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("openai audio: model is required")
	}
	endpoint, err := apiEndpointURL(c.config.APIURL, "audio/"+operation)
	if err != nil {
		return nil, err
	}
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	for key, value := range fields {
		if err := writeMultipartValue(writer, key, value); err != nil {
			return nil, err
		}
	}
	if file.Filename == "" {
		file.Filename = "audio"
	}
	if err := writeImagePart(writer, "file", file); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	headers := c.headers()
	headers.Set("Content-Type", writer.FormDataContentType())
	return c.requester.DoBodyStream(ctx, http.MethodPost, endpoint, headers, body)
}

func writeMultipartValue(writer *multipart.Writer, key string, value any) error {
	if value == nil {
		return nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("openai multipart: field %s: %w", key, err)
	}
	if len(encoded) > 0 && encoded[0] == '[' {
		var values []json.RawMessage
		if err := json.Unmarshal(encoded, &values); err != nil {
			return err
		}
		for _, item := range values {
			if err := writeMultipartValue(writer, strings.TrimSuffix(key, "[]")+"[]", item); err != nil {
				return err
			}
		}
		return nil
	}
	text := string(encoded)
	if len(encoded) > 0 && encoded[0] == '"' {
		if err := json.Unmarshal(encoded, &text); err != nil {
			return err
		}
	}
	return writer.WriteField(key, text)
}

func decodeTranscription(response *http.Response, format string) (*spec.TranscriptionResponse, error) {
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("openai audio: read transcription: %w", err)
	}
	var result spec.TranscriptionResponse
	switch format {
	case "text", "srt", "vtt":
		result.Text = string(raw)
	default:
		if err := decodeAPIResponse(raw, &result); err != nil {
			return nil, err
		}
	}
	result.RawResponse = raw
	return &result, nil
}

func consumeTranscriptionStream(ctx context.Context, response *http.Response, callback spec.TranscriptionEventCallback) (*spec.TranscriptionResponse, error) {
	defer response.Body.Close()
	result := &spec.TranscriptionResponse{}
	var text strings.Builder
	completed := false
	err := scanSSE(response, func(data string) error {
		if data == "[DONE]" {
			return errSSEDone
		}
		var event spec.TranscriptionEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return fmt.Errorf("openai audio: decode stream event: %w", err)
		}
		event.Raw = json.RawMessage(data)
		if callback != nil {
			if err := callback(ctx, event); err != nil {
				return err
			}
		}
		if event.Error != nil {
			return event.Error
		}
		if event.Type == "error" {
			return &spec.APIError{Code: event.Code, Message: event.Message}
		}
		switch event.Type {
		case "transcript.text.delta":
			text.WriteString(event.Delta)
		case "transcript.text.segment":
			result.Segments = append(result.Segments, spec.TranscriptionSegment{ID: event.SegmentID, Text: event.Text, Speaker: event.Speaker, Start: event.Start, End: event.End})
		case "transcript.text.done":
			result.Text, result.Usage, result.Logprobs = event.Text, event.Usage, event.Logprobs
			if result.Text == "" {
				result.Text = text.String()
			}
			completed = true
			return errSSEDone
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if !completed {
		return nil, fmt.Errorf("openai audio: stream ended before transcript.text.done: %w", io.ErrUnexpectedEOF)
	}
	result.RawResponse, _ = json.Marshal(result)
	return result, nil
}
