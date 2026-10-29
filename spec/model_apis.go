package spec

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// EmbeddingRequest accepts text, batches of text, or token ID arrays.
type EmbeddingRequest struct {
	Model          string         `json:"model"`
	Input          any            `json:"input"`
	Dimensions     *int           `json:"dimensions,omitempty"`
	EncodingFormat string         `json:"encoding_format,omitempty"`
	User           string         `json:"user,omitempty"`
	ExtraFields    map[string]any `json:"-"`
}

func (r EmbeddingRequest) MarshalJSON() ([]byte, error) {
	type alias EmbeddingRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

// UnmarshalJSON accepts both float arrays and OpenAI's base64 encoded
// little-endian float32 vectors. Embedding always contains usable floats.
func (e *EmbeddingData) UnmarshalJSON(data []byte) error {
	var wire struct {
		Object    string          `json:"object"`
		Index     int             `json:"index"`
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*e = EmbeddingData{Object: wire.Object, Index: wire.Index}
	value := bytes.TrimSpace(wire.Embedding)
	if len(value) == 0 || value[0] != '"' {
		if len(value) == 0 {
			return nil
		}
		return json.Unmarshal(value, &e.Embedding)
	}
	if err := json.Unmarshal(value, &e.Base64Embedding); err != nil {
		return err
	}
	decoded, err := base64.StdEncoding.DecodeString(e.Base64Embedding)
	if err != nil {
		return fmt.Errorf("embedding: invalid base64: %w", err)
	}
	if len(decoded)%4 != 0 {
		return fmt.Errorf("embedding: base64 vector length is not a multiple of 4")
	}
	e.Embedding = make([]float32, len(decoded)/4)
	for i := range e.Embedding {
		e.Embedding[i] = math.Float32frombits(binary.LittleEndian.Uint32(decoded[i*4:]))
	}
	return nil
}

type ModelInfo struct {
	ID          string          `json:"id"`
	Object      string          `json:"object"`
	Created     int64           `json:"created"`
	OwnedBy     string          `json:"owned_by"`
	RawResponse json.RawMessage `json:"-"`
}

type ModelList struct {
	Object      string          `json:"object"`
	Data        []ModelInfo     `json:"data"`
	RawResponse json.RawMessage `json:"-"`
}

// ModerationRequest.Input accepts a string, []string, or multimodal inputs.
type ModerationRequest struct {
	Model       string         `json:"model,omitempty"`
	Input       any            `json:"input"`
	ExtraFields map[string]any `json:"-"`
}

func (r ModerationRequest) MarshalJSON() ([]byte, error) {
	type alias ModerationRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

type ModerationInput struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ModerationResponse struct {
	ID          string             `json:"id"`
	Model       string             `json:"model"`
	Results     []ModerationResult `json:"results"`
	RawResponse json.RawMessage    `json:"-"`
}

// Category maps preserve new categories without library updates.
type ModerationResult struct {
	Flagged                   bool                `json:"flagged"`
	Categories                map[string]bool     `json:"categories"`
	CategoryScores            map[string]float64  `json:"category_scores"`
	CategoryAppliedInputTypes map[string][]string `json:"category_applied_input_types,omitempty"`
}
