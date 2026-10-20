package spec

import "encoding/json"

// SystemOneQuestionType identifies a structured decision, not a text completion.
type SystemOneQuestionType string

const (
	SystemOneNoul   SystemOneQuestionType = "noul"
	SystemOneChoice SystemOneQuestionType = "choice"
	SystemOneScore  SystemOneQuestionType = "score"
)

// SystemOneQuestion describes one question about the shared state.
// Instructions accepts text, an object, or an array. Criteria is optional for
// noul (an object with "true"/"false" descriptions), required for choice (an
// object mapping option IDs to descriptions), and required for score (an ordered
// array of rubric descriptions, indexed from zero). Descriptions may be structured.
type SystemOneQuestion struct {
	Type         SystemOneQuestionType `json:"type"`
	Instructions any                   `json:"instructions,omitempty"`
	Criteria     any                   `json:"criteria,omitempty"`
}

// SystemOneRequest evaluates all named questions against the same state.
// State accepts a string, object, or array. Model may be omitted to use the
// configured model, or the provider's default decision model.
type SystemOneRequest struct {
	Model     string                       `json:"model"`
	State     any                          `json:"state"`
	Questions map[string]SystemOneQuestion `json:"questions"`
}

// SystemOneAnswer preserves the decision's native numeric values. Noul is the
// probability of yes (0..1), not a bool. Score is a probability-weighted average
// of rubric indices and may be fractional. Pointers distinguish zero from absent.
type SystemOneAnswer struct {
	Type          SystemOneQuestionType      `json:"type"`
	Noul          *float64                   `json:"noul,omitempty"`
	Choice        string                     `json:"choice,omitempty"`
	Score         *float64                   `json:"score,omitempty"`
	Confidence    *float64                   `json:"confidence,omitempty"`
	Probabilities map[string]float64         `json:"probabilities,omitempty"`
	Legend        map[string]json.RawMessage `json:"legend,omitempty"`
}

type SystemOneUsage struct {
	InputTokens int `json:"input_tokens"`
	// OutputTokens is zero when omitted by the provider; RawResponse preserves
	// whether the field was reported.
	OutputTokens int `json:"output_tokens"`
}

// SystemOneResponse contains one answer per question ID and the original JSON.
type SystemOneResponse struct {
	Model     string `json:"model"`
	RequestID string `json:"request_id,omitempty"`
	// LatencyMS is the server-reported latency in milliseconds, when supplied.
	LatencyMS   *float64                   `json:"latency_ms,omitempty"`
	Answers     map[string]SystemOneAnswer `json:"answers"`
	Usage       SystemOneUsage             `json:"usage"`
	RawResponse []byte                     `json:"-"`
}
