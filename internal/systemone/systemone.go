// Package systemone handles the shared System One JSON protocol.
package systemone

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/internal/requester"
	"github.com/iEvan-lhr/go-llm-client/spec"
)

// Do sends an already-routed request. Providers supply model defaults,
// endpoint selection, and authentication headers.
func Do(ctx context.Context, httpRequester *requester.Requester, endpoint string, headers http.Header, request spec.SystemOneRequest) (*spec.SystemOneResponse, error) {
	// Marshal once to snapshot caller-owned maps and validate the actual wire
	// representation, including typed maps/slices and json.RawMessage values.
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("systemone: encode request: %w", err)
	}
	var wire struct {
		State     json.RawMessage `json:"state"`
		Questions map[string]struct {
			Type     spec.SystemOneQuestionType `json:"type"`
			Criteria json.RawMessage            `json:"criteria"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, fmt.Errorf("systemone: decode request: %w", err)
	}
	if len(wire.State) == 0 || !strings.ContainsRune("\"{[", rune(wire.State[0])) {
		return nil, fmt.Errorf("systemone: state must be a string, object, or array")
	}
	if len(wire.Questions) == 0 {
		return nil, fmt.Errorf("systemone: at least one question is required")
	}
	for id, question := range wire.Questions {
		switch question.Type {
		case spec.SystemOneNoul:
			if len(question.Criteria) > 0 && string(question.Criteria) != "null" && question.Criteria[0] != '{' {
				return nil, fmt.Errorf("systemone: noul question %q criteria must be an object", id)
			}
		case spec.SystemOneChoice:
			var criteria map[string]json.RawMessage
			if json.Unmarshal(question.Criteria, &criteria) != nil || len(criteria) < 1 || len(criteria) > 255 {
				return nil, fmt.Errorf("systemone: choice question %q requires 1..255 named criteria", id)
			}
		case spec.SystemOneScore:
			var criteria []json.RawMessage
			if json.Unmarshal(question.Criteria, &criteria) != nil || len(criteria) < 1 || len(criteria) > 10 {
				return nil, fmt.Errorf("systemone: score question %q requires 1..10 ordered criteria", id)
			}
		default:
			return nil, fmt.Errorf("systemone: question %q has unsupported type %q", id, question.Type)
		}
	}
	raw, err := httpRequester.Post(ctx, endpoint, headers, json.RawMessage(body))
	if err != nil {
		return nil, err
	}
	var response spec.SystemOneResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, fmt.Errorf("systemone: decode response: %w", err)
	}
	if response.Model == "" || len(response.Answers) != len(wire.Questions) {
		return nil, fmt.Errorf("systemone: response must contain a model and an answer for every question")
	}
	for id, question := range wire.Questions {
		answer, ok := response.Answers[id]
		if !ok || answer.Type != question.Type {
			return nil, fmt.Errorf("systemone: missing or mismatched answer for question %q", id)
		}
		switch answer.Type {
		case spec.SystemOneNoul:
			if answer.Noul == nil {
				return nil, fmt.Errorf("systemone: answer %q is missing noul", id)
			}
		case spec.SystemOneChoice, spec.SystemOneScore:
			if answer.Confidence == nil || len(answer.Probabilities) == 0 ||
				(answer.Type == spec.SystemOneScore && (answer.Score == nil || len(answer.Legend) == 0)) {
				return nil, fmt.Errorf("systemone: answer %q is missing decision fields", id)
			}
			if answer.Type == spec.SystemOneChoice {
				if _, ok := answer.Probabilities[answer.Choice]; !ok {
					return nil, fmt.Errorf("systemone: answer %q is missing its selected choice probability", id)
				}
			}
		}
	}
	response.RawResponse = raw
	return &response, nil
}
