package spec

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// SnapshotJSON freezes a JSON request value without retaining caller-owned
// maps, slices or pointers. Raw JSON preserves custom marshalers and number
// precision. The returned bytes belong to the caller.
func SnapshotJSON(value any) (json.RawMessage, error) {
	if value == nil {
		return nil, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("snapshot JSON: %w", err)
	}
	return json.RawMessage(data), nil
}

// SnapshotParameters copies a parameter map as JSON values. Object and array
// values become map[string]any and []any; numbers use json.Number so schema
// constants do not lose precision. Custom marshalers run when snapshotting.
func SnapshotParameters(parameters map[string]any) (map[string]any, error) {
	if parameters == nil {
		return nil, nil
	}
	data, err := SnapshotJSON(parameters)
	if err != nil {
		return nil, err
	}
	var result map[string]any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&result); err != nil {
		return nil, fmt.Errorf("snapshot parameters: %w", err)
	}
	return result, nil
}

// SnapshotInput preserves the message types that providers normalize for their
// wire protocols. Other input values are frozen as JSON.
func SnapshotInput(input any) (any, error) {
	switch value := input.(type) {
	case nil, string:
		return value, nil
	case Message:
		return value.Clone(), nil
	case *Message:
		if value == nil {
			return nil, nil
		}
		cloned := value.Clone()
		return &cloned, nil
	case []Message:
		return CloneMessages(value), nil
	case ContentPart:
		return (Message{Parts: []ContentPart{value}}).Clone().Parts[0], nil
	case []ContentPart:
		return (Message{Parts: value}).Clone().Parts, nil
	default:
		frozen, err := SnapshotJSON(input)
		if err != nil {
			return nil, err
		}
		return frozen, nil
	}
}

// CloneMessages copies messages, including all nested mutable content. It
// preserves nil slices, message order and text exactly.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	result := make([]Message, len(messages))
	for i := range messages {
		result[i] = messages[i].Clone()
	}
	return result
}

// Clone returns an independently mutable copy of a message.
func (m Message) Clone() Message {
	if m.Parts != nil {
		parts := make([]ContentPart, len(m.Parts))
		copy(parts, m.Parts)
		for i := range parts {
			if parts[i].ImageURL != nil {
				value := *parts[i].ImageURL
				parts[i].ImageURL = &value
			}
			if parts[i].VideoURL != nil {
				value := *parts[i].VideoURL
				parts[i].VideoURL = &value
			}
			if parts[i].InputAudio != nil {
				value := *parts[i].InputAudio
				parts[i].InputAudio = &value
			}
		}
		m.Parts = parts
	}
	if m.ToolCalls != nil {
		calls := make([]ToolCall, len(m.ToolCalls))
		copy(calls, m.ToolCalls)
		for i := range calls {
			if calls[i].Index != nil {
				value := *calls[i].Index
				calls[i].Index = &value
			}
		}
		m.ToolCalls = calls
	}
	if m.Annotations != nil {
		annotations := make([]json.RawMessage, len(m.Annotations))
		for i, annotation := range m.Annotations {
			if annotation != nil {
				annotations[i] = append(json.RawMessage{}, annotation...)
			}
		}
		m.Annotations = annotations
	}
	return m
}
