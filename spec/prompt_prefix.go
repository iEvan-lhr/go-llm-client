package spec

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// PromptPrefixConfig describes reusable prompt content. Examples and Context
// retain their supplied roles and order; reference data is never promoted to
// system instructions. Tools must already use the target provider's schema.
type PromptPrefixConfig struct {
	Version      string
	SystemPrompt string
	Examples     []Message
	Context      []Message
	Tools        []any
}

// PromptPrefix is an immutable snapshot, safe to reuse across requests. It
// builds input only: it does not enable server caching or set provider options.
type PromptPrefix struct {
	version     string
	messages    []Message
	tools       []json.RawMessage
	fingerprint string
}

// NewPromptPrefix freezes the fixed rules, examples, reference messages and
// tool definitions. It never sorts arrays, rewrites text or pads token counts.
func NewPromptPrefix(config PromptPrefixConfig) (*PromptPrefix, error) {
	prefix := &PromptPrefix{version: config.Version}
	if config.SystemPrompt != "" {
		prefix.messages = append(prefix.messages, NewSystemMessage(config.SystemPrompt))
	}
	prefix.messages = append(prefix.messages, CloneMessages(config.Examples)...)
	prefix.messages = append(prefix.messages, CloneMessages(config.Context)...)
	if config.Tools != nil {
		prefix.tools = make([]json.RawMessage, len(config.Tools))
		for i, tool := range config.Tools {
			frozen, err := SnapshotJSON(tool)
			if err != nil {
				return nil, fmt.Errorf("prompt prefix tool %d: %w", i, err)
			}
			prefix.tools[i] = frozen
		}
	}
	data, err := json.Marshal(struct {
		Version  string            `json:"version"`
		Messages []Message         `json:"messages"`
		Tools    []json.RawMessage `json:"tools"`
	}{prefix.version, prefix.messages, prefix.tools})
	if err != nil {
		return nil, fmt.Errorf("prompt prefix: %w", err)
	}
	// Canonicalize object keys, including raw tool schemas. UseNumber prevents
	// large numeric schema constants from losing precision.
	var canonical any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&canonical); err != nil {
		return nil, fmt.Errorf("prompt prefix fingerprint: %w", err)
	}
	data, err = json.Marshal(canonical)
	if err != nil {
		return nil, fmt.Errorf("prompt prefix fingerprint: %w", err)
	}
	sum := sha256.Sum256(data)
	prefix.fingerprint = hex.EncodeToString(sum[:])
	return prefix, nil
}

// BuildMessages returns fixed content followed by history and current input.
// History must exclude the fixed prefix; this method deliberately does not
// guess whether equal-looking messages are duplicates. Every message is copied.
func (p *PromptPrefix) BuildMessages(history []Message, current ...Message) []Message {
	var result []Message
	if p != nil {
		result = CloneMessages(p.messages)
	}
	result = append(result, CloneMessages(history)...)
	return append(result, CloneMessages(current)...)
}

// Tools returns a fresh snapshot of the tool array, preserving its order.
func (p *PromptPrefix) Tools() []any {
	if p == nil || p.tools == nil {
		return nil
	}
	result := make([]any, len(p.tools))
	for i, tool := range p.tools {
		result[i] = append(json.RawMessage(nil), tool...)
	}
	return result
}

// Parameters copies base parameters and applies the fixed tools when supplied.
// A nil Tools configuration leaves base["tools"] unchanged; an empty non-nil
// Tools configuration explicitly supplies an empty tool array.
func (p *PromptPrefix) Parameters(base map[string]any) (map[string]any, error) {
	result, err := SnapshotParameters(base)
	if err != nil {
		return nil, err
	}
	if p != nil && p.tools != nil {
		if result == nil {
			result = make(map[string]any)
		}
		result["tools"] = p.Tools()
	}
	return result, nil
}

// Fingerprint identifies this local template, including its version and tools.
// It excludes model, endpoint and dynamic input; it is neither a provider cache
// key nor evidence that the provider will match or cache the resulting tokens.
func (p *PromptPrefix) Fingerprint() string {
	if p == nil {
		return ""
	}
	return p.fingerprint
}

func (p *PromptPrefix) Version() string {
	if p == nil {
		return ""
	}
	return p.version
}
