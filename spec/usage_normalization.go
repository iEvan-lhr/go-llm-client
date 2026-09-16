package spec

import "encoding/json"

// CacheUsage distinguishes a reported zero from an absent cache counter.
type CacheUsage struct {
	ReadTokens    int  `json:"read_tokens"`
	WriteTokens   int  `json:"write_tokens"`
	ReadReported  bool `json:"read_reported"`
	WriteReported bool `json:"write_reported"`
}

func (r *Response) CacheUsage() CacheUsage {
	if r == nil {
		return CacheUsage{}
	}
	var value CacheUsage
	if r.Usage != nil {
		for _, detail := range []*TokenDetails{r.Usage.PromptTokensDetails, r.Usage.InputTokensDetails} {
			if detail == nil {
				continue
			}
			if detail.CachedTokensReported || detail.CachedTokens != 0 {
				value.ReadTokens, value.ReadReported = detail.CachedTokens, true
			}
			if detail.CacheWriteTokensReported || detail.CacheWriteTokens != 0 {
				value.WriteTokens, value.WriteReported = detail.CacheWriteTokens, true
			}
		}
	}
	return value
}

func (t *TokenDetails) UnmarshalJSON(data []byte) error {
	type plain TokenDetails
	var value plain
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	*t = TokenDetails(value)
	if raw, ok := fields["cached_tokens"]; ok && string(raw) != "null" {
		t.CachedTokensReported = true
	}
	if raw, ok := fields["cache_write_tokens"]; ok && string(raw) != "null" {
		t.CacheWriteTokensReported = true
	}
	return nil
}

// NormalizeUsage handles compatible providers' additional cache counters.
// InputTokens/PromptTokens remain total input, including cached input.
func NormalizeUsage(data []byte) *Usage {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	var usage Usage
	if json.Unmarshal(data, &usage) != nil {
		return nil
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(data, &fields) != nil || len(fields) == 0 {
		return nil
	}
	read := func(names ...string) (int, bool) {
		for _, name := range names {
			if raw, ok := fields[name]; ok && string(raw) != "null" {
				var value int
				if json.Unmarshal(raw, &value) == nil && value >= 0 {
					return value, true
				}
			}
		}
		return 0, false
	}
	if n, ok := read("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens"); ok {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &TokenDetails{}
		}
		usage.PromptTokensDetails.CachedTokens = n
		usage.PromptTokensDetails.CachedTokensReported = true
	}
	if n, ok := read("cache_creation_input_tokens", "cache_write_tokens"); ok {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &TokenDetails{}
		}
		usage.PromptTokensDetails.CacheWriteTokens = n
		usage.PromptTokensDetails.CacheWriteTokensReported = true
	}
	if usage.InputTokens == 0 && usage.PromptTokens == 0 {
		hit, hasHit := read("prompt_cache_hit_tokens")
		miss, hasMiss := read("prompt_cache_miss_tokens")
		if hasHit && hasMiss {
			usage.PromptTokens = hit + miss
		}
	}
	return &usage
}

// ApplyResponseMetadata also accepts a streaming terminal envelope. Later
// chunks without usage never erase a previously reported usage snapshot.
func ApplyResponseMetadata(response *Response, data []byte) {
	if response == nil {
		return
	}
	var wire struct {
		ID       string          `json:"id"`
		Model    string          `json:"model"`
		Usage    json.RawMessage `json:"usage"`
		Response json.RawMessage `json:"response"`
		Choices  []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if json.Unmarshal(data, &wire) != nil {
		return
	}
	response.Protocol = ProtocolChatCompletions
	if wire.ID != "" {
		response.ID = wire.ID
	}
	if wire.Model != "" {
		response.Model = wire.Model
	}
	if usage := NormalizeUsage(wire.Usage); usage != nil {
		response.Usage = usage
	}
	for _, choice := range wire.Choices {
		if choice.FinishReason != "" {
			response.Status = choice.FinishReason
			break
		}
	}
	if len(wire.Response) > 0 {
		ApplyResponseMetadata(response, wire.Response)
		response.Protocol = ProtocolResponses
	}
}
