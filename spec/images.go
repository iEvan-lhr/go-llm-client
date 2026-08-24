package spec

import (
	"os"
	"path/filepath"
)

// ImageGenerationRequest is the JSON body for the Images generations API.
// ExtraFields allows compatible providers to accept fields that are not yet
// represented explicitly; typed fields take precedence when both are set.
type ImageGenerationRequest struct {
	Model             string         `json:"model"`
	Prompt            string         `json:"prompt"`
	Background        string         `json:"background,omitempty"`
	Moderation        string         `json:"moderation,omitempty"`
	N                 int            `json:"n,omitempty"`
	OutputCompression *int           `json:"output_compression,omitempty"`
	OutputFormat      string         `json:"output_format,omitempty"`
	PartialImages     int            `json:"partial_images,omitempty"`
	Quality           string         `json:"quality,omitempty"`
	ResponseFormat    string         `json:"response_format,omitempty"`
	Size              string         `json:"size,omitempty"`
	Style             string         `json:"style,omitempty"`
	User              string         `json:"user,omitempty"`
	ExtraFields       map[string]any `json:"-"`
}

func (r ImageGenerationRequest) MarshalJSON() ([]byte, error) {
	type alias ImageGenerationRequest
	return marshalWithExtra(alias(r), r.ExtraFields)
}

// ImageEditRequest contains the fields sent to the multipart Images edits API.
type ImageEditRequest struct {
	Model             string
	Prompt            string
	Image             ImageFile
	Mask              *ImageFile
	Background        string
	InputFidelity     string
	N                 int
	OutputCompression *int
	OutputFormat      string
	Quality           string
	ResponseFormat    string
	Size              string
	User              string
	ExtraFields       map[string]string
}

// ImageFile is an in-memory file uploaded by an Images API request.
type ImageFile struct {
	Filename    string
	ContentType string
	Data        []byte
}

// NewImageFile reads an image from disk for use in an Images edit request.
func NewImageFile(path, contentType string) (ImageFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ImageFile{}, err
	}
	return NewImageFileBytes(filepath.Base(path), contentType, data), nil
}

// NewImageFileBytes creates an upload file from bytes already in memory.
func NewImageFileBytes(filename, contentType string, data []byte) ImageFile {
	return ImageFile{
		Filename:    filename,
		ContentType: contentType,
		Data:        append([]byte(nil), data...),
	}
}

// ImageResponse is returned by both Images generations and edits.
type ImageResponse struct {
	Created      int64       `json:"created,omitempty"`
	Data         []ImageData `json:"data"`
	Background   string      `json:"background,omitempty"`
	OutputFormat string      `json:"output_format,omitempty"`
	Quality      string      `json:"quality,omitempty"`
	Size         string      `json:"size,omitempty"`
	Usage        *ImageUsage `json:"usage,omitempty"`
	RawResponse  []byte      `json:"-"`
}

// ImageData is one generated or edited image result.
type ImageData struct {
	Base64JSON    string `json:"b64_json,omitempty"`
	URL           string `json:"url,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// ImageUsage reports token usage for GPT Image models.
type ImageUsage struct {
	InputTokens         int                `json:"input_tokens,omitempty"`
	OutputTokens        int                `json:"output_tokens,omitempty"`
	TotalTokens         int                `json:"total_tokens,omitempty"`
	InputTokensDetails  *ImageTokenDetails `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *ImageTokenDetails `json:"output_tokens_details,omitempty"`
}

// ImageTokenDetails splits image-model usage into text and image tokens.
type ImageTokenDetails struct {
	ImageTokens int `json:"image_tokens,omitempty"`
	TextTokens  int `json:"text_tokens,omitempty"`
}

// Base64Images returns all non-empty base64 image payloads in response order.
func (r *ImageResponse) Base64Images() []string {
	if r == nil {
		return nil
	}
	images := make([]string, 0, len(r.Data))
	for _, item := range r.Data {
		if item.Base64JSON != "" {
			images = append(images, item.Base64JSON)
		}
	}
	return images
}
