package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"strconv"
	"strings"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// NewImagesClient creates an OpenAI client with the Images API available
// without a type assertion.
func NewImagesClient(opts ...spec.ClientOption) (spec.ImagesClient, error) {
	client, err := NewClient(opts...)
	if err != nil {
		return nil, err
	}
	imagesClient, ok := client.(spec.ImagesClient)
	if !ok {
		return nil, fmt.Errorf("openai provider: Images API is unavailable")
	}
	return imagesClient, nil
}

// CreateImage calls POST /v1/images/generations with a JSON request body.
func (c *clientImpl) CreateImage(ctx context.Context, request spec.ImageGenerationRequest) (*spec.ImageResponse, error) {
	if strings.TrimSpace(request.Model) == "" {
		return nil, fmt.Errorf("openai images: model is required")
	}
	if strings.TrimSpace(request.Prompt) == "" {
		return nil, fmt.Errorf("openai images: prompt is required")
	}
	endpoint, err := imagesEndpointURL(c.config.APIURL, "generations")
	if err != nil {
		return nil, err
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	return decodeImageResponse(raw)
}

// EditImage calls POST /v1/images/edits with a multipart/form-data request.
func (c *clientImpl) EditImage(ctx context.Context, request spec.ImageEditRequest) (*spec.ImageResponse, error) {
	if strings.TrimSpace(request.Model) == "" {
		return nil, fmt.Errorf("openai images: model is required")
	}
	if strings.TrimSpace(request.Prompt) == "" {
		return nil, fmt.Errorf("openai images: prompt is required")
	}
	if len(request.Image.Data) == 0 {
		return nil, fmt.Errorf("openai images: image data is required")
	}

	endpoint, err := imagesEndpointURL(c.config.APIURL, "edits")
	if err != nil {
		return nil, err
	}
	body, contentType, err := imageEditMultipart(request)
	if err != nil {
		return nil, err
	}
	headers := c.headers()
	headers.Set("Content-Type", contentType)
	raw, err := c.requester.DoBody(ctx, http.MethodPost, endpoint, headers, body)
	if err != nil {
		return nil, err
	}
	return decodeImageResponse(raw)
}

func imageEditMultipart(request spec.ImageEditRequest) (*bytes.Buffer, string, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	fields := make(map[string]string, len(request.ExtraFields)+11)
	for key, value := range request.ExtraFields {
		fields[key] = value
	}
	fields["model"] = request.Model
	fields["prompt"] = request.Prompt
	setImageFormField(fields, "background", request.Background)
	setImageFormField(fields, "input_fidelity", request.InputFidelity)
	if request.N > 0 {
		fields["n"] = strconv.Itoa(request.N)
	}
	if request.OutputCompression != nil {
		fields["output_compression"] = strconv.Itoa(*request.OutputCompression)
	}
	setImageFormField(fields, "output_format", request.OutputFormat)
	setImageFormField(fields, "quality", request.Quality)
	setImageFormField(fields, "response_format", request.ResponseFormat)
	setImageFormField(fields, "size", request.Size)
	setImageFormField(fields, "user", request.User)
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			return nil, "", fmt.Errorf("openai images: write multipart field %q: %w", key, err)
		}
	}
	if err := writeImagePart(writer, "image", request.Image); err != nil {
		return nil, "", err
	}
	if request.Mask != nil {
		if err := writeImagePart(writer, "mask", *request.Mask); err != nil {
			return nil, "", err
		}
	}
	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("openai images: close multipart body: %w", err)
	}
	return body, writer.FormDataContentType(), nil
}

func writeImagePart(writer *multipart.Writer, field string, file spec.ImageFile) error {
	if len(file.Data) == 0 {
		return fmt.Errorf("openai images: %s data is required", field)
	}
	filename := file.Filename
	if filename == "" {
		filename = field + ".png"
	}
	contentType := file.ContentType
	if contentType == "" {
		contentType = http.DetectContentType(file.Data)
	}
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", mime.FormatMediaType("form-data", map[string]string{
		"name":     field,
		"filename": filename,
	}))
	header.Set("Content-Type", contentType)
	part, err := writer.CreatePart(header)
	if err != nil {
		return fmt.Errorf("openai images: create multipart %s: %w", field, err)
	}
	if _, err := part.Write(file.Data); err != nil {
		return fmt.Errorf("openai images: write multipart %s: %w", field, err)
	}
	return nil
}

func setImageFormField(fields map[string]string, key, value string) {
	if value != "" {
		fields[key] = value
	}
}

func decodeImageResponse(raw []byte) (*spec.ImageResponse, error) {
	var response spec.ImageResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, fmt.Errorf("openai images: decode response: %w", err)
	}
	var envelope struct {
		Error *spec.APIError `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error != nil {
		code := envelope.Error.Code
		if code == "" {
			code = envelope.Error.Type
		}
		if code != "" {
			return nil, fmt.Errorf("openai images: %s: %s", code, envelope.Error.Message)
		}
		return nil, fmt.Errorf("openai images: %s", envelope.Error.Message)
	}
	response.RawResponse = append(response.RawResponse[:0], raw...)
	return &response, nil
}

func imagesEndpointURL(configured, operation string) (string, error) {
	if operation != "generations" && operation != "edits" {
		return "", fmt.Errorf("openai images: unsupported operation %q", operation)
	}
	parsed, err := url.Parse(configured)
	if err != nil {
		return "", fmt.Errorf("openai images: invalid API URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("openai images: API URL must be absolute")
	}
	path := strings.TrimRight(parsed.Path, "/")
	for _, suffix := range []string{
		"/chat/completions",
		"/images/generations",
		"/images/edits",
		"/responses",
	} {
		if strings.HasSuffix(path, suffix) {
			path = strings.TrimSuffix(path, suffix)
			break
		}
	}
	if strings.HasSuffix(path, "/images") {
		path += "/" + operation
	} else {
		path += "/images/" + operation
	}
	parsed.Path = path
	parsed.RawPath = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

var _ spec.ImagesClient = (*clientImpl)(nil)
