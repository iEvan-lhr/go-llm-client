package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
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
	imagesClient, ok := client.(spec.ImagesAPIClient)
	if !ok {
		return nil, fmt.Errorf("openai provider: Images API is unavailable")
	}
	return imagesClient, nil
}

// NewImagesAPIClient also exposes streaming and image variations.
func NewImagesAPIClient(opts ...spec.ClientOption) (spec.ImagesAPIClient, error) {
	return NewAPIClient(opts...)
}

// CreateImage calls POST /v1/images/generations with a JSON request body.
func (c *clientImpl) CreateImage(ctx context.Context, request spec.ImageGenerationRequest) (*spec.ImageResponse, error) {
	stream, _ := request.ExtraFields["stream"].(bool)
	return c.createImage(ctx, request, stream, nil)
}

// CreateImageStream receives partial images and returns completed images only.
func (c *clientImpl) CreateImageStream(ctx context.Context, request spec.ImageGenerationRequest, callback spec.ImageEventCallback) (*spec.ImageResponse, error) {
	return c.createImage(ctx, request, true, callback)
}

func (c *clientImpl) createImage(ctx context.Context, request spec.ImageGenerationRequest, stream bool, callback spec.ImageEventCallback) (*spec.ImageResponse, error) {
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
	if stream {
		body, err := requestMap(request)
		if err != nil {
			return nil, err
		}
		body["stream"] = true
		resp, err := c.requester.PostStream(ctx, endpoint, c.headers(), body)
		if err != nil {
			return nil, err
		}
		return consumeImageStream(ctx, resp, request.N, callback)
	}
	raw, err := c.requester.Post(ctx, endpoint, c.headers(), request)
	if err != nil {
		return nil, err
	}
	return decodeImageResponse(raw)
}

// EditImage calls POST /v1/images/edits with a multipart/form-data request.
func (c *clientImpl) EditImage(ctx context.Context, request spec.ImageEditRequest) (*spec.ImageResponse, error) {
	return c.editImage(ctx, request, request.ExtraFields["stream"] == "true", nil)
}

func (c *clientImpl) EditImageStream(ctx context.Context, request spec.ImageEditRequest, callback spec.ImageEventCallback) (*spec.ImageResponse, error) {
	return c.editImage(ctx, request, true, callback)
}

func (c *clientImpl) editImage(ctx context.Context, request spec.ImageEditRequest, stream bool, callback spec.ImageEventCallback) (*spec.ImageResponse, error) {
	if strings.TrimSpace(request.Model) == "" {
		return nil, fmt.Errorf("openai images: model is required")
	}
	if strings.TrimSpace(request.Prompt) == "" {
		return nil, fmt.Errorf("openai images: prompt is required")
	}
	if len(request.Image.Data) == 0 && len(request.Images) == 0 {
		return nil, fmt.Errorf("openai images: image data is required")
	}
	if len(request.Image.Data) > 0 && len(request.Images) > 0 {
		return nil, fmt.Errorf("openai images: use Image or Images, not both")
	}
	if stream {
		fields := make(map[string]string, len(request.ExtraFields)+1)
		for key, value := range request.ExtraFields {
			fields[key] = value
		}
		fields["stream"] = "true"
		request.ExtraFields = fields
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
	if stream {
		resp, err := c.requester.DoBodyStream(ctx, http.MethodPost, endpoint, headers, body)
		if err != nil {
			return nil, err
		}
		return consumeImageStream(ctx, resp, request.N, callback)
	}
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
	if request.PartialImages > 0 {
		fields["partial_images"] = strconv.Itoa(request.PartialImages)
	}
	setImageFormField(fields, "quality", request.Quality)
	setImageFormField(fields, "response_format", request.ResponseFormat)
	setImageFormField(fields, "size", request.Size)
	setImageFormField(fields, "user", request.User)
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			return nil, "", fmt.Errorf("openai images: write multipart field %q: %w", key, err)
		}
	}
	if len(request.Images) > 0 {
		for _, file := range request.Images {
			if err := writeImagePart(writer, "image[]", file); err != nil {
				return nil, "", err
			}
		}
	} else {
		if err := writeImagePart(writer, "image", request.Image); err != nil {
			return nil, "", err
		}
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
		return fmt.Errorf("openai multipart: %s data is required", field)
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
		return fmt.Errorf("openai multipart: create %s: %w", field, err)
	}
	if _, err := part.Write(file.Data); err != nil {
		return fmt.Errorf("openai multipart: write %s: %w", field, err)
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
	if err := decodeAPIResponse(raw, &response); err != nil {
		return nil, err
	}
	response.RawResponse = append(response.RawResponse[:0], raw...)
	return &response, nil
}

func imagesEndpointURL(configured, operation string) (string, error) {
	if operation != "generations" && operation != "edits" && operation != "variations" {
		return "", fmt.Errorf("openai images: unsupported operation %q", operation)
	}
	return apiEndpointURL(configured, "images/"+operation)
}

func (c *clientImpl) CreateImageVariation(ctx context.Context, request spec.ImageVariationRequest) (*spec.ImageResponse, error) {
	endpoint, err := imagesEndpointURL(c.config.APIURL, "variations")
	if err != nil {
		return nil, err
	}
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	fields := make(map[string]string, len(request.ExtraFields)+5)
	for key, value := range request.ExtraFields {
		fields[key] = value
	}
	setImageFormField(fields, "model", request.Model)
	setImageFormField(fields, "size", request.Size)
	setImageFormField(fields, "response_format", request.ResponseFormat)
	setImageFormField(fields, "user", request.User)
	if request.N != 0 {
		fields["n"] = strconv.Itoa(request.N)
	}
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			return nil, err
		}
	}
	if err := writeImagePart(writer, "image", request.Image); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	headers := c.headers()
	headers.Set("Content-Type", writer.FormDataContentType())
	raw, err := c.requester.DoBody(ctx, http.MethodPost, endpoint, headers, body)
	if err != nil {
		return nil, err
	}
	return decodeImageResponse(raw)
}

func consumeImageStream(ctx context.Context, response *http.Response, expectedImages int, callback spec.ImageEventCallback) (*spec.ImageResponse, error) {
	defer response.Body.Close()
	if expectedImages <= 0 {
		expectedImages = 1
	}
	result := &spec.ImageResponse{}
	err := scanSSE(response, func(data string) error {
		if data == "[DONE]" {
			return errSSEDone
		}
		var event spec.ImageStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return fmt.Errorf("openai images: decode stream event: %w", err)
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
		case "image_generation.completed", "image_edit.completed":
			if event.Base64JSON == "" {
				return fmt.Errorf("openai images: completed event has no image")
			}
			result.Created = event.CreatedAt
			result.Background, result.OutputFormat = event.Background, event.OutputFormat
			result.Quality, result.Size = event.Quality, event.Size
			result.Usage = event.Usage
			result.Data = append(result.Data, spec.ImageData{Base64JSON: event.Base64JSON})
			if len(result.Data) >= expectedImages {
				return errSSEDone
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(result.Data) < expectedImages {
		return nil, fmt.Errorf("openai images: stream ended after %d of %d images: %w", len(result.Data), expectedImages, io.ErrUnexpectedEOF)
	}
	result.RawResponse, _ = json.Marshal(result)
	return result, nil
}

var _ spec.ImagesAPIClient = (*clientImpl)(nil)
