package client

import (
	"context"
	"fmt"
	"io"

	"github.com/iEvan-lhr/go-llm-client/spec"
)

// StartRealtimeTranscription opens a bidirectional speech recognition session.
// An empty request.Model uses the model configured on Client.
func (c *Client) StartRealtimeTranscription(ctx context.Context, request spec.RealtimeTranscriptionRequest) (spec.RealtimeTranscriptionSession, error) {
	api, ok := c.client.(spec.RealtimeTranscriptionClient)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support realtime transcription", c.config.Provider)
	}
	if request.Model == "" {
		request.Model = c.config.Model
	}
	return api.StartRealtimeTranscription(ctx, request)
}

// StreamRealtimeTranscription consumes a live audio stream until EOF, sends
// it to the provider, dispatches transcription callbacks, finishes the task,
// waits for the terminal server event, and closes the session. A microphone,
// FFmpeg stdout pipe, network stream, or any other io.Reader can be supplied.
// The reader should return EOF when capture ends and should be closed by the
// caller when ctx is cancelled if its Read method can otherwise block forever.
//
// When request.Manual is true, the complete reader is treated as one utterance
// and committed automatically before the session is finished. Use the lower-
// level StartRealtimeTranscription API to commit multiple utterances.
func (c *Client) StreamRealtimeTranscription(
	ctx context.Context,
	audio io.Reader,
	request spec.RealtimeTranscriptionRequest,
	options spec.RealtimeTranscriptionStreamOptions,
) (returnErr error) {
	if audio == nil {
		return fmt.Errorf("realtime transcription: audio reader is required")
	}
	chunkSize := options.ChunkSize
	if chunkSize == 0 {
		chunkSize = 3200
	}
	if chunkSize < 1 {
		return fmt.Errorf("realtime transcription: chunk size must be positive")
	}

	session, err := c.StartRealtimeTranscription(ctx, request)
	if err != nil {
		return err
	}
	receiveCtx, cancelReceive := context.WithCancel(ctx)
	receiverResult := make(chan error, 1)
	receiverExited := make(chan struct{})
	go func() {
		defer close(receiverExited)
		for {
			event, receiveErr := session.Receive(receiveCtx)
			if receiveErr != nil {
				if receiveCtx.Err() != nil {
					receiverResult <- receiveCtx.Err()
				} else {
					receiverResult <- receiveErr
				}
				return
			}
			if event.Error != nil {
				receiverResult <- event.Error
				return
			}
			if options.OnEvent != nil {
				if callbackErr := options.OnEvent(receiveCtx, event); callbackErr != nil {
					receiverResult <- fmt.Errorf("realtime transcription event callback: %w", callbackErr)
					return
				}
			}
			if options.OnText != nil && event.Transcript != "" {
				if callbackErr := options.OnText(receiveCtx, event.Transcript, event.Final); callbackErr != nil {
					receiverResult <- fmt.Errorf("realtime transcription text callback: %w", callbackErr)
					return
				}
			}
			if event.Terminal {
				receiverResult <- nil
				return
			}
		}
	}()
	defer func() {
		cancelReceive()
		closeErr := session.Close()
		<-receiverExited
		if returnErr == nil && closeErr != nil {
			returnErr = fmt.Errorf("realtime transcription: close session: %w", closeErr)
		}
	}()

	buffer := make([]byte, chunkSize)
	for {
		n, readErr := io.ReadFull(audio, buffer)
		if n > 0 {
			if sendErr := session.SendAudio(ctx, buffer[:n]); sendErr != nil {
				return fmt.Errorf("realtime transcription: send audio: %w", sendErr)
			}
		}
		select {
		case receiveErr := <-receiverResult:
			return receiveErr
		default:
		}
		switch readErr {
		case nil:
			continue
		case io.EOF, io.ErrUnexpectedEOF:
			goto audioComplete
		default:
			return fmt.Errorf("realtime transcription: read audio: %w", readErr)
		}
	}

audioComplete:
	if request.Manual {
		if err := session.Commit(ctx); err != nil {
			return fmt.Errorf("realtime transcription: commit audio: %w", err)
		}
	}
	if err := session.Finish(ctx); err != nil {
		return fmt.Errorf("realtime transcription: finish session: %w", err)
	}
	select {
	case receiveErr := <-receiverResult:
		return receiveErr
	case <-ctx.Done():
		return ctx.Err()
	}
}
