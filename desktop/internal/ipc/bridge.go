package ipc

import (
	"context"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/services"
)

// Bridge exposes Go service methods as Wails-bound functions callable from Svelte.
type Bridge struct {
	inferenceService *services.InferenceService
	modelService     *services.ModelService
}

// NewBridge constructs a Bridge with the required services.
func NewBridge(is *services.InferenceService, ms *services.ModelService) *Bridge {
	return &Bridge{
		inferenceService: is,
		modelService:     ms,
	}
}

// RunInference sends a prompt to the inference service and returns the response.
// Called from Svelte via: import { RunInference } from '../../wailsjs/go/ipc/Bridge'
func (b *Bridge) RunInference(prompt, modelID string) (*services.InferenceResponse, error) {
	req := &services.InferenceRequest{
		Prompt:      prompt,
		ModelID:     modelID,
		MaxTokens:   512,
		Temperature: 0.7,
		TopP:        0.9,
		Metadata:    map[string]interface{}{},
	}
	return b.inferenceService.Execute(context.Background(), req)
}

// ListModels returns all currently loaded models.
func (b *Bridge) ListModels() ([]services.ModelInfo, error) {
	return b.modelService.ListModels(context.Background())
}

// LoadModel loads the specified model and returns any error.
func (b *Bridge) LoadModel(modelID string) error {
	_, err := b.modelService.LoadModel(context.Background(), modelID)
	return err
}
