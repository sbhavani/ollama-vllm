package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/model"
)

// vLLMServer is an implementation of the LlamaServer interface for vLLM
type vLLMServer struct {
	baseURL      string
	options      api.Options
	numParallel  int
	modelPath    string
	textProcessor model.TextProcessor
	estimate     MemoryEstimate
	totalLayers  uint64
	gpus         discover.GpuInfoList
	loadDuration time.Duration
	loadProgress float32
	sem          *semaphore.Weighted
}

// NewVLLMServer creates a new vLLM server instance
func NewVLLMServer(gpus discover.GpuInfoList, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	vllmHost := envconfig.GetString("OLLAMA_VLLM_HOST", "localhost:8000")
	baseURL := fmt.Sprintf("http://%s", vllmHost)

	// Check if vLLM server is running
	resp, err := http.Get(baseURL + "/health")
	if err != nil {
		return nil, fmt.Errorf("vLLM server not available: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vLLM server not healthy. Status: %d", resp.StatusCode)
	}

	// Get text processor for the model
	processor, err := model.GetTextProcessor(f)
	if err != nil {
		return nil, fmt.Errorf("failed to get text processor: %v", err)
	}

	estimate := EstimateGPULayers(gpus, f, projectors, opts, numParallel)
	slog.Info("vLLM server", "baseURL", baseURL, "offload", estimate)

	server := &vLLMServer{
		baseURL:       baseURL,
		options:       opts,
		numParallel:   numParallel,
		modelPath:     modelPath,
		textProcessor: processor,
		estimate:      estimate,
		totalLayers:   f.GetLayerCount(),
		gpus:          gpus,
		loadProgress:  1.0, // vLLM server is already loaded
		sem:           semaphore.NewWeighted(int64(numParallel)),
	}

	return server, nil
}

func (s *vLLMServer) Ping(ctx context.Context) error {
	resp, err := http.Get(s.baseURL + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vLLM server not healthy. Status: %d", resp.StatusCode)
	}

	return nil
}

func (s *vLLMServer) WaitUntilRunning(ctx context.Context) error {
	for {
		err := s.Ping(ctx)
		if err == nil {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
			// Try again
		}
	}
}

// vLLM API request/response structures
type vLLMCompletionRequest struct {
	Prompt            string   `json:"prompt"`
	MaxTokens         int      `json:"max_tokens"`
	Temperature       float64  `json:"temperature"`
	TopP              float64  `json:"top_p"`
	TopK              int      `json:"top_k,omitempty"`
	PresencePenalty   float64  `json:"presence_penalty,omitempty"`
	FrequencyPenalty  float64  `json:"frequency_penalty,omitempty"`
	StopTokens        []string `json:"stop,omitempty"`
	Stream            bool     `json:"stream"`
}

type vLLMCompletionResponseChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		Delta        struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

type vLLMEmbeddingRequest struct {
	Inputs []string `json:"inputs"`
}

type vLLMEmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

func (s *vLLMServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return err
	}
	defer s.sem.Release(1)

	opts := s.options
	if req.Options != nil {
		opts = *req.Options
	}

	// Convert stop tokens into a slice of strings
	var stopTokens []string
	if opts.Stop != nil {
		stopTokens = opts.Stop
	}

	vllmReq := vLLMCompletionRequest{
		Prompt:           req.Prompt,
		MaxTokens:        opts.NumPredict,
		Temperature:      opts.Temperature,
		TopP:             opts.TopP,
		TopK:             opts.TopK,
		PresencePenalty:  opts.PresencePenalty,
		FrequencyPenalty: opts.FrequencyPenalty,
		StopTokens:       stopTokens,
		Stream:           true,
	}

	jsonData, err := json.Marshal(vllmReq)
	if err != nil {
		return err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/v1/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return fmt.Errorf("vLLM API error (status %d): %s", httpResp.StatusCode, string(body))
	}

	reader := bufio.NewReader(httpResp.Body)
	startTime := time.Now()
	evalCount := 0
	promptEvalDuration := time.Duration(0)
	done := false

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		// Each SSE message starts with "data: "
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}

		data := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			done = true
			break
		}

		var chunk vLLMCompletionResponseChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return err
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		content := chunk.Choices[0].Delta.Content
		evalCount++

		doneReason := DoneReasonNone
		if chunk.Choices[0].FinishReason != "" {
			done = true
			switch chunk.Choices[0].FinishReason {
			case "stop":
				doneReason = DoneReasonStop
			case "length":
				doneReason = DoneReasonLength
			}
		}

		evalDuration := time.Since(startTime)
		if promptEvalDuration == 0 && evalCount > 0 {
			promptEvalDuration = evalDuration
		}

		response := CompletionResponse{
			Content:            content,
			Done:               done,
			DoneReason:         doneReason,
			EvalCount:          evalCount,
			EvalDuration:       evalDuration,
			PromptEvalCount:    0, // vLLM doesn't provide this info
			PromptEvalDuration: promptEvalDuration,
		}

		fn(response)

		if done {
			break
		}
	}

	return nil
}

func (s *vLLMServer) Embedding(ctx context.Context, input string) ([]float32, error) {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	defer s.sem.Release(1)

	vllmReq := vLLMEmbeddingRequest{
		Inputs: []string{input},
	}

	jsonData, err := json.Marshal(vllmReq)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, fmt.Errorf("vLLM API error (status %d): %s", httpResp.StatusCode, string(body))
	}

	var response vLLMEmbeddingResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if len(response.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return response.Data[0].Embedding, nil
}

func (s *vLLMServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	// Use the model's text processor to tokenize the content
	if s.textProcessor != nil {
		return s.textProcessor.Encode(content)
	}
	return nil, fmt.Errorf("text processor not available")
}

func (s *vLLMServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	// Use the model's text processor to detokenize the tokens
	if s.textProcessor != nil {
		return s.textProcessor.Decode(tokens)
	}
	return "", fmt.Errorf("text processor not available")
}

func (s *vLLMServer) Close() error {
	// vLLM runs as a separate service, no need to close anything
	return nil
}

func (s *vLLMServer) EstimatedVRAM() uint64 {
	return s.estimate.VRAMSize
}

func (s *vLLMServer) EstimatedTotal() uint64 {
	return s.estimate.TotalSize
}

func (s *vLLMServer) EstimatedVRAMByGPU(gpuID string) uint64 {
	// For simplicity, distribute VRAM evenly across GPUs
	for _, gpu := range s.gpus {
		if gpu.ID == gpuID {
			return s.estimate.VRAMSize / uint64(len(s.gpus))
		}
	}
	return 0
} 