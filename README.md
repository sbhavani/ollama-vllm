# Ollama with vLLM Integration

This integration allows Ollama to use vLLM as an external LLM engine alongside the default llama.cpp implementation. vLLM is designed for higher performance inference, especially on systems with NVIDIA GPUs.

## Setup

### Using Docker Compose

The included `docker-compose.yml` file sets up both Ollama and vLLM services with shared model storage:

```bash
# Start services with default llama.cpp backend
docker-compose up -d

# Or start with vLLM backend
OLLAMA_LLM_BACKEND=vllm docker-compose up -d
```

### Configuration

You can configure the integration using environment variables:

#### Ollama Service
- `OLLAMA_LLM_BACKEND`: Set to `vllm` to use vLLM or `llama` (default) to use llama.cpp
- `OLLAMA_VLLM_HOST`: Host and port of the vLLM service (default: `vllm:8000`)

#### vLLM Service
- `VLLM_MODEL_PATH`: Path to model files (default: `/data/models`)
- `VLLM_TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism (default: 1)
- `VLLM_GPU_MEMORY_UTILIZATION`: Fraction of GPU memory to use (default: 0.9)
- `VLLM_MAX_MODEL_LEN`: Maximum model sequence length (default: 8192)

## Usage

After starting the services, you can use Ollama as you normally would. The backend engine (llama.cpp or vLLM) is determined by the `OLLAMA_LLM_BACKEND` environment variable.

### Example:

```bash
# Pull a model
curl http://localhost:11434/api/pull -d '{"name": "llama3"}'

# Run inference
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "What is machine learning?"
}'
```

## Performance Considerations

- **vLLM Backend**: Better for larger models and batch inference, leverages CUDA more efficiently
- **llama.cpp Backend**: Better for CPU-only setups or lower memory requirements

## Switching Backends

To switch backends, stop the services and restart with the desired backend:

```bash
# Stop services
docker-compose down

# Restart with vLLM backend
OLLAMA_LLM_BACKEND=vllm docker-compose up -d

# Or restart with llama.cpp backend
docker-compose up -d
```

## Troubleshooting

1. **vLLM Not Available**: Check that the vLLM service is running with `docker-compose ps`
2. **Model Loading Issues**: Ensure models are in the correct format and location
3. **GPU Memory Errors**: Try reducing `VLLM_GPU_MEMORY_UTILIZATION` to a lower value

## Limitations

- vLLM requires NVIDIA GPUs with CUDA support
- Not all Ollama models may work with vLLM without conversion
- Embedding models may have different behavior between backends

## Technical Architecture

This integration connects Ollama with vLLM using a microservices architecture that maintains Ollama's existing LLM abstraction.

### Design Approach

1. **Interface Implementation**: vLLM is integrated by implementing Ollama's `LlamaServer` interface in `llm/vllm.go`. This allows vLLM to be used as a drop-in replacement for llama.cpp without changing the core Ollama code.

2. **Factory Pattern**: A factory function approach is used in the scheduler to dynamically select which backend (llama.cpp or vLLM) to use based on environment configuration. This makes the integration extendable for future backends.

3. **Microservices Communication**: The implementation uses HTTP requests to communicate with the vLLM service, translating between Ollama's internal API and vLLM's OpenAI-compatible API endpoints.

4. **Shared Storage**: Docker volumes are used to share model files between Ollama and vLLM services, ensuring both services have access to the same models.

### Key Technical Components

- **API Translation**: Converts between Ollama's completion requests and vLLM's OpenAI-compatible API
- **Streaming Support**: Implements server-sent events (SSE) processing to stream responses back to clients
- **Text Processing**: Uses Ollama's existing tokenizers with vLLM for consistent tokenization
- **Resource Estimation**: Adapts Ollama's memory estimation system to work with vLLM's requirements

### Performance Optimizations

- **Parallel Processing**: Configurable parallel inference for handling multiple requests
- **GPU Memory Utilization**: Adjustable GPU memory utilization to balance between performance and stability
- **Tensor Parallelism**: Support for multi-GPU inference via tensor parallelism configuration 