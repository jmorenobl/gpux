# GPUX API Reference

## Python API

### GPUXRuntime

The main runtime class for running ML inference.

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime(
    model_path: str | Path,
    provider: str | None = None,
    gpu_memory: str | None = None,
    batch_size: int = 1,
    timeout: int = 30
)
```

#### Parameters

- **model_path**: Path to the ONNX model file
- **provider**: Execution provider ("auto", "cuda", "coreml", "rocm", etc.)
- **gpu_memory**: GPU memory allocation (e.g., "2GB", "512MB")
- **batch_size**: Batch size for inference
- **timeout**: Timeout for inference operations

#### Methods

##### `infer(input_data: dict[str, Any]) -> dict[str, Any]`

Run inference on input data.

```python
results = runtime.infer({
    "input1": [1, 2, 3, 4],
    "input2": [0.1, 0.2, 0.3, 0.4]
})
```

##### `benchmark(input_data: dict[str, Any], num_runs: int = 100, warmup_runs: int = 10) -> dict[str, float]`

Run performance benchmark.

```python
metrics = runtime.benchmark(input_data, num_runs=1000, warmup_runs=10)
# Returns: {
#     "avg_inference_time": 1.23,
#     "min_inference_time": 1.12,
#     "max_inference_time": 1.45,
#     "fps": 813.0
# }
```

##### `get_model_info() -> ModelInfo`

Get model information and metadata.

```python
info = runtime.get_model_info()
print(f"Inputs: {info.inputs}")
print(f"Outputs: {info.outputs}")
print(f"Size: {info.size_mb} MB")
```

##### `get_provider_info() -> dict[str, Any]`

Get current execution provider information.

```python
provider_info = runtime.get_provider_info()
# Returns: {
#     "provider": "CUDAExecutionProvider",
#     "platform": "NVIDIA",
#     "available": True,
#     "description": "CUDA provider"
# }
```

##### `get_available_providers() -> list[str]`

Get list of available execution providers.

```python
providers = runtime.get_available_providers()
# Returns: ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

##### `cleanup() -> None`

Clean up resources and close sessions.

```python
runtime.cleanup()
```

### ModelInfo

Model information and metadata.

```python
from gpux.core.models import ModelInfo

info = ModelInfo(
    name: str,
    version: str,
    format: str,
    size_bytes: int,
    path: Path,
    inputs: list[InputSpec],
    outputs: list[OutputSpec],
    metadata: dict[str, Any]
)
```

#### Properties

- **name**: Model name
- **version**: Model version
- **format**: Model format (e.g., "onnx")
- **size_bytes**: Model size in bytes
- **size_mb**: Model size in MB
- **path**: Path to model file
- **inputs**: List of input specifications
- **outputs**: List of output specifications
- **metadata**: Additional model metadata

#### Methods

##### `to_dict() -> dict[str, Any]`

Convert model info to dictionary.

##### `save(path: Path) -> None`

Save model info to JSON file.

### InputSpec / OutputSpec

Input and output specifications.

```python
from gpux.core.models import InputSpec, OutputSpec

input_spec = InputSpec(
    name: str,
    type: str,
    shape: list[int],
    required: bool = True,
    description: str = ""
)

output_spec = OutputSpec(
    name: str,
    type: str,
    shape: list[int],
    labels: list[str] | None = None,
    description: str = ""
)
```

### ProviderManager

Manage execution providers.

```python
from gpux.core.providers import ProviderManager, ExecutionProvider

manager = ProviderManager()
```

#### Methods

##### `get_best_provider(preferred: str | None = None) -> ExecutionProvider`

Get the best available execution provider.

```python
provider = manager.get_best_provider()  # Auto-select
provider = manager.get_best_provider("cuda")  # Specific provider
```

##### `get_available_providers() -> list[str]`

Get list of available providers.

##### `get_provider_info(provider: ExecutionProvider) -> dict[str, Any]`

Get information about a specific provider.

### GPUXConfigParser

Parse and validate GPUX configuration files.

```python
from gpux.config.parser import GPUXConfigParser

parser = GPUXConfigParser()
config = parser.parse_file("gpux.yml")
```

#### Methods

##### `parse_file(path: str | Path) -> GPUXConfig`

Parse configuration file.

##### `validate_model_path(project_path: Path) -> bool`

Validate that model file exists.

##### `get_model_path(project_path: Path) -> Path | None`

Get resolved model file path.

##### `to_dict() -> dict[str, Any]`

Convert configuration to dictionary.

##### `save(path: str | Path) -> None`

Save configuration to file.

## CLI API

### Commands

#### `gpux build [PATH]`

Build and optimize models for GPU inference.

```bash
gpux build .                           # Build from current directory
gpux build ./my-model --provider cuda  # Build with specific provider
gpux build . --no-optimize             # Build without optimization
gpux build . --verbose                 # Verbose output
```

**Options:**
- `--config, -c`: Configuration file name (default: gpux.yml)
- `--provider, -p`: Preferred execution provider
- `--no-optimize`: Disable model optimization
- `--verbose`: Enable verbose output

#### `gpux run MODEL_NAME`

Run inference on a model.

```bash
gpux run my-model --input '{"data": [1,2,3]}'  # JSON input
gpux run my-model --file input.json            # File input
gpux run my-model --benchmark --runs 1000      # Benchmark
```

**Options:**
- `--input, -i`: Input data (JSON string or @file)
- `--file, -f`: Input file path
- `--output, -o`: Output file path
- `--config, -c`: Configuration file name
- `--provider, -p`: Preferred execution provider
- `--benchmark`: Run benchmark instead of single inference
- `--runs`: Number of benchmark runs (default: 100)
- `--warmup`: Number of warmup runs (default: 10)
- `--verbose`: Enable verbose output

#### `gpux serve MODEL_NAME`

Start HTTP server for model serving.

```bash
gpux serve my-model                     # Start server
gpux serve my-model --port 9000        # Custom port
gpux serve my-model --workers 4        # Multiple workers
gpux serve my-model --host 127.0.0.1   # Custom host
```

**Options:**
- `--port, -p`: Port to serve on (default: 8080)
- `--host, -h`: Host to serve on (default: 0.0.0.0)
- `--config, -c`: Configuration file name
- `--provider`: Preferred execution provider
- `--workers`: Number of worker processes (default: 1)
- `--verbose`: Enable verbose output

#### `gpux inspect [MODEL_NAME]`

Inspect models and runtime information.

```bash
gpux inspect my-model              # Inspect model by name
gpux inspect --model model.onnx    # Inspect model file directly
gpux inspect --json                # JSON output
gpux inspect                       # Show runtime information
```

**Options:**
- `--config, -c`: Configuration file name
- `--model, -m`: Direct path to model file
- `--json`: Output in JSON format
- `--verbose`: Enable verbose output

## HTTP API

When serving a model with `gpux serve`, GPUX provides a REST API.

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "my-model"
}
```

#### `GET /info`

Get model information.

**Response:**
```json
{
  "name": "my-model",
  "version": "1.0.0",
  "format": "onnx",
  "size_mb": 25.6,
  "inputs": [
    {
      "name": "input1",
      "type": "float32",
      "shape": [1, 10],
      "required": true,
      "description": "Input data"
    }
  ],
  "outputs": [
    {
      "name": "output1",
      "type": "float32",
      "shape": [1, 2],
      "labels": ["class1", "class2"],
      "description": "Prediction results"
    }
  ]
}
```

#### `GET /metrics`

Get performance metrics.

**Response:**
```json
{
  "provider": {
    "provider": "CUDAExecutionProvider",
    "platform": "NVIDIA",
    "available": true,
    "description": "CUDA provider"
  },
  "available_providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

#### `POST /predict`

Run inference on input data.

**Request:**
```json
{
  "input1": [1, 2, 3, 4, 5],
  "input2": [0.1, 0.2, 0.3, 0.4, 0.5]
}
```

**Response:**
```json
{
  "output1": [0.8, 0.2],
  "output2": [0.1, 0.9]
}
```

**Error Responses:**
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Model not loaded or inference error

## Configuration Schema

### gpux.yml

```yaml
# Required
name: string                    # Model name
version: string                 # Model version

# Model configuration
model:
  source: string | Path         # Path to model file
  format: string                # Model format (default: onnx)
  version: string | null        # Model version override

# Input specifications
inputs:
  input_name:
    type: string                # Data type (float32, int64, string, etc.)
    shape: list[int]            # Input shape
    required: bool              # Whether input is required (default: true)
    description: string         # Input description

# Output specifications
outputs:
  output_name:
    type: string                # Data type
    shape: list[int]            # Output shape
    labels: list[string] | null # Class labels
    description: string         # Output description

# Runtime configuration
runtime:
  gpu:
    memory: string              # GPU memory (e.g., "2GB", "512MB")
    backend: string             # Backend (auto, cuda, coreml, rocm, etc.)
  batch_size: int               # Batch size (default: 1)
  timeout: int                  # Timeout in seconds (default: 30)

# Serving configuration
serving:
  port: int                     # Server port (default: 8080)
  host: string                  # Server host (default: 0.0.0.0)
  batch_size: int               # Serving batch size (default: 1)
  timeout: int                  # Request timeout (default: 5)

# Preprocessing configuration (optional)
preprocessing:
  tokenizer: string             # Tokenizer name
  max_length: int               # Maximum sequence length
  resize: list[int]             # Image resize dimensions
  normalize: string             # Normalization method
```

## Error Handling

### Common Exceptions

#### `GPUXError`
Base exception for all GPUX errors.

#### `ModelLoadError`
Raised when model loading fails.

#### `ProviderError`
Raised when execution provider issues occur.

#### `ValidationError`
Raised when input validation fails.

#### `ConfigurationError`
Raised when configuration is invalid.

### Error Response Format

```json
{
  "error": "Error type",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional error details"
  }
}
```

## Examples

### Basic Usage

```python
from gpux import GPUXRuntime

# Initialize runtime
runtime = GPUXRuntime("model.onnx")

# Run inference
results = runtime.infer({"input": [1, 2, 3, 4]})
print(results)

# Cleanup
runtime.cleanup()
```

### With Configuration

```python
from gpux import GPUXRuntime
from gpux.config.parser import GPUXConfigParser

# Parse configuration
parser = GPUXConfigParser()
config = parser.parse_file("gpux.yml")

# Initialize runtime with config
runtime = GPUXRuntime(
    model_path=config.model.source,
    provider=config.runtime.gpu.backend,
    gpu_memory=config.runtime.gpu.memory,
    batch_size=config.runtime.batch_size
)

# Run inference
results = runtime.infer({"input": data})
```

### Benchmarking

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime("model.onnx")

# Run benchmark
metrics = runtime.benchmark(
    input_data={"input": [1, 2, 3, 4]},
    num_runs=1000,
    warmup_runs=10
)

print(f"Average inference time: {metrics['avg_inference_time']:.2f}ms")
print(f"FPS: {metrics['fps']:.1f}")
```
