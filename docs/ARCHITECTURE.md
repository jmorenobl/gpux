# GPUX Architecture

## Overview

GPUX is designed as a platform layer that provides universal GPU compatibility for ML inference. It sits between your ML models and the underlying hardware, abstracting away GPU-specific complexities.

## Core Components

### 1. Runtime Engine (`src/gpux/core/runtime.py`)

The `GPUXRuntime` class is the heart of the system:

- **Model Loading**: Loads ONNX models and creates inference sessions
- **Provider Management**: Automatically selects the best execution provider
- **Inference Execution**: Runs inference with proper input/output handling
- **Performance Monitoring**: Tracks metrics and provides benchmarking

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime(
    model_path="model.onnx",
    provider="auto",  # or specific provider
    gpu_memory="2GB",
    batch_size=1
)

results = runtime.infer({"input": data})
```

### 2. Provider System (`src/gpux/core/providers.py`)

Intelligent provider selection based on hardware capabilities:

```python
from gpux.core.providers import ProviderManager, ExecutionProvider

manager = ProviderManager()
provider = manager.get_best_provider()  # Auto-select
provider = manager.get_best_provider("cuda")  # Specific provider
```

**Provider Priority Order:**
1. TensorRT (NVIDIA, best performance)
2. CUDA (NVIDIA, good performance)
3. ROCm (AMD, good performance)
4. CoreML (Apple Silicon, optimized)
5. DirectML (Windows, GPU acceleration)
6. OpenVINO (Intel, optimization)
7. CPU (Universal fallback)

### 3. Model Management (`src/gpux/core/models.py`)

Model inspection and metadata handling:

```python
from gpux.core.models import ModelInspector

inspector = ModelInspector()
info = inspector.inspect("model.onnx")
print(f"Inputs: {info.inputs}")
print(f"Outputs: {info.outputs}")
print(f"Size: {info.size_mb} MB")
```

### 4. Configuration System (`src/gpux/config/parser.py`)

YAML-based configuration with validation:

```yaml
# gpux.yml
name: my-model
version: 1.0.0

model:
  source: ./model.onnx
  format: onnx

inputs:
  input1:
    type: float32
    shape: [1, 10]
    required: true

outputs:
  output1:
    type: float32
    shape: [1, 2]
    labels: [class1, class2]

runtime:
  gpu:
    memory: 2GB
    backend: auto
```

### 5. CLI Interface (`src/gpux/cli/`)

Command-line interface built with Typer:

- `gpux build` - Build and optimize models
- `gpux run` - Run inference
- `gpux serve` - Start HTTP server
- `gpux inspect` - Inspect models

### 6. Utilities (`src/gpux/utils/helpers.py`)

Helper functions for common tasks:

- System information gathering
- File operations
- Command execution
- Dependency checking

## Data Flow

```
User Input (CLI/API)
    ↓
Configuration Parser (gpux.yml)
    ↓
Runtime Engine (GPUXRuntime)
    ↓
Provider Manager (Auto-select best provider)
    ↓
ONNX Runtime (Hardware-specific execution)
    ↓
Results (Formatted output)
```

## Design Principles

### 1. Universal Compatibility
- Works on any GPU through execution providers
- Automatic fallback to CPU if GPU unavailable
- Cross-platform support (Windows, macOS, Linux)

### 2. Zero Configuration
- Auto-detects best execution provider
- Sensible defaults for all parameters
- Works out of the box

### 3. Docker-like UX
- Familiar commands (`build`, `run`, `serve`)
- Configuration files (`gpux.yml`)
- Container-like model packaging

### 4. High Performance
- Optimized ONNX Runtime backends
- Provider-specific optimizations
- Efficient memory management

### 5. Developer Experience
- Rich terminal output
- Comprehensive error messages
- Extensive logging and debugging

## Extension Points

### Custom Providers
You can add custom execution providers by extending the `ExecutionProvider` enum and implementing provider-specific logic.

### Custom Preprocessing
Add preprocessing steps by extending the configuration schema and implementing custom handlers.

### Custom Output Formats
Extend the output formatting system to support custom result formats.

## Performance Considerations

### Memory Management
- Automatic GPU memory allocation
- Configurable memory limits
- Efficient tensor operations

### Batch Processing
- Support for batch inference
- Configurable batch sizes
- Memory-efficient batching

### Provider Selection
- Runtime provider detection
- Performance-based selection
- Fallback mechanisms

## Security

### Model Validation
- ONNX model validation
- Input/output shape checking
- Type safety enforcement

### Sandboxing
- Isolated execution environments
- Resource limits
- Safe model loading

## Monitoring and Observability

### Metrics Collection
- Inference latency
- Throughput (FPS)
- Memory usage
- Provider performance

### Logging
- Structured logging
- Configurable log levels
- Performance traces

### Health Checks
- Model loading status
- Provider availability
- System health monitoring
