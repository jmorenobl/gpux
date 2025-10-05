# Architecture

GPUX system architecture and design principles.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  CLI (build/run/serve/inspect)  │  HTTP API (FastAPI)   │
├─────────────────────────────────────────────────────────┤
│                   GPUX Runtime Layer                     │
│  • Configuration Parser  • Model Inspector              │
│  • Provider Manager      • Inference Engine             │
├─────────────────────────────────────────────────────────┤
│                  ONNX Runtime (Core)                     │
│  • Model Loading  • Graph Optimization                  │
│  • Memory Management  • Execution Orchestration         │
├─────────────────────────────────────────────────────────┤
│              Execution Providers (Backends)              │
│  TensorRT│CUDA│ROCm│CoreML│DirectML│OpenVINO│CPU       │
├─────────────────────────────────────────────────────────┤
│                     Hardware Layer                       │
│  NVIDIA GPU │ AMD GPU │ Apple Silicon │ Intel │ CPU     │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. CLI Layer (`cli/`)

**Purpose**: User-facing command-line interface

**Components**:
- `main.py` - Entry point and command routing
- `build.py` - Build and validate command
- `run.py` - Inference execution command
- `serve.py` - HTTP server command
- `inspect.py` - Model inspection command

**Technology**: Typer (CLI framework)

### 2. Core Runtime (`core/`)

**Purpose**: Core inference and provider management

**Components**:
- `runtime.py` - `GPUXRuntime` main class
- `providers.py` - Execution provider selection
- `models.py` - Model introspection

**Key Classes**:
```python
class GPUXRuntime:
    - load_model()
    - infer()
    - benchmark()
    - get_provider_info()

class ProviderManager:
    - get_best_provider()
    - get_available_providers()

class ModelInspector:
    - inspect()
    - extract_metadata()
```

### 3. Configuration (`config/`)

**Purpose**: Configuration parsing and validation

**Components**:
- `parser.py` - YAML parsing with Pydantic

**Models**:
```python
GPUXConfig
├── ModelConfig
├── InputConfig
├── OutputConfig
├── RuntimeConfig
│   └── GPUConfig
├── ServingConfig
└── PreprocessingConfig
```

### 4. Utilities (`utils/`)

**Purpose**: Helper functions and utilities

**Components**:
- `helpers.py` - System info, file operations

---

## Data Flow

### Build Flow

```
gpux build .
    ↓
1. Parse gpux.yml → GPUXConfig
2. Validate model path
3. Inspect ONNX model → ModelInfo
4. Check provider compatibility
5. Save build artifacts (.gpux/)
    ↓
Build artifacts:
- model_info.json
- provider_info.json
```

### Run Flow

```
gpux run model-name --input data.json
    ↓
1. Load configuration
2. Initialize GPUXRuntime
3. Select execution provider
4. Create ONNX Runtime session
5. Prepare input data
6. Execute inference
7. Format and return results
```

### Serve Flow

```
gpux serve model-name --port 8080
    ↓
1. Load configuration
2. Initialize GPUXRuntime
3. Create FastAPI app
4. Register endpoints:
   - POST /predict
   - GET /health
   - GET /info
   - GET /metrics
5. Start Uvicorn server
```

---

## Provider Selection

### Selection Algorithm

```python
def get_best_provider(preferred=None):
    # 1. If preferred provider specified, try to use it
    if preferred:
        if is_available(preferred):
            return preferred
        else:
            log_warning(f"{preferred} not available, falling back")

    # 2. Try providers in priority order
    for provider in priority_list:
        if is_available(provider):
            return provider

    # 3. Fallback to CPU
    return CPUExecutionProvider
```

### Priority Order

1. **TensorRT** (NVIDIA - best performance)
2. **CUDA** (NVIDIA - good performance)
3. **ROCm** (AMD)
4. **CoreML** (Apple Silicon)
5. **DirectML** (Windows)
6. **OpenVINO** (Intel)
7. **CPU** (Universal fallback)

### Platform-Specific Priority

**Apple Silicon (M1/M2/M3/M4)**:
```python
priority = [CoreML, CPU]
```

**Windows**:
```python
priority = [TensorRT, CUDA, DirectML, OpenVINO, CPU]
```

---

## Configuration Schema

### File Format (gpux.yml)

```yaml
# Required
name: string
version: string
model:
  source: path
  format: onnx
inputs: [InputSpec]
outputs: [OutputSpec]

# Optional
runtime:
  gpu:
    backend: auto|cuda|coreml|rocm|...
    memory: 2GB
  batch_size: 1
  timeout: 30

serving:
  port: 8080
  host: 0.0.0.0
  max_workers: 4

preprocessing:
  tokenizer: string
  resize: [width, height]
  normalize: method
```

### Validation Pipeline

```
YAML file
    ↓
yaml.safe_load()
    ↓
Normalize (dict → list for inputs/outputs)
    ↓
Pydantic validation
    ↓
GPUXConfig object
```

---

## Inference Pipeline

### Step-by-Step

1. **Input Preparation**
   ```python
   # Convert to numpy arrays
   numpy_input = {
       key: np.array(value) if isinstance(value, list) else value
       for key, value in input_data.items()
   }
   ```

2. **Session Execution**
   ```python
   # ONNX Runtime inference
   results = session.run(
       output_names,
       numpy_input
   )
   ```

3. **Output Formatting**
   ```python
   # Convert to JSON-serializable
   output_data = {
       key: value.tolist() if hasattr(value, 'tolist') else value
       for key, value in results.items()
   }
   ```

---

## HTTP API Architecture

### FastAPI Application

```python
app = FastAPI(
    title="GPUX Server",
    version=config.version
)

# Endpoints
@app.post("/predict")
async def predict(data: dict) -> dict

@app.get("/health")
async def health() -> dict

@app.get("/info")
async def info() -> ModelInfo

@app.get("/metrics")
async def metrics() -> dict
```

### Request Flow

```
HTTP Request
    ↓
FastAPI routing
    ↓
Input validation
    ↓
GPUXRuntime.infer()
    ↓
ONNX Runtime execution
    ↓
Result formatting
    ↓
HTTP Response (JSON)
```

---

## Design Principles

### 1. Simplicity First
- Single configuration file
- Docker-like commands
- Sensible defaults

### 2. Universal Compatibility
- Any GPU, any platform
- Automatic provider selection
- Graceful fallback to CPU

### 3. Performance
- Optimized execution providers
- Efficient memory management
- Minimal overhead

### 4. Extensibility
- Plugin architecture for providers
- Configurable preprocessing
- Custom endpoints

---

## Technology Choices

### Why ONNX Runtime?

✅ **Pros**:
- Battle-tested (Microsoft production)
- Universal GPU support
- Excellent performance
- Active development

❌ **Alternatives Rejected**:
- PyTorch: NVIDIA-focused
- TensorFlow: Framework-specific
- Raw WebGPU: Too low-level, immature

### Why Typer for CLI?

✅ **Pros**:
- Modern Python CLI framework
- Type-safe
- Auto-generated help
- Subcommand support

### Why FastAPI for HTTP?

✅ **Pros**:
- Fast, async
- Auto OpenAPI docs
- Type validation
- Modern Python

---

## Performance Considerations

### Memory Management

- Configurable GPU memory limits
- Automatic memory cleanup
- Session caching (planned)

### Execution Optimization

- Provider-specific optimizations
- Dynamic batching (planned)
- Graph optimization (ONNX Runtime)

### Concurrency

- Multi-worker HTTP serving
- Process-based parallelism
- GPU memory per worker

---

## Security

### Current

- Input validation (Pydantic)
- Safe YAML parsing
- No code execution

### Planned

- Authentication (API keys, OAuth)
- Model encryption
- Secure serving (HTTPS, mTLS)
- Audit logging

---

## Testing Strategy

### Unit Tests
- Component isolation
- Mocked ONNX Runtime
- 90%+ coverage target

### Integration Tests
- End-to-end CLI tests
- HTTP API tests
- Multi-platform validation

### Performance Tests
- Benchmark suite
- Regression detection
- Platform-specific tests

---

## Deployment Patterns

### 1. Single Instance
```
Client → GPUX Server → GPU
```

### 2. Load Balanced
```
         ┌→ GPUX Server 1 → GPU 1
Client → LB → GPUX Server 2 → GPU 2
         └→ GPUX Server 3 → GPU 3
```

### 3. Microservices
```
API Gateway
    ↓
GPUX Model A (Sentiment)
GPUX Model B (NER)
GPUX Model C (Classification)
```

---

## Future Architecture

### Planned Enhancements

1. **Model Registry**
   - Centralized model storage
   - Version management
   - Metadata tracking

2. **Distributed Inference**
   - Multi-GPU support
   - Model sharding
   - Pipeline parallelism

3. **Advanced Monitoring**
   - Prometheus metrics
   - OpenTelemetry tracing
   - Performance profiling

---

## See Also

- [About GPUX](index.md)
- [Contributing Guide](contributing.md)
- [Developer Docs](../guide/index.md)
