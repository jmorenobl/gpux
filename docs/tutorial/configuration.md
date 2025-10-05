# Configuration

Master the `gpux.yml` configuration file and customize your models for optimal performance.

---

## 🎯 What You'll Learn

- ✅ Complete `gpux.yml` structure
- ✅ Model configuration options
- ✅ Input and output specifications
- ✅ Runtime settings and GPU configuration
- ✅ Serving configuration
- ✅ Best practices and common patterns

---

## 📝 Configuration File Structure

The `gpux.yml` file is the heart of your GPUX project. It defines everything about your model:

```yaml
name: model-name              # Required: Model identifier
version: 1.0.0                # Required: Semantic version
description: "Description"     # Optional: Model description

model:                        # Required: Model configuration
  source: ./model.onnx
  format: onnx

inputs:                       # Required: Input specifications
  input_name:
    type: float32
    shape: [1, 10]
    required: true

outputs:                      # Required: Output specifications
  output_name:
    type: float32
    shape: [1, 2]

runtime:                      # Optional: Runtime settings
  gpu:
    memory: 2GB
    backend: auto
  batch_size: 1
  timeout: 30

serving:                      # Optional: HTTP server config
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5

preprocessing:                # Optional: Preprocessing config
  tokenizer: bert-base-uncased
  max_length: 512
```

---

## 📦 Model Configuration

The `model` section defines your ONNX model file:

### Basic Example

```yaml
model:
  source: ./model.onnx  # Path to ONNX file
  format: onnx          # Model format (currently only onnx)
```

### Path Options

=== "Relative Path"

    ```yaml
    model:
      source: ./models/sentiment.onnx
      format: onnx
    ```

=== "Absolute Path"

    ```yaml
    model:
      source: /Users/jorge/models/sentiment.onnx
      format: onnx
    ```

=== "HTTP URL"

    ```yaml
    model:
      source: https://example.com/models/model.onnx
      format: onnx
    ```

    !!! warning "HTTP URLs"
        HTTP model loading is planned but not yet implemented.

### Model Version

Optionally specify the model version separately from the project version:

```yaml
model:
  source: ./model.onnx
  format: onnx
  version: 2.0.0  # Model-specific version
```

---

## 📥 Input Configuration

Define your model's input specifications:

### Single Input

```yaml
inputs:
  text:
    type: string
    required: true
    max_length: 512
    description: "Input text for sentiment analysis"
```

### Multiple Inputs

```yaml
inputs:
  image:
    type: float32
    shape: [1, 3, 224, 224]
    required: true
    description: "RGB image tensor"

  mask:
    type: float32
    shape: [1, 1, 224, 224]
    required: false
    description: "Optional attention mask"
```

### Input Types

Supported data types:

| Type | Description | Example |
|------|-------------|---------|
| `float32` | 32-bit floating point | `[1.0, 2.5, 3.7]` |
| `float64` | 64-bit floating point | `[1.0, 2.5, 3.7]` |
| `int32` | 32-bit integer | `[1, 2, 3]` |
| `int64` | 64-bit integer | `[1, 2, 3]` |
| `bool` | Boolean | `[true, false]` |
| `string` | String | `"hello world"` |

### Shape Specification

Define tensor shapes:

```yaml
inputs:
  data:
    type: float32
    shape: [1, 10]        # Fixed shape: batch=1, features=10
```

#### Dynamic Shapes

Use `-1` or omit dimensions for dynamic shapes:

```yaml
inputs:
  data:
    type: float32
    shape: [-1, 10]       # Dynamic batch size, fixed features
```

```yaml
inputs:
  text:
    type: string
    shape: []             # Scalar (single value)
```

### Input Options

| Field | Required | Description |
|-------|----------|-------------|
| `type` | ✅ Yes | Data type |
| `shape` | No | Tensor shape |
| `required` | No | Whether input is required (default: `true`) |
| `max_length` | No | Maximum length for strings |
| `description` | No | Human-readable description |

---

## 📤 Output Configuration

Define your model's output specifications:

### Single Output

```yaml
outputs:
  sentiment:
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
    description: "Sentiment probabilities"
```

### Multiple Outputs

```yaml
outputs:
  logits:
    type: float32
    shape: [1, 1000]
    description: "Raw model outputs"

  probabilities:
    type: float32
    shape: [1, 1000]
    labels: [class1, class2, ...]  # 1000 classes
    description: "Softmax probabilities"
```

### Output Labels

Add human-readable labels for classification:

```yaml
outputs:
  emotion:
    type: float32
    shape: [1, 6]
    labels:
      - happy
      - sad
      - angry
      - surprised
      - neutral
      - fearful
    description: "Emotion classification"
```

### Output Options

| Field | Required | Description |
|-------|----------|-------------|
| `type` | ✅ Yes | Data type |
| `shape` | No | Tensor shape |
| `labels` | No | Class labels (for classification) |
| `description` | No | Human-readable description |

---

## ⚙️ Runtime Configuration

Configure GPU and performance settings:

### Complete Example

```yaml
runtime:
  gpu:
    memory: 4GB           # GPU memory limit
    backend: auto         # Provider selection
  batch_size: 1           # Default batch size
  timeout: 30             # Timeout in seconds
  enable_profiling: false # Enable performance profiling
```

### GPU Configuration

```yaml
runtime:
  gpu:
    memory: 2GB
    backend: auto  # auto, cuda, coreml, rocm, directml, openvino, cpu
```

#### Backend Options

| Backend | Description | Use Case |
|---------|-------------|----------|
| `auto` | Automatic selection | Default, recommended |
| `cuda` | NVIDIA CUDA | NVIDIA GPUs |
| `coreml` | Apple CoreML | Apple Silicon (M1/M2/M3) |
| `rocm` | AMD ROCm | AMD GPUs |
| `directml` | DirectML | Windows GPUs |
| `openvino` | Intel OpenVINO | Intel GPUs |
| `cpu` | CPU only | No GPU / debugging |

#### Memory Configuration

Specify GPU memory allocation:

```yaml
runtime:
  gpu:
    memory: 512MB   # Megabytes
    # or
    memory: 2GB     # Gigabytes
    # or
    memory: 4096    # Bytes
```

### Batch Size

Set default batch size for inference:

```yaml
runtime:
  batch_size: 1     # Process one sample at a time
  # or
  batch_size: 32    # Process 32 samples together
```

!!! tip "Batch Size Optimization"
    Larger batch sizes improve throughput but require more memory.
    Start with `batch_size: 1` and increase gradually.

### Timeout

Set maximum inference time:

```yaml
runtime:
  timeout: 30       # Seconds
```

### Performance Profiling

Enable detailed performance profiling:

```yaml
runtime:
  enable_profiling: true
```

This generates detailed timing information for debugging performance issues.

---

## 🌐 Serving Configuration

Configure HTTP server for production deployment:

### Basic Example

```yaml
serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5
```

### Complete Example

```yaml
serving:
  port: 8080              # HTTP port
  host: 0.0.0.0           # Bind address (0.0.0.0 = all interfaces)
  batch_size: 1           # Server batch size
  timeout: 5              # Request timeout (seconds)
  max_workers: 4          # Number of worker processes
```

### Serving Options

| Field | Default | Description |
|-------|---------|-------------|
| `port` | `8080` | HTTP server port |
| `host` | `0.0.0.0` | Bind address |
| `batch_size` | `1` | Batch size for requests |
| `timeout` | `5` | Request timeout (seconds) |
| `max_workers` | `4` | Worker processes |

!!! warning "Production Deployment"
    For production, use a reverse proxy (nginx, Caddy) in front of GPUX.

---

## 🔧 Preprocessing Configuration

Define preprocessing pipelines (advanced feature):

### Text Preprocessing

```yaml
preprocessing:
  tokenizer: bert-base-uncased
  max_length: 512
```

### Image Preprocessing

```yaml
preprocessing:
  resize: [224, 224]
  normalize: imagenet  # or custom values
```

### Custom Preprocessing

```yaml
preprocessing:
  custom:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    resize: [224, 224]
```

!!! info "Preprocessing Status"
    Preprocessing features are planned but not fully implemented yet.
    For now, preprocess data before sending to GPUX.

---

## 📋 Complete Examples

### Example 1: Text Classification

```yaml
name: sentiment-analysis
version: 1.0.0
description: "BERT-based sentiment analysis"

model:
  source: ./bert-sentiment.onnx
  format: onnx

inputs:
  text:
    type: string
    required: true
    max_length: 512
    description: "Input text to classify"

outputs:
  sentiment:
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
    description: "Sentiment probabilities"

runtime:
  gpu:
    memory: 2GB
    backend: auto
  batch_size: 1
  timeout: 30

serving:
  port: 8080
  host: 0.0.0.0
  timeout: 5
```

### Example 2: Image Classification

```yaml
name: image-classifier
version: 2.0.0
description: "ResNet-50 ImageNet classifier"

model:
  source: ./resnet50.onnx
  format: onnx

inputs:
  image:
    type: float32
    shape: [1, 3, 224, 224]
    required: true
    description: "RGB image tensor (normalized)"

outputs:
  probabilities:
    type: float32
    shape: [1, 1000]
    description: "ImageNet class probabilities"

runtime:
  gpu:
    memory: 4GB
    backend: auto
  batch_size: 8
  timeout: 10

serving:
  port: 9000
  host: 127.0.0.1
  batch_size: 16
  timeout: 10
  max_workers: 2
```

### Example 3: Multi-Input Model

```yaml
name: multi-modal-model
version: 1.0.0
description: "Image + text multi-modal model"

model:
  source: ./clip-model.onnx
  format: onnx

inputs:
  image:
    type: float32
    shape: [1, 3, 224, 224]
    required: true
    description: "Image tensor"

  text:
    type: string
    required: true
    max_length: 77
    description: "Text description"

outputs:
  similarity:
    type: float32
    shape: [1, 1]
    description: "Image-text similarity score"

runtime:
  gpu:
    memory: 8GB
    backend: auto
  batch_size: 1
  timeout: 15
```

---

## ✅ Validation

Validate your configuration file:

```bash
# Build validates configuration
gpux build .

# Or use Python
python -c "from gpux.config.parser import GPUXConfigParser; GPUXConfigParser().parse_file('gpux.yml')"
```

---

## 🎓 Best Practices

### 1. Use Descriptive Names

❌ **Bad:**
```yaml
name: model1
```

✅ **Good:**
```yaml
name: bert-sentiment-analysis
description: "BERT-base fine-tuned on IMDB sentiment"
```

### 2. Document Inputs/Outputs

❌ **Bad:**
```yaml
inputs:
  x:
    type: float32
```

✅ **Good:**
```yaml
inputs:
  text_embeddings:
    type: float32
    shape: [1, 768]
    description: "BERT embeddings for input text"
```

### 3. Start Conservative

Start with conservative settings and optimize later:

```yaml
runtime:
  gpu:
    memory: 2GB      # Start small
    backend: auto    # Let GPUX choose
  batch_size: 1      # Start with 1
  timeout: 30        # Generous timeout
```

### 4. Use Semantic Versioning

```yaml
version: 1.0.0  # Major.Minor.Patch
```

- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

### 5. Environment-Specific Configs

Create separate configs for different environments:

```
project/
├── gpux.yml              # Default/development
├── gpux.prod.yml         # Production
└── gpux.test.yml         # Testing
```

Use with:
```bash
gpux build . --config gpux.prod.yml
```

---

## 🐛 Common Issues

### Invalid YAML Syntax

**Error**: `Invalid YAML in configuration file`

**Solution**: Check indentation and syntax:
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('gpux.yml'))"
```

### Missing Required Fields

**Error**: `At least one input must be specified`

**Solution**: Ensure you have all required sections:
- ✅ `name`
- ✅ `version`
- ✅ `model`
- ✅ `inputs`
- ✅ `outputs`

### Type Mismatch

**Error**: `Type mismatch for input: expected float32, got int32`

**Solution**: Ensure input types in `gpux.yml` match your ONNX model:
```bash
gpux inspect my-model  # Check actual model types
```

---

## 📚 What's Next?

Now that you understand configuration, learn how to run inference:

- **[Running Inference →](running-inference.md)** - Master the `gpux run` command
- **[Benchmarking →](benchmarking.md)** - Measure performance
- **[API Reference](../reference/configuration/schema.md)** - Complete schema reference

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ Complete `gpux.yml` structure
    ✅ How to configure inputs and outputs
    ✅ Runtime and GPU settings
    ✅ Serving configuration for production
    ✅ Best practices for configuration
    ✅ Common issues and solutions

---

**Previous:** [First Steps](first-steps.md) | **Next:** [Running Inference](running-inference.md)
