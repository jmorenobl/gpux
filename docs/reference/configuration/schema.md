# Configuration Schema

Complete reference for `gpux.yml` configuration file.

---

## Overview

The `gpux.yml` file is the single source of truth for GPUX model configuration. It defines everything from model paths to runtime settings and serving configuration.

```yaml
name: string              # Required: Model name
version: string           # Optional: Model version (default: "1.0.0")
description: string       # Optional: Model description

model:                    # Required: Model configuration
  source: string          # Required: Path to model file
  format: string          # Optional: Model format (default: "onnx")
  version: string         # Optional: Model version

inputs:                   # Required: Input specifications
  - name: string          # Required: Input name
    type: string          # Required: Data type
    shape: [int]          # Optional: Tensor shape
    required: bool        # Optional: Required field (default: true)
    max_length: int       # Optional: Maximum length
    description: string   # Optional: Input description

outputs:                  # Required: Output specifications
  - name: string          # Required: Output name
    type: string          # Required: Data type
    shape: [int]          # Optional: Tensor shape
    labels: [string]      # Optional: Class labels
    description: string   # Optional: Output description

runtime:                  # Optional: Runtime configuration
  gpu:
    memory: string        # GPU memory limit (default: "2GB")
    backend: string       # GPU backend (default: "auto")
  timeout: int            # Timeout in seconds (default: 30)
  batch_size: int         # Batch size (default: 1)
  enable_profiling: bool  # Enable profiling (default: false)

serving:                  # Optional: HTTP serving configuration
  port: int               # Server port (default: 8080)
  host: string            # Server host (default: "0.0.0.0")
  batch_size: int         # Serving batch size (default: 1)
  timeout: int            # Request timeout (default: 5)
  max_workers: int        # Max worker processes (default: 4)

preprocessing:            # Optional: Preprocessing configuration
  tokenizer: string       # Tokenizer name
  max_length: int         # Max tokenization length
  resize: [int, int]      # Image resize dimensions
  normalize: string       # Normalization method
  custom: {}              # Custom preprocessing config

metadata:                 # Optional: Custom metadata
  key: value              # Any custom key-value pairs
```

---

## Minimal Example

```yaml
name: sentiment-analysis
version: 1.0.0

model:
  source: ./model.onnx

inputs:
  - name: text
    type: string
    required: true

outputs:
  - name: sentiment
    type: float32
    shape: [2]
    labels: [negative, positive]
```

---

## Complete Example

```yaml
name: sentiment-analysis
version: 1.0.0
description: BERT-based sentiment analysis model

model:
  source: ./model.onnx
  format: onnx
  version: 1.0.0

inputs:
  - name: input_ids
    type: int64
    shape: [1, 128]
    required: true
    description: Tokenized input IDs
  - name: attention_mask
    type: int64
    shape: [1, 128]
    required: true
    description: Attention mask for input

outputs:
  - name: logits
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
    description: Sentiment classification logits

runtime:
  gpu:
    memory: 2GB
    backend: auto
  timeout: 30
  batch_size: 1
  enable_profiling: false

serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5
  max_workers: 4

preprocessing:
  tokenizer: bert-base-uncased
  max_length: 128

metadata:
  author: GPUX Team
  license: MIT
  dataset: SST-2
```

---

## Top-Level Fields

### `name` *(required)*

Model name used for identification.

- **Type**: `string`
- **Required**: Yes
- **Example**: `sentiment-analysis`, `image-classifier`

```yaml
name: sentiment-analysis
```

### `version`

Model version following semantic versioning.

- **Type**: `string`
- **Required**: No
- **Default**: `1.0.0`
- **Example**: `1.0.0`, `2.1.3`

```yaml
version: 1.0.0
```

### `description`

Human-readable model description.

- **Type**: `string`
- **Required**: No
- **Example**: `BERT-based sentiment analysis`

```yaml
description: BERT-based sentiment analysis model for binary classification
```

---

## Section References

Detailed documentation for each configuration section:

- **[Model](model.md)** - Model source and format configuration
- **[Inputs](inputs.md)** - Input specifications and validation
- **[Outputs](outputs.md)** - Output specifications and labels
- **[Runtime](runtime.md)** - GPU, timeout, and batch settings
- **[Serving](serving.md)** - HTTP server configuration
- **[Preprocessing](preprocessing.md)** - Data preprocessing settings

---

## Data Types

### Supported Types

GPUX supports the following data types:

| Type | Description | Example |
|------|-------------|---------|
| `float32` | 32-bit floating point | `[0.5, 1.2, -0.3]` |
| `float64` | 64-bit floating point | `[0.123456789]` |
| `int32` | 32-bit integer | `[1, 2, 3]` |
| `int64` | 64-bit integer | `[100, 200, 300]` |
| `uint8` | 8-bit unsigned integer | `[0, 255]` |
| `bool` | Boolean | `[true, false]` |
| `string` | Text string | `["hello", "world"]` |

### Type Conversion

GPUX automatically converts compatible types:

- Python lists → NumPy arrays
- JSON numbers → float32/int64
- JSON strings → string tensors

---

## Shape Specifications

### Fixed Shapes

Specify exact tensor dimensions:

```yaml
inputs:
  - name: image
    type: float32
    shape: [1, 3, 224, 224]  # [batch, channels, height, width]
```

### Dynamic Shapes

Use `-1` or omit shape for dynamic dimensions:

```yaml
inputs:
  - name: text
    type: int64
    shape: [1, -1]  # Variable sequence length
```

Or omit shape entirely:

```yaml
inputs:
  - name: text
    type: int64  # Fully dynamic shape
```

---

## Alternative Syntax

### Dict-Style Inputs/Outputs

You can also use dictionary syntax for inputs and outputs:

```yaml
inputs:
  input_ids:
    type: int64
    shape: [1, 128]
    required: true
  attention_mask:
    type: int64
    shape: [1, 128]

outputs:
  logits:
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
```

This is equivalent to the list syntax:

```yaml
inputs:
  - name: input_ids
    type: int64
    shape: [1, 128]
    required: true
  - name: attention_mask
    type: int64
    shape: [1, 128]

outputs:
  - name: logits
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
```

---

## Validation Rules

### Required Fields

The following fields are required:

- `name` - Model name
- `model.source` - Model file path
- `inputs` - At least one input
- `outputs` - At least one output

### Input Validation

- At least one input must be specified
- Each input must have `name` and `type`
- `shape` is optional but recommended
- `required` defaults to `true`

### Output Validation

- At least one output must be specified
- Each output must have `name` and `type`
- `labels` should match output shape

### Memory Validation

GPU memory must be specified with units:

```yaml
runtime:
  gpu:
    memory: 2GB    # ✅ Valid
    memory: 512MB  # ✅ Valid
    memory: 1024KB # ✅ Valid
    memory: 2      # ❌ Invalid (missing units)
```

---

## Environment Variables

You can use environment variables in configuration:

```yaml
model:
  source: ${MODEL_PATH}/model.onnx

runtime:
  gpu:
    memory: ${GPU_MEMORY:-2GB}  # Default to 2GB

serving:
  port: ${PORT:-8080}
```

---

## File Paths

### Relative Paths

Paths are relative to the `gpux.yml` file:

```yaml
model:
  source: ./model.onnx        # Same directory
  source: ./models/model.onnx # Subdirectory
  source: ../model.onnx       # Parent directory
```

### Absolute Paths

You can also use absolute paths:

```yaml
model:
  source: /opt/models/sentiment.onnx
```

---

## Common Patterns

### Image Classification

```yaml
name: image-classifier
model:
  source: ./resnet50.onnx
inputs:
  - name: image
    type: float32
    shape: [1, 3, 224, 224]
outputs:
  - name: probabilities
    type: float32
    shape: [1, 1000]
preprocessing:
  resize: [224, 224]
  normalize: imagenet
```

### Text Classification

```yaml
name: sentiment-analysis
model:
  source: ./bert.onnx
inputs:
  - name: input_ids
    type: int64
    shape: [1, 128]
  - name: attention_mask
    type: int64
    shape: [1, 128]
outputs:
  - name: logits
    type: float32
    shape: [1, 2]
    labels: [negative, positive]
preprocessing:
  tokenizer: bert-base-uncased
  max_length: 128
```

### Object Detection

```yaml
name: yolo-detector
model:
  source: ./yolov8.onnx
inputs:
  - name: images
    type: float32
    shape: [1, 3, 640, 640]
outputs:
  - name: boxes
    type: float32
    shape: [-1, 4]
  - name: scores
    type: float32
    shape: [-1]
  - name: classes
    type: int64
    shape: [-1]
preprocessing:
  resize: [640, 640]
```

---

## Best Practices

!!! tip "Always Specify Shapes"
    Include shape information for better validation and performance:
    ```yaml
    inputs:
      - name: input
        type: float32
        shape: [1, 10]  # ✅ Recommended
    ```

!!! tip "Use Descriptive Names"
    Choose clear, descriptive names for inputs/outputs:
    ```yaml
    inputs:
      - name: input_ids        # ✅ Clear
        # vs
      - name: x                # ❌ Unclear
    ```

!!! tip "Document with Descriptions"
    Add descriptions for complex inputs/outputs:
    ```yaml
    inputs:
      - name: attention_mask
        type: int64
        description: Binary mask indicating valid tokens
    ```

!!! warning "GPU Memory Limits"
    Set appropriate GPU memory limits based on model size:
    ```yaml
    runtime:
      gpu:
        memory: 2GB  # Adjust based on model
    ```

---

## Troubleshooting

### Validation Errors

**Error**: "At least one input must be specified"
```yaml
# ❌ Missing inputs
name: model
model:
  source: ./model.onnx

# ✅ Fixed
name: model
model:
  source: ./model.onnx
inputs:
  - name: input
    type: float32
```

**Error**: "Memory must be specified as GB, MB, or KB"
```yaml
# ❌ Missing units
runtime:
  gpu:
    memory: 2

# ✅ Fixed
runtime:
  gpu:
    memory: 2GB
```

---

## See Also

- [Model Configuration](model.md)
- [Input Configuration](inputs.md)
- [Output Configuration](outputs.md)
- [Runtime Configuration](runtime.md)
- [Serving Configuration](serving.md)
- [Preprocessing Configuration](preprocessing.md)
