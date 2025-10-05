# `gpux inspect`

Inspect models and runtime information.

---

## Overview

The `gpux inspect` command provides detailed information about models, their inputs/outputs, metadata, and available execution providers. It's essential for understanding model specifications and debugging.

```bash
gpux inspect [MODEL_NAME] [OPTIONS]
```

---

## Arguments

### `MODEL_NAME`

Name of the model to inspect (optional).

- **Type**: `string`
- **Required**: No

**Behavior**:
- If provided: Inspects the specified model
- If omitted: Shows runtime information (available providers)

**Examples**:
```bash
gpux inspect sentiment-analysis    # Inspect specific model
gpux inspect                       # Show runtime info
```

---

## Options

### `--config`, `-c`

Configuration file name.

- **Type**: `string`
- **Default**: `gpux.yml`

```bash
gpux inspect sentiment --config custom.yml
```

### `--model`, `-m`

Direct path to model file (bypasses model lookup).

- **Type**: `string`

```bash
gpux inspect --model ./model.onnx
gpux inspect -m /path/to/model.onnx
```

### `--json`

Output in JSON format (useful for scripting).

- **Type**: `boolean`
- **Default**: `false`

```bash
gpux inspect sentiment --json
gpux inspect --json > runtime-info.json
```

### `--verbose`

Enable verbose output.

- **Type**: `boolean`
- **Default**: `false`

```bash
gpux inspect sentiment --verbose
```

---

## Inspection Modes

### 1. Inspect Model by Name

Inspect a model using its name:

```bash
gpux inspect sentiment-analysis
```

**Output**:

#### Configuration
| Property | Value |
|----------|-------|
| Name | sentiment-analysis |
| Version | 1.0.0 |
| Model Source | ./model.onnx |
| Model Format | onnx |
| GPU Memory | 2GB |
| GPU Backend | auto |
| Batch Size | 1 |
| Timeout | 30s |

#### Model Information
| Property | Value |
|----------|-------|
| Name | sentiment-analysis |
| Version | 1.0.0 |
| Format | onnx |
| Size | 256.0 MB |
| Path | ./model.onnx |

#### Input Specifications
| Name | Type | Shape | Required | Description |
|------|------|-------|----------|-------------|
| input_ids | int64 | [1, 128] | ✅ | Tokenized input IDs |
| attention_mask | int64 | [1, 128] | ✅ | Attention mask |

#### Output Specifications
| Name | Type | Shape | Labels | Description |
|------|------|-------|--------|-------------|
| logits | float32 | [1, 2] | negative, positive | Sentiment logits |

### 2. Inspect Model File

Inspect a model file directly (no configuration required):

```bash
gpux inspect --model ./model.onnx
```

**Output**:

#### Model Information
| Property | Value |
|----------|-------|
| Name | model |
| Version | 1 |
| Format | onnx |
| Size | 256.0 MB |
| Path | ./model.onnx |

#### Input Specifications
| Name | Type | Shape | Required | Description |
|------|------|-------|----------|-------------|
| input | float32 | [1, 3, 224, 224] | ✅ | N/A |

#### Output Specifications
| Name | Type | Shape | Labels | Description |
|------|------|-------|--------|-------------|
| output | float32 | [1, 1000] | N/A | N/A |

### 3. Inspect Runtime

Show available execution providers (no model name):

```bash
gpux inspect
```

**Output**:

#### Available Execution Providers
| Provider | Available | Platform | Description |
|----------|-----------|----------|-------------|
| TensorrtExecutionProvider | ❌ | NVIDIA TensorRT | NVIDIA TensorRT optimization |
| CUDAExecutionProvider | ✅ | NVIDIA CUDA | NVIDIA CUDA GPU acceleration |
| ROCmExecutionProvider | ❌ | AMD ROCm | AMD GPU acceleration |
| CoreMLExecutionProvider | ❌ | Apple CoreML | Apple Silicon optimization |
| DmlExecutionProvider | ❌ | DirectML | Windows DirectX acceleration |
| OpenVINOExecutionProvider | ❌ | Intel OpenVINO | Intel hardware acceleration |
| CPUExecutionProvider | ✅ | CPU | Universal CPU fallback |

#### Provider Priority
| Priority | Provider | Status |
|----------|----------|--------|
| 1 | TensorrtExecutionProvider | Not Available |
| 2 | CUDAExecutionProvider | Available |
| 3 | ROCmExecutionProvider | Not Available |
| 4 | CoreMLExecutionProvider | Not Available |
| 5 | DmlExecutionProvider | Not Available |
| 6 | OpenVINOExecutionProvider | Not Available |
| 7 | CPUExecutionProvider | Available |

---

## JSON Output

### Model Inspection (JSON)

```bash
gpux inspect sentiment --json
```

**Output**:
```json
{
  "config": {
    "name": "sentiment-analysis",
    "version": "1.0.0",
    "model": {
      "source": "./model.onnx",
      "format": "onnx"
    },
    "inputs": {
      "input_ids": {
        "type": "int64",
        "shape": [1, 128],
        "required": true
      }
    },
    "outputs": {
      "logits": {
        "type": "float32",
        "shape": [1, 2],
        "labels": ["negative", "positive"]
      }
    },
    "runtime": {
      "gpu": {
        "memory": "2GB",
        "backend": "auto"
      },
      "batch_size": 1,
      "timeout": 30
    }
  },
  "model_info": {
    "name": "sentiment-analysis",
    "version": "1.0.0",
    "format": "onnx",
    "size_mb": 256.0,
    "path": "./model.onnx",
    "inputs": [
      {
        "name": "input_ids",
        "type": "int64",
        "shape": [1, 128],
        "required": true,
        "description": "Tokenized input IDs"
      }
    ],
    "outputs": [
      {
        "name": "logits",
        "type": "float32",
        "shape": [1, 2],
        "labels": ["negative", "positive"]
      }
    ]
  }
}
```

### Runtime Inspection (JSON)

```bash
gpux inspect --json
```

**Output**:
```json
{
  "available_providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ],
  "provider_details": {
    "TensorrtExecutionProvider": {
      "available": false,
      "platform": "NVIDIA TensorRT",
      "description": "NVIDIA TensorRT optimization"
    },
    "CUDAExecutionProvider": {
      "available": true,
      "platform": "NVIDIA CUDA",
      "description": "NVIDIA CUDA GPU acceleration"
    },
    "CPUExecutionProvider": {
      "available": true,
      "platform": "CPU",
      "description": "Universal CPU fallback"
    }
  }
}
```

---

## Examples

### Inspect Sentiment Model

```bash
gpux inspect sentiment-analysis
```

### Inspect ONNX File Directly

```bash
gpux inspect --model ./models/bert-base.onnx
```

### Check Available Providers

```bash
gpux inspect
```

### JSON Output for Scripting

```bash
gpux inspect sentiment --json | jq '.model_info.size_mb'
# Output: 256.0
```

### Save Inspection to File

```bash
gpux inspect sentiment --json > model-info.json
```

### Check if GPU is Available

```bash
gpux inspect --json | jq '.available_providers | contains(["CUDAExecutionProvider"])'
# Output: true or false
```

---

## Use Cases

### 1. Verify Model Inputs/Outputs

Before running inference, check expected inputs:

```bash
gpux inspect sentiment
```

### 2. Debug Configuration Issues

Verify configuration is correctly parsed:

```bash
gpux inspect sentiment --verbose
```

### 3. Check Provider Availability

Ensure GPU providers are available:

```bash
gpux inspect
```

### 4. Automate Model Validation

Use JSON output in CI/CD:

```bash
#!/bin/bash
SIZE=$(gpux inspect sentiment --json | jq '.model_info.size_mb')
if (( $(echo "$SIZE > 500" | bc -l) )); then
  echo "Error: Model too large ($SIZE MB)"
  exit 1
fi
```

### 5. Generate Model Documentation

Extract model specs for documentation:

```bash
gpux inspect model --json | jq '.model_info.inputs'
```

---

## Error Handling

### Model Not Found

```bash
Error: Model 'sentiment-analysis' not found
```

**Solution**: Ensure the model exists and `gpux.yml` is configured.

### Model File Not Found

```bash
Error: Model file not found: ./model.onnx
```

**Solution**: Check the `model.source` path in `gpux.yml`.

### Invalid Model File

```bash
Inspect failed: Invalid ONNX model
```

**Solution**: Verify the ONNX model is valid:
```bash
python -c "import onnx; onnx.checker.check_model('model.onnx')"
```

---

## Best Practices

!!! tip "Inspect Before Running"
    Always inspect a model before running inference to understand its inputs/outputs:
    ```bash
    gpux inspect sentiment
    gpux run sentiment --input '{"text": "test"}'
    ```

!!! tip "Use JSON for Automation"
    Use `--json` flag for scripting and automation:
    ```bash
    gpux inspect model --json | jq '.model_info.size_mb'
    ```

!!! tip "Check Providers Before Deployment"
    Verify GPU providers are available on target platform:
    ```bash
    gpux inspect --json | jq '.available_providers'
    ```

!!! tip "Save Model Info"
    Save inspection results for documentation:
    ```bash
    gpux inspect sentiment --json > docs/model-spec.json
    ```

---

## Output Fields

### Model Information

- `name`: Model name
- `version`: Model version
- `format`: Model format (onnx)
- `size_mb`: Model size in megabytes
- `path`: Path to model file

### Input Specifications

- `name`: Input name
- `type`: Data type (float32, int64, etc.)
- `shape`: Tensor shape
- `required`: Whether input is required
- `description`: Input description

### Output Specifications

- `name`: Output name
- `type`: Data type
- `shape`: Tensor shape
- `labels`: Class labels (if applicable)
- `description`: Output description

### Provider Information

- `provider`: Provider name
- `available`: Whether provider is available
- `platform`: Platform/hardware type
- `description`: Provider description

---

## Related Commands

- [`gpux build`](build.md) - Build and validate models
- [`gpux run`](run.md) - Run inference
- [`gpux serve`](serve.md) - Start HTTP server

---

## See Also

- [Model Inspector API](../python-api/models.md)
- [Provider Manager API](../python-api/providers.md)
- [Configuration Schema](../configuration/schema.md)
