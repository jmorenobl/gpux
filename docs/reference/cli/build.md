# `gpux build`

Build and optimize models for GPU inference.

---

## Overview

The `gpux build` command validates your GPUX configuration, inspects your model, checks provider compatibility, and prepares everything for optimal GPU inference. It creates build artifacts in the `.gpux` directory.

```bash
gpux build [PATH] [OPTIONS]
```

---

## Arguments

### `PATH`

Path to the GPUX project directory.

- **Type**: `string`
- **Default**: `.` (current directory)
- **Required**: No

**Examples**:
```bash
gpux build                    # Build from current directory
gpux build ./my-model        # Build from specific directory
gpux build ../sentiment      # Build from parent directory
```

---

## Options

### `--config`, `-c`

Configuration file name.

- **Type**: `string`
- **Default**: `gpux.yml`

```bash
gpux build --config custom.yml
gpux build -c model-config.yml
```

### `--provider`, `-p`

Preferred execution provider (cuda, coreml, rocm, etc.).

- **Type**: `string`
- **Default**: Auto-detected
- **Choices**: `cuda`, `coreml`, `rocm`, `directml`, `openvino`, `tensorrt`, `cpu`

```bash
gpux build --provider cuda
gpux build -p coreml
```

### `--optimize` / `--no-optimize`

Enable or disable model optimization.

- **Type**: `boolean`
- **Default**: `true`

```bash
gpux build --optimize         # Enable optimization (default)
gpux build --no-optimize     # Disable optimization
```

### `--verbose`

Enable verbose output for debugging.

- **Type**: `boolean`
- **Default**: `false`

```bash
gpux build --verbose
```

---

## Build Process

The build command performs the following steps:

1. **Parse Configuration**: Reads and validates `gpux.yml`
2. **Validate Model Path**: Checks if the model file exists
3. **Inspect Model**: Extracts input/output specifications, metadata
4. **Check Provider Compatibility**: Determines the best execution provider
5. **Optimize Model**: (Optional) Applies model optimizations
6. **Save Build Artifacts**: Stores model info and provider info in `.gpux/`

---

## Build Artifacts

The build process creates the following files in `.gpux/`:

```
.gpux/
├── model_info.json       # Model specifications (inputs, outputs, size)
└── provider_info.json    # Provider information (platform, availability)
```

### `model_info.json`

```json
{
  "name": "sentiment-analysis",
  "version": "1.0.0",
  "format": "onnx",
  "size_bytes": 268435456,
  "inputs": [
    {
      "name": "input_ids",
      "type": "int64",
      "shape": [1, 128],
      "required": true
    }
  ],
  "outputs": [
    {
      "name": "logits",
      "type": "float32",
      "shape": [1, 2]
    }
  ]
}
```

### `provider_info.json`

```json
{
  "provider": "CUDAExecutionProvider",
  "platform": "NVIDIA CUDA",
  "available": true,
  "description": "NVIDIA CUDA GPU acceleration"
}
```

---

## Examples

### Basic Build

Build from the current directory:

```bash
gpux build
```

**Output**:
```
✅ Build completed successfully!
Build artifacts saved to: .gpux
```

### Build with Specific Provider

Build with CUDA provider:

```bash
gpux build --provider cuda
```

### Build from Another Directory

Build a model in a different directory:

```bash
gpux build ./models/sentiment-analysis
```

### Build Without Optimization

Skip model optimization:

```bash
gpux build --no-optimize
```

### Verbose Build

Show detailed build information:

```bash
gpux build --verbose
```

---

## Output

The build command displays the following information:

### Model Information

| Property | Value |
|----------|-------|
| Name | sentiment-analysis |
| Version | 1.0.0 |
| Format | onnx |
| Size | 256.0 MB |
| Inputs | 2 |
| Outputs | 1 |

### Execution Provider

| Property | Value |
|----------|-------|
| Provider | CUDAExecutionProvider |
| Platform | NVIDIA CUDA |
| Available | ✅ Yes |
| Description | NVIDIA CUDA GPU acceleration |

### Input Specifications

| Name | Type | Shape | Required |
|------|------|-------|----------|
| input_ids | int64 | [1, 128] | ✅ |
| attention_mask | int64 | [1, 128] | ✅ |

### Output Specifications

| Name | Type | Shape |
|------|------|-------|
| logits | float32 | [1, 2] |

---

## Error Handling

### Configuration File Not Found

```bash
Error: Configuration file not found: ./gpux.yml
```

**Solution**: Ensure `gpux.yml` exists in the project directory.

### Model File Not Found

```bash
Error: Model file not found
```

**Solution**: Check the `model.source` path in `gpux.yml`.

### Invalid Configuration

```bash
Build failed: Invalid configuration: missing required field 'name'
```

**Solution**: Validate your `gpux.yml` against the schema.

---

## Best Practices

!!! tip "Always Build Before Deploying"
    Run `gpux build` before deploying to production to catch configuration errors early.

!!! tip "Use Specific Providers in Production"
    Specify the provider explicitly in production to ensure consistent behavior:
    ```bash
    gpux build --provider cuda
    ```

!!! tip "Enable Optimization"
    Keep optimization enabled (default) for better inference performance.

!!! warning "Check Provider Availability"
    The build command will show provider availability. Ensure your target provider is available before deploying.

---

## Related Commands

- [`gpux run`](run.md) - Run inference on models
- [`gpux serve`](serve.md) - Start HTTP server
- [`gpux inspect`](inspect.md) - Inspect model details

---

## See Also

- [Configuration Reference](../configuration/schema.md)
- [Execution Providers](../../guide/providers.md)
- [Model Optimization](../../advanced/optimization.md)
