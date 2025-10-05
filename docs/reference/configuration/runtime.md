# Runtime Configuration

Runtime settings for GPU, timeout, and batch processing.

---

## Overview

The `runtime` section controls execution settings.

```yaml
runtime:
  gpu:
    memory: string        # GPU memory limit (default: "2GB")
    backend: string       # GPU backend (default: "auto")
  timeout: int            # Timeout in seconds (default: 30)
  batch_size: int         # Batch size (default: 1)
  enable_profiling: bool  # Enable profiling (default: false)
```

---

## GPU Configuration

### `gpu.memory`

GPU memory limit.

- **Type**: `string`
- **Required**: No
- **Default**: `2GB`
- **Format**: Number + Unit (`GB`, `MB`, `KB`)

```yaml
runtime:
  gpu:
    memory: 2GB    # 2 gigabytes
    memory: 512MB  # 512 megabytes
    memory: 1024KB # 1024 kilobytes
```

### `gpu.backend`

Preferred GPU backend.

- **Type**: `string`
- **Required**: No
- **Default**: `auto`
- **Values**: `auto`, `cuda`, `coreml`, `rocm`, `directml`, `openvino`, `tensorrt`

```yaml
runtime:
  gpu:
    backend: auto      # Auto-detect best provider
    backend: cuda      # Force CUDA
    backend: coreml    # Force CoreML (Apple Silicon)
```

---

## Execution Settings

### `timeout`

Inference timeout in seconds.

- **Type**: `integer`
- **Required**: No
- **Default**: `30`

```yaml
runtime:
  timeout: 30   # 30 second timeout
  timeout: 60   # 1 minute timeout
```

### `batch_size`

Default batch size for inference.

- **Type**: `integer`
- **Required**: No
- **Default**: `1`

```yaml
runtime:
  batch_size: 1   # Single inference
  batch_size: 32  # Batch of 32
```

### `enable_profiling`

Enable performance profiling.

- **Type**: `boolean`
- **Required**: No
- **Default**: `false`

```yaml
runtime:
  enable_profiling: true
```

---

## Examples

### Minimal

```yaml
runtime:
  gpu:
    memory: 2GB
```

### CUDA Configuration

```yaml
runtime:
  gpu:
    memory: 4GB
    backend: cuda
  timeout: 60
  batch_size: 8
```

### CoreML (Apple Silicon)

```yaml
runtime:
  gpu:
    memory: 1GB
    backend: coreml
  timeout: 30
  batch_size: 1
```

### CPU-Only

```yaml
runtime:
  gpu:
    backend: cpu
  timeout: 120
  batch_size: 4
```

---

## Platform-Specific Examples

### NVIDIA GPU

```yaml
runtime:
  gpu:
    memory: 8GB
    backend: cuda  # or tensorrt for optimization
  batch_size: 16
```

### AMD GPU

```yaml
runtime:
  gpu:
    memory: 4GB
    backend: rocm
  batch_size: 8
```

### Apple Silicon

```yaml
runtime:
  gpu:
    memory: 2GB
    backend: coreml
  batch_size: 1
```

### Windows DirectML

```yaml
runtime:
  gpu:
    memory: 4GB
    backend: directml
  batch_size: 8
```

---

## Best Practices

!!! tip "Set Appropriate Memory Limits"
    Set GPU memory based on model size:
    - Small models (<100MB): `1GB`
    - Medium models (100-500MB): `2GB`
    - Large models (>500MB): `4GB+`

!!! tip "Use Auto Backend in Development"
    Let GPUX choose the best provider:
    ```yaml
    runtime:
      gpu:
        backend: auto
    ```

!!! tip "Specify Backend in Production"
    Use explicit backend in production:
    ```yaml
    runtime:
      gpu:
        backend: cuda  # Known hardware
    ```

!!! warning "Adjust Timeout for Large Models"
    Increase timeout for large models:
    ```yaml
    runtime:
      timeout: 120  # 2 minutes for LLMs
    ```

---

## See Also

- [Configuration Schema](schema.md)
- [Execution Providers](../../guide/providers.md)
- [Performance Optimization](../../advanced/optimization.md)
