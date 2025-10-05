# CPU-Only Deployment

Run GPUX on CPU without GPU acceleration.

---

## Overview

The CPU execution provider provides universal compatibility without GPU requirements.

**Execution Provider**: CPUExecutionProvider

---

## When to Use CPU

✅ **Good For**:
- Development and testing
- Systems without GPUs
- Very small models
- Batch processing (non-realtime)

❌ **Not Recommended For**:
- Real-time inference
- Large models
- High-throughput applications

---

## Configuration

```yaml
runtime:
  gpu:
    backend: cpu  # Force CPU
  batch_size: 4
  timeout: 60  # Longer timeout for CPU
```

---

## Performance

### CPU vs GPU Comparison

| Model | CPU (16-core) | GPU (RTX 3080) | Speedup |
|-------|---------------|----------------|---------|
| BERT | 50 FPS | 800 FPS | 16x |
| ResNet-50 | 80 FPS | 1,800 FPS | 22x |

### Optimization Tips

1. **Use All Cores**:
   ```yaml
   runtime:
     batch_size: 16  # Larger batch for CPU
   ```

2. **Quantization**:
   - INT8 models: 2-4x faster
   - Minimal accuracy loss

3. **Model Size**:
   - Keep models small (<100M parameters)
   - Use distilled models

---

## Installation

```bash
# CPU-only installation
uv add onnxruntime
uv add gpux
```

---

## Multi-Threading

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime(
    model_path="model.onnx",
    provider="cpu",
    inter_op_num_threads=16,  # Use all CPU cores
    intra_op_num_threads=1
)
```

---

## Best Practices

!!! tip "Quantization"
    Use INT8 models for 2-4x speedup:
    ```python
    runtime = GPUXRuntime("model_int8.onnx", provider="cpu")
    ```

!!! tip "Batch Processing"
    Process multiple items at once:
    ```yaml
    runtime:
      batch_size: 32
    ```

!!! warning "Performance Expectations"
    CPU is 10-50x slower than GPU. Plan accordingly.

---

## See Also

- [Platform Comparison](index.md)
- [Model Optimization](../guide/models.md)
- [Quantization](../advanced/optimization.md#quantization)
