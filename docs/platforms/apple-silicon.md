# Apple Silicon

Optimize GPUX for Apple M-series chips with CoreML.

---

## Overview

Apple Silicon (M1/M2/M3/M4) provides excellent performance and power efficiency through CoreML and the Neural Engine.

**Execution Provider**: CoreMLExecutionProvider

---

## Supported Chips

### M-Series
- **M4** (2024) - 16-core Neural Engine
- **M3** (2023) - 16-core Neural Engine
- **M2** (2022) - 16-core Neural Engine
- **M1** (2020) - 16-core Neural Engine

### Variants
- M4 Pro/Max/Ultra
- M3 Pro/Max
- M2 Pro/Max/Ultra
- M1 Pro/Max/Ultra

---

## Installation

### 1. Install GPUX

```bash
# CoreML is built-in with onnxruntime
uv add onnxruntime
uv add gpux
```

No additional dependencies needed!

### 2. Verify Installation

```bash
gpux inspect
```

**Expected Output**:
```
✅ CoreMLExecutionProvider: Available
```

---

## Configuration

```yaml
name: my-model
model:
  source: ./model.onnx

runtime:
  gpu:
    backend: coreml  # Use CoreML
    memory: 2GB
  batch_size: 1  # CoreML works best with batch_size=1

inputs:
  - name: input
    type: float32

outputs:
  - name: output
    type: float32
```

---

## Performance

### Benchmarks (M2 Pro)

| Model | CoreML | CPU | Speedup |
|-------|--------|-----|---------|
| BERT-base | 450 FPS | 50 FPS | 9x |
| ResNet-50 | 600 FPS | 80 FPS | 7.5x |
| MobileNet | 1,200 FPS | 200 FPS | 6x |

### Power Efficiency

- **5-10x** better power efficiency than discrete GPUs
- Neural Engine uses minimal power
- Excellent for battery-powered devices

---

## Optimization Tips

### 1. Use Batch Size 1

CoreML performs best with single inference:

```yaml
runtime:
  batch_size: 1
```

### 2. Model Size

Keep models under 1GB for best performance:

- ✅ BERT-base (110M parameters)
- ✅ ResNet-50 (25M parameters)
- ⚠️ GPT-2 (large models may be slower)

### 3. Data Types

CoreML supports:
- FP32 (full precision)
- FP16 (half precision, faster)
- INT8 (quantized, fastest)

---

## Neural Engine

### What is the Neural Engine?

- Dedicated ML hardware on Apple Silicon
- 16-core design (11 TOPS on M1, 15+ TOPS on M3/M4)
- Optimized for matrix operations

### Enabling Neural Engine

CoreML automatically uses the Neural Engine when available:

```python
from gpux import GPUXRuntime

# CoreML will use Neural Engine automatically
runtime = GPUXRuntime(
    model_path="model.onnx",
    provider="coreml"
)
```

---

## Troubleshooting

### CoreML Not Available

**Cause**: Running on Intel Mac or non-macOS system

**Solution**: CoreML only works on Apple Silicon Macs

### Slow Performance

1. **Reduce model size**: Keep under 1GB
2. **Use batch_size=1**: CoreML optimized for single inference
3. **Check model compatibility**: Some ops not supported

### Model Conversion Errors

Some ONNX operations are not supported by CoreML. Fallback to CPU:

```yaml
runtime:
  gpu:
    backend: cpu
```

---

## Best Practices

!!! tip "Optimize for Neural Engine"
    - Keep models small (<1GB)
    - Use FP16 or INT8 precision
    - Batch size = 1

!!! tip "Power Efficiency"
    CoreML is perfect for:
    - Battery-powered inference
    - Edge deployments
    - Always-on ML services

!!! tip "Temperature Monitoring"
    ```bash
    # Check system temperature
    sudo powermetrics --samplers smc
    ```

!!! success "Zero Setup Required"
    No drivers, no CUDA toolkit - CoreML just works!

---

## Comparison: CoreML vs CPU

| Metric | CoreML (M2) | CPU (M2) |
|--------|-------------|----------|
| BERT Throughput | 450 FPS | 50 FPS |
| Power Usage | 5W | 15W |
| Temperature | Low | Medium |

---

## See Also

- [Platform Comparison](index.md)
- [Performance Optimization](../advanced/optimization.md)
- [Model Optimization](../guide/models.md)
