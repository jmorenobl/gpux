# Performance Optimization

Optimize models for maximum throughput and minimal latency.

---

## 🎯 Optimization Strategies

### 1. Model Optimization

**Quantization:**
```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_quant.onnx",
    weight_type=QuantType.QUInt8
)
```

**Benefits:**
- 4x smaller models
- 2-4x faster inference
- Minimal accuracy loss

### 2. Batch Size Tuning

```yaml
runtime:
  batch_size: 32  # Optimize for throughput
```

**Guidelines:**
| Batch Size | Use Case | Latency | Throughput |
|------------|----------|---------|------------|
| 1 | Real-time | Low | Low |
| 8-16 | Balanced | Medium | Medium |
| 32-64 | Batch processing | High | High |

### 3. Provider Selection

**Performance ranking:**
1. TensorRT (NVIDIA) - Best
2. CUDA (NVIDIA)
3. CoreML (Apple Silicon)
4. ROCm (AMD)
5. CPU - Fallback

### 4. Memory Optimization

```yaml
runtime:
  gpu:
    memory: 4GB  # Allocate sufficient memory
```

### 5. Graph Optimization

Enable ONNX Runtime optimizations:
```yaml
runtime:
  enable_profiling: true
```

---

## 📊 Benchmarking

Compare optimizations:

```bash
# Baseline
gpux run model --benchmark --runs 1000

# Optimized
gpux run model_optimized --benchmark --runs 1000
```

---

## 💡 Key Takeaways

!!! success
    ✅ Quantization reduces size and improves speed
    ✅ Batch size affects latency/throughput tradeoff
    ✅ Provider selection critical for performance
    ✅ Memory allocation impacts stability
    ✅ Always benchmark before/after

---

**Previous:** [Custom Providers](custom-providers.md) | **Next:** [Memory Management →](memory-management.md)
