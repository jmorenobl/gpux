# Batch Inference

Process multiple inputs efficiently with batching.

---

## 🎯 Overview

Learn batch processing for higher throughput.

---

## 📦 Batch Processing

### Input Format

```json
[
  {"text": "First input"},
  {"text": "Second input"},
  {"text": "Third input"}
]
```

### Run Batch

```bash
gpux run model --file batch_input.json
```

---

## ⚡ Performance

### Batch Size Optimization

```yaml
runtime:
  batch_size: 8  # Process 8 at once
```

**Guidelines:**
- Start with `batch_size: 1`
- Increase until memory limit
- Benchmark to find optimal

### Throughput Comparison

| Batch Size | Time (ms) | Throughput (samples/sec) |
|------------|-----------|--------------------------|
| 1 | 10 | 100 |
| 8 | 35 | 229 |
| 32 | 120 | 267 |

---

## 🐍 Python API

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime("model.onnx")

# Batch inference
batch = [
    {"input": np.array([[1,2,3]])},
    {"input": np.array([[4,5,6]])},
]

results = runtime.batch_infer(batch)
```

---

## 💡 Key Takeaways

!!! success
    ✅ Batch input format
    ✅ Batch size optimization
    ✅ Performance gains
    ✅ Python API usage

---

**Previous:** [Preprocessing](preprocessing.md) | **Next:** [Python API →](python-api.md)
