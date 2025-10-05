# Python API

Using GPUX programmatically in your applications.

---

## 🎯 Overview

Complete guide to the GPUX Python API.

---

## 🚀 Quick Start

```python
from gpux import GPUXRuntime

# Initialize
runtime = GPUXRuntime(model_path="model.onnx")

# Inference
result = runtime.infer({"input": data})

# Cleanup
runtime.cleanup()
```

---

## 🏗️ GPUXRuntime

### Initialization

```python
runtime = GPUXRuntime(
    model_path="model.onnx",
    provider="auto",  # or "cuda", "coreml", etc.
    memory_limit="2GB",
    batch_size=1,
    timeout=30
)
```

### Methods

#### `infer(input_data)`

Run inference:

```python
result = runtime.infer({
    "input": np.array([[1, 2, 3]])
})
```

#### `batch_infer(batch_data)`

Batch processing:

```python
results = runtime.batch_infer([
    {"input": data1},
    {"input": data2}
])
```

#### `benchmark(input_data, num_runs, warmup_runs)`

Performance testing:

```python
metrics = runtime.benchmark(
    {"input": data},
    num_runs=1000,
    warmup_runs=100
)
```

#### `get_model_info()`

Model information:

```python
info = runtime.get_model_info()
print(info.name, info.version)
```

### Context Manager

```python
with GPUXRuntime("model.onnx") as runtime:
    result = runtime.infer(data)
```

---

## 🔧 Configuration

### From Python

```python
from gpux.config.parser import GPUXConfigParser

parser = GPUXConfigParser()
config = parser.parse_file("gpux.yml")

runtime = GPUXRuntime(
    model_path=parser.get_model_path("."),
    **config.runtime.dict()
)
```

---

## 🧪 Testing

```python
import pytest
from gpux import GPUXRuntime

def test_inference():
    runtime = GPUXRuntime("model.onnx")
    result = runtime.infer({"input": test_data})
    assert result is not None
    runtime.cleanup()
```

---

## 💡 Key Takeaways

!!! success
    ✅ GPUXRuntime initialization
    ✅ Inference methods
    ✅ Batch processing
    ✅ Benchmarking
    ✅ Context managers
    ✅ Configuration

---

**Previous:** [Batch Inference](batch-inference.md) | **Next:** [Error Handling →](error-handling.md)
