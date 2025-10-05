# GPUXRuntime

Main runtime class for ML inference with universal GPU compatibility.

---

## Overview

`GPUXRuntime` is the core class for loading ONNX models and running inference with automatic GPU provider selection.

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime(model_path="model.onnx")
result = runtime.infer({"input": data})
```

---

## Class: `GPUXRuntime`

### Constructor

```python
GPUXRuntime(
    model_path: str | Path | None = None,
    provider: str | None = None,
    **kwargs
)
```

**Parameters**:

- `model_path` (`str | Path`, optional): Path to ONNX model file
- `provider` (`str`, optional): Preferred execution provider (`cuda`, `coreml`, `rocm`, etc.)
- `**kwargs`: Additional runtime configuration
  - `memory_limit` (`str`): GPU memory limit (default: `"2GB"`)
  - `batch_size` (`int`): Batch size (default: `1`)
  - `timeout` (`int`): Timeout in seconds (default: `30`)
  - `enable_profiling` (`bool`): Enable profiling (default: `False`)

**Example**:
```python
runtime = GPUXRuntime(
    model_path="sentiment.onnx",
    provider="cuda",
    memory_limit="4GB",
    batch_size=8
)
```

---

## Methods

### `load_model()`

Load an ONNX model for inference.

```python
load_model(
    model_path: str | Path,
    provider: str | None = None
) -> None
```

**Parameters**:

- `model_path` (`str | Path`): Path to ONNX model file
- `provider` (`str`, optional): Preferred execution provider

**Raises**:

- `FileNotFoundError`: If model file doesn't exist
- `RuntimeError`: If model cannot be loaded

**Example**:
```python
runtime = GPUXRuntime()
runtime.load_model("model.onnx", provider="cuda")
```

### `infer()`

Run inference on input data.

```python
infer(inputs: dict[str, Any]) -> dict[str, np.ndarray]
```

**Parameters**:

- `inputs` (`dict`): Input data dictionary mapping input names to NumPy arrays or values

**Returns**:

- `dict[str, np.ndarray]`: Output dictionary mapping output names to NumPy arrays

**Raises**:

- `ValueError`: If model not loaded or invalid inputs
- `RuntimeError`: If inference fails

**Example**:
```python
result = runtime.infer({
    "input_ids": np.array([[1, 2, 3]]),
    "attention_mask": np.array([[1, 1, 1]])
})
print(result["logits"])
```

### `benchmark()`

Run performance benchmark.

```python
benchmark(
    inputs: dict[str, Any],
    num_runs: int = 100,
    warmup_runs: int = 10
) -> dict[str, float]
```

**Parameters**:

- `inputs` (`dict`): Input data for benchmarking
- `num_runs` (`int`): Number of benchmark iterations (default: `100`)
- `warmup_runs` (`int`): Number of warmup iterations (default: `10`)

**Returns**:

- `dict[str, float]`: Performance metrics
  - `mean_time`: Mean inference time (ms)
  - `min_time`: Minimum time (ms)
  - `max_time`: Maximum time (ms)
  - `std_dev`: Standard deviation (ms)
  - `throughput_fps`: Throughput in FPS

**Example**:
```python
metrics = runtime.benchmark(
    inputs={"input": np.random.rand(1, 10).astype(np.float32)},
    num_runs=1000,
    warmup_runs=50
)
print(f"Mean time: {metrics['mean_time']:.2f} ms")
print(f"Throughput: {metrics['throughput_fps']:.1f} FPS")
```

### `get_model_info()`

Get model information.

```python
get_model_info() -> ModelInfo | None
```

**Returns**:

- `ModelInfo | None`: Model information or None if not loaded

**Example**:
```python
info = runtime.get_model_info()
print(f"Model: {info.name} v{info.version}")
print(f"Size: {info.size_mb:.1f} MB")
```

### `get_provider_info()`

Get selected provider information.

```python
get_provider_info() -> dict[str, Any]
```

**Returns**:

- `dict`: Provider information

**Example**:
```python
provider = runtime.get_provider_info()
print(f"Provider: {provider['name']}")
print(f"Platform: {provider['platform']}")
```

### `get_available_providers()`

Get list of available providers.

```python
get_available_providers() -> list[str]
```

**Returns**:

- `list[str]`: List of available provider names

**Example**:
```python
providers = runtime.get_available_providers()
print(f"Available: {', '.join(providers)}")
```

### `cleanup()`

Clean up resources.

```python
cleanup() -> None
```

**Example**:
```python
runtime.cleanup()
```

---

## Complete Example

```python
import numpy as np
from gpux import GPUXRuntime

# Initialize runtime
runtime = GPUXRuntime(
    model_path="sentiment.onnx",
    provider="cuda",
    memory_limit="2GB"
)

# Get model info
info = runtime.get_model_info()
print(f"Model: {info.name}")
print(f"Inputs: {[inp.name for inp in info.inputs]}")

# Run inference
result = runtime.infer({
    "input_ids": np.array([[101, 2054, 2003, ...]]),
    "attention_mask": np.array([[1, 1, 1, ...]])
})
print(f"Sentiment: {result['logits']}")

# Benchmark
metrics = runtime.benchmark(
    inputs={"input_ids": np.array([[101, 2054]]), "attention_mask": np.array([[1, 1]])},
    num_runs=1000
)
print(f"Mean time: {metrics['mean_time']:.2f} ms")

# Cleanup
runtime.cleanup()
```

---

## See Also

- [ModelInspector](models.md)
- [ProviderManager](providers.md)
- [Configuration](config.md)
