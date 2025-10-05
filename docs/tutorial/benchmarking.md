# Benchmarking

Measure and optimize model performance with built-in benchmarking tools.

---

## 🎯 What You'll Learn

- ✅ Running performance benchmarks
- ✅ Understanding metrics
- ✅ Comparing providers
- ✅ Optimization strategies

---

## 🚀 Quick Benchmark

Run a quick benchmark:

```bash
gpux run model-name --benchmark --runs 100
```

Output:
```
╭─ Benchmark Results ─────────────────────╮
│ Mean Time     │ 2.45 ms                 │
│ Std Time      │ 0.12 ms                 │
│ Min Time      │ 2.30 ms                 │
│ Max Time      │ 2.85 ms                 │
│ Median Time   │ 2.43 ms                 │
│ P95 Time      │ 2.68 ms                 │
│ P99 Time      │ 2.78 ms                 │
│ Throughput    │ 408.2 fps               │
╰─────────────────────────────────────────╯
```

---

## 📊 Benchmark Options

### Number of Runs

```bash
# Quick test (10 runs)
gpux run model --benchmark --runs 10

# Standard (100 runs)
gpux run model --benchmark --runs 100

# Thorough (1000 runs)
gpux run model --benchmark --runs 1000
```

### Warmup Runs

Allow model to warm up before measuring:

```bash
gpux run model --benchmark --runs 1000 --warmup 50
```

---

## 📈 Understanding Metrics

| Metric | Description |
|--------|-------------|
| **Mean Time** | Average inference time |
| **Std Time** | Standard deviation (consistency) |
| **Min Time** | Fastest inference |
| **Max Time** | Slowest inference |
| **Median Time** | Middle value (50th percentile) |
| **P95 Time** | 95th percentile (tail latency) |
| **P99 Time** | 99th percentile (worst case) |
| **Throughput** | Inferences per second (fps) |

---

## 🔄 Comparing Providers

Test different providers:

```bash
# Benchmark with auto-selected provider
gpux run model --benchmark --runs 1000

# Benchmark with CUDA
gpux build . --provider cuda
gpux run model --benchmark --runs 1000

# Benchmark with CPU
gpux build . --provider cpu
gpux run model --benchmark --runs 1000
```

---

## 🐍 Python API Benchmarking

```python
from gpux import GPUXRuntime
import numpy as np

runtime = GPUXRuntime(model_path="model.onnx")

# Prepare test data
test_data = {"input": np.random.rand(1, 10).astype(np.float32)}

# Run benchmark
metrics = runtime.benchmark(
    input_data=test_data,
    num_runs=1000,
    warmup_runs=100
)

print(f"Mean time: {metrics['mean_time_ms']:.2f} ms")
print(f"Throughput: {metrics['throughput_fps']:.1f} fps")
```

---

## 📊 Saving Results

Save benchmark results to file:

```bash
gpux run model --benchmark --runs 1000 --output benchmark.json
```

Results in JSON:
```json
{
  "mean_time_ms": 2.45,
  "std_time_ms": 0.12,
  "min_time_ms": 2.30,
  "max_time_ms": 2.85,
  "median_time_ms": 2.43,
  "p95_time_ms": 2.68,
  "p99_time_ms": 2.78,
  "throughput_fps": 408.2
}
```

---

## 🚀 Optimization Tips

### 1. Use GPU Acceleration

```yaml
runtime:
  gpu:
    backend: auto  # Let GPUX choose best provider
```

### 2. Optimize Batch Size

```yaml
runtime:
  batch_size: 8  # Process 8 samples at once
```

Benchmark different batch sizes to find optimal value.

### 3. Enable TensorRT (NVIDIA)

For NVIDIA GPUs, TensorRT provides best performance:

```bash
gpux build . --provider tensorrt
```

### 4. Profile Your Model

Enable profiling to identify bottlenecks:

```yaml
runtime:
  enable_profiling: true
```

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ How to run benchmarks
    ✅ Understanding performance metrics
    ✅ Comparing different providers
    ✅ Optimization strategies
    ✅ Saving and analyzing results

---

**Previous:** [Running Inference](running-inference.md) | **Next:** [Serving](serving.md)
