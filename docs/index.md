# GPUX

<div class="hero" markdown>

# Docker-like GPU Runtime for ML Inference

**GPUX** provides universal GPU compatibility for ML inference workloads. Run the same model on **any GPU** without compatibility issues.

[Get Started](tutorial/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/gpux/gpux-runtime){ .md-button }

</div>

---

## ⚡ Why GPUX?

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### 🌍 Universal GPU Support

Works on NVIDIA, AMD, Apple Silicon, Intel, and Windows GPUs. No more "works on my GPU" problems.

</div>

<div class="feature-card" markdown>

### 🐳 Docker-like UX

Familiar commands and configuration. If you know Docker, you know GPUX.

```bash
gpux build .
gpux run model-name
gpux serve model-name
```

</div>

<div class="feature-card" markdown>

### ⚙️ Zero Configuration

Automatically selects the best GPU provider. Works out of the box.

</div>

<div class="feature-card" markdown>

### 🚀 High Performance

Leverages optimized ONNX Runtime backends with TensorRT, CUDA, CoreML, and more.

</div>

<div class="feature-card" markdown>

### 🔧 Production Ready

Built on mature, battle-tested technologies. Ready for production workloads.

</div>

<div class="feature-card" markdown>

### 🐍 Python First

Simple Python API for seamless integration into your ML pipelines.

</div>

</div>

---

## 🎯 Quick Example

Create a `gpux.yml` file:

```yaml
name: sentiment-analysis
version: 1.0.0

model:
  source: ./model.onnx
  format: onnx

inputs:
  text:
    type: string
    required: true

outputs:
  sentiment:
    type: float32
    shape: [1, 2]
    labels: [negative, positive]

runtime:
  gpu:
    memory: 2GB
    backend: auto  # Automatically selects best GPU
```

Run inference:

=== "CLI"

    ```bash
    # Build your model
    gpux build .

    # Run inference
    gpux run sentiment-analysis --input '{"text": "I love this!"}'

    # Start HTTP server
    gpux serve sentiment-analysis --port 8080
    ```

=== "Python"

    ```python
    from gpux import GPUXRuntime

    # Initialize runtime
    runtime = GPUXRuntime(model_path="model.onnx")

    # Run inference
    result = runtime.infer({"text": "I love this!"})
    print(result)  # {'sentiment': [0.1, 0.9]}
    ```

=== "HTTP API"

    ```bash
    # Start server
    gpux serve sentiment-analysis --port 8080

    # Make request
    curl -X POST http://localhost:8080/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "I love this!"}'
    ```

---

## 🖥️ Supported Platforms

| Platform | GPU | Provider | Status |
|----------|-----|----------|--------|
| **NVIDIA** | CUDA | TensorRT, CUDA | ✅ Supported |
| **AMD** | ROCm | ROCm | ✅ Supported |
| **Apple** | Metal | CoreML | ✅ Supported |
| **Intel** | OpenVINO | OpenVINO | ✅ Supported |
| **Windows** | DirectML | DirectML | ✅ Supported |
| **Universal** | CPU | CPU | ✅ Supported |

---

## 📦 Installation

Install GPUX using `uv` (recommended) or `pip`:

=== "uv"

    ```bash
    uv add gpux
    ```

=== "pip"

    ```bash
    pip install gpux
    ```

!!! tip "Why uv?"
    We recommend using [uv](https://github.com/astral-sh/uv) for faster, more reliable dependency management.

---

## 🚀 Key Features

### Automatic Provider Selection

GPUX automatically selects the best execution provider for your hardware:

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime(model_path="model.onnx")
# Automatically uses:
# - TensorRT/CUDA on NVIDIA GPUs
# - CoreML on Apple Silicon
# - ROCm on AMD GPUs
# - CPU as fallback
```

### Benchmarking Built-in

Measure performance with ease:

```bash
gpux run model-name --benchmark --runs 1000
```

```
╭─ Benchmark Results ─────────────────────╮
│ Mean Time     │ 0.42 ms                 │
│ Std Time      │ 0.05 ms                 │
│ Min Time      │ 0.38 ms                 │
│ Max Time      │ 0.55 ms                 │
│ Throughput    │ 2,380 fps               │
╰─────────────────────────────────────────╯
```

### HTTP Server

Serve models with a single command:

```bash
gpux serve model-name --port 8080
```

Automatic OpenAPI/Swagger documentation at `/docs`.

---

## 📚 Learn More

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### [Tutorial](tutorial/index.md)

Step-by-step guide from installation to production deployment.

</div>

<div class="feature-card" markdown>

### [User Guide](guide/index.md)

In-depth documentation of core concepts and features.

</div>

<div class="feature-card" markdown>

### [Examples](examples/index.md)

Real-world examples: sentiment analysis, image classification, LLM inference, and more.

</div>

<div class="feature-card" markdown>

### [API Reference](reference/index.md)

Complete CLI, configuration, and Python API reference.

</div>

<div class="feature-card" markdown>

### [Deployment](deployment/index.md)

Deploy to Docker, Kubernetes, AWS, GCP, Azure, and edge devices.

</div>

<div class="feature-card" markdown>

### [Advanced](advanced/index.md)

Performance optimization, custom providers, production best practices.

</div>

</div>

---

## 🌟 Show Your Support

If you find GPUX useful, please consider:

- ⭐ [Star us on GitHub](https://github.com/gpux/gpux-runtime)
- 🐛 [Report bugs or request features](https://github.com/gpux/gpux-runtime/issues)
- 💬 [Join our Discord community](https://discord.gg/gpux)
- 📢 [Share on Twitter](https://twitter.com/intent/tweet?text=Check%20out%20GPUX%20-%20Docker-like%20GPU%20runtime%20for%20ML%20inference!&url=https://github.com/gpux/gpux-runtime)

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](about/contributing.md) to get started.

---

## 📄 License

GPUX is licensed under the [MIT License](about/license.md).

---

**Ready to get started?** Check out our [Installation Guide](tutorial/installation.md)!
