# GPUX - Docker-like GPU Runtime for ML Inference

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

GPUX provides universal GPU compatibility for ML inference workloads, allowing you to run the same model on any GPU without compatibility issues. Think Docker, but for ML models.

## 🚀 Features

- **Universal GPU Support**: Works on NVIDIA, AMD, Apple Silicon, and Intel GPUs
- **Docker-like UX**: Familiar commands and configuration files
- **Zero Configuration**: Automatically selects the best execution provider
- **High Performance**: Optimized ONNX Runtime backends
- **Easy Deployment**: Simple HTTP server for production use
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📦 Installation

```bash
# Install with pip
pip install gpux

# Or install with uv (recommended)
uv add gpux
```

## 🎯 Quick Start

### 1. Create a gpux.yml

```yaml
# gpux.yml
name: sentiment-analysis
version: 1.0.0

model:
  source: ./model.onnx
  format: onnx

inputs:
  text:
    type: string
    max_length: 512
    required: true

outputs:
  sentiment:
    type: float32
    shape: [1, 2]
    labels: [negative, positive]

runtime:
  gpu:
    memory: 2GB
    backend: auto
```

### 2. Build your model

```bash
gpux build .
```

### 3. Run inference

```bash
# Single inference
gpux run sentiment-analysis --input '{"text": "I love this product!"}'

# From file
gpux run sentiment-analysis --file input.json

# Benchmark
gpux run sentiment-analysis --benchmark --runs 1000
```

### 4. Start a server

```bash
gpux serve sentiment-analysis --port 8080
```

## 🛠️ Commands

### Build
```bash
gpux build [PATH]                    # Build model from directory
gpux build . --provider cuda         # Build with specific GPU provider
gpux build . --no-optimize           # Build without optimization
```

### Run
```bash
gpux run MODEL_NAME                  # Run inference
gpux run MODEL_NAME --input DATA     # Run with input data
gpux run MODEL_NAME --file FILE      # Run with input file
gpux run MODEL_NAME --benchmark      # Run benchmark
```

### Serve
```bash
gpux serve MODEL_NAME                # Start HTTP server
gpux serve MODEL_NAME --port 9000    # Custom port
gpux serve MODEL_NAME --workers 4    # Multiple workers
```

### Inspect
```bash
gpux inspect MODEL_NAME              # Inspect model
gpux inspect --model model.onnx      # Inspect model file directly
gpux inspect --json                  # JSON output
```

## 🔧 Configuration

### gpux.yml Format

```yaml
name: model-name
version: 1.0.0
description: "Model description"

model:
  source: ./model.onnx
  format: onnx

inputs:
  input_name:
    type: float32
    shape: [1, 10]
    required: true
    description: "Input description"

outputs:
  output_name:
    type: float32
    shape: [1, 2]
    labels: [class1, class2]
    description: "Output description"

runtime:
  gpu:
    memory: 2GB
    backend: auto  # auto, cuda, coreml, rocm, vulkan, metal, dx12
  batch_size: 1
  timeout: 30

serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5

preprocessing:
  tokenizer: bert-base-uncased
  max_length: 512
  resize: [224, 224]
  normalize: imagenet
```

## 🎯 Supported Platforms

| Platform | GPU | Provider | Status |
|----------|-----|----------|--------|
| **NVIDIA** | CUDA | CUDAExecutionProvider | ✅ |
| **NVIDIA** | TensorRT | TensorrtExecutionProvider | ✅ |
| **AMD** | ROCm | ROCmExecutionProvider | ✅ |
| **Apple** | Metal | CoreMLExecutionProvider | ✅ |
| **Intel** | OpenVINO | OpenVINOExecutionProvider | ✅ |
| **Windows** | DirectML | DirectMLExecutionProvider | ✅ |
| **Universal** | CPU | CPUExecutionProvider | ✅ |

## 🚀 Performance

GPUX automatically selects the best execution provider for your hardware:

- **Apple Silicon**: CoreML (optimized for M1/M2/M3)
- **NVIDIA**: TensorRT > CUDA (best performance)
- **AMD**: ROCm (ROCm acceleration)
- **Intel**: OpenVINO (Intel optimization)
- **Windows**: DirectML (Windows GPU acceleration)
- **Fallback**: CPU (universal compatibility)

## 📚 Examples

Check out the [examples/](examples/) directory for complete examples:

- [Sentiment Analysis](examples/sentiment-analysis/) - BERT-based text classification
- [Image Classification](examples/image-classification/) - ResNet-50 for ImageNet

## 🔌 API Reference

### Python API

```python
from gpux import GPUXRuntime

# Initialize runtime
runtime = GPUXRuntime(model_path="model.onnx")

# Run inference
results = runtime.infer({"input": data})

# Benchmark
metrics = runtime.benchmark(data, num_runs=100)

# Get model info
info = runtime.get_model_info()
```

### HTTP API

When serving a model, GPUX provides a REST API:

```bash
# Health check
GET /health

# Model information
GET /info

# Run inference
POST /predict
Content-Type: application/json

{
  "input": "your data here"
}
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src/gpux

# Run specific test
pytest tests/test_runtime.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ONNX Runtime](https://onnxruntime.ai/) for the excellent ML runtime
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Typer](https://typer.tiangolo.com/) for the CLI framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output

## 📞 Support

- 📖 [Documentation](https://docs.gpux.io)
- 🐛 [Issues](https://github.com/gpux/gpux-runtime/issues)
- 💬 [Discussions](https://github.com/gpux/gpux-runtime/discussions)
- 📧 [Email](mailto:support@gpux.io)