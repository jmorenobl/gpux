# About GPUX

Docker-like GPU runtime for ML inference with universal GPU compatibility.

---

## What is GPUX?

GPUX is a lightweight, Docker-inspired runtime that makes GPU-accelerated ML inference work everywhere - NVIDIA, AMD, Apple Silicon, Intel, and even CPU-only systems.

### Key Features

- 🚀 **Universal GPU Support** - Works on any GPU (NVIDIA, AMD, Apple, Intel)
- 🐳 **Docker-like UX** - Familiar `build`, `run`, `serve` commands
- ⚡ **Excellent Performance** - ONNX Runtime with optimized execution providers
- 🔧 **Simple Configuration** - Single `gpux.yml` file
- 📦 **Zero Vendor Lock-in** - Use ONNX models from any framework

---

## Why GPUX?

### The Problem

ML deployment is fragmented:
- Different GPUs need different runtimes (CUDA, ROCm, CoreML, DirectML)
- Complex setup and configuration
- Vendor lock-in with frameworks

### The Solution

GPUX provides a unified interface:
```bash
# Works everywhere - same commands, any GPU
gpux build .
gpux run model-name --input '{"data": [1,2,3]}'
gpux serve model-name --port 8080
```

---

## Architecture

GPUX is a **platform layer** built on proven technologies:

```
┌─────────────────────────────────────┐
│         GPUX (Docker-like UX)       │
├─────────────────────────────────────┤
│        ONNX Runtime (Core)          │
├─────────────────────────────────────┤
│  Execution Providers (GPU Backends) │
│  TensorRT│CUDA│ROCm│CoreML│DirectML │
├─────────────────────────────────────┤
│     Hardware (Any GPU or CPU)       │
└─────────────────────────────────────┘
```

**Philosophy**: We focus on UX and tooling, leveraging ONNX Runtime's battle-tested ML execution.

---

## Technology Stack

- **Runtime**: ONNX Runtime (Microsoft)
- **Execution Providers**: TensorRT, CUDA, ROCm, CoreML, DirectML, OpenVINO, CPU
- **CLI**: Typer (Python)
- **Serving**: FastAPI + Uvicorn
- **Configuration**: YAML + Pydantic

---

## Project Status

- ✅ **Production Ready**: Built on mature ONNX Runtime
- 🚀 **Active Development**: Regular updates and improvements
- 🌟 **Open Source**: MIT License

---

## Performance

GPUX delivers excellent performance through optimized execution providers:

| Hardware | Provider | BERT Throughput | vs CPU |
|----------|----------|-----------------|--------|
| RTX 3080 | TensorRT | 2,400 FPS | 48x |
| M2 Pro | CoreML | 450 FPS | 9x |
| RX 6800 XT | ROCm | 600 FPS | 15x |

---

## Get Involved

### Use GPUX
- 📖 [Documentation](../index.md)
- 🚀 [Quick Start](../tutorial/installation.md)
- 💡 [Examples](../examples/index.md)

### Contribute
- 🐛 [Report Issues](https://github.com/gpux/gpux-runtime/issues)
- 💬 [Discussions](https://github.com/gpux/gpux-runtime/discussions)
- 🤝 [Contributing Guide](contributing.md)

### Stay Updated
- ⭐ [Star on GitHub](https://github.com/gpux/gpux-runtime)
- 📰 [Changelog](changelog.md)
- 🗺️ [Roadmap](roadmap.md)

---

## License

GPUX is open source under the [MIT License](license.md).
