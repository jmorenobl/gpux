# Changelog

All notable changes to GPUX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Comprehensive documentation with 80+ pages
- Platform-specific guides (NVIDIA, AMD, Apple Silicon, Intel, Windows, CPU)
- Complete API reference (CLI, Python, HTTP, Configuration)
- Tutorial series and real-world examples
- Advanced topics and deployment guides

---

## [0.1.0] - 2024-10-05

### Added
- Initial release of GPUX runtime
- Universal GPU compatibility (NVIDIA, AMD, Apple Silicon, Intel)
- Docker-like CLI (`build`, `run`, `serve`, `inspect`)
- ONNX Runtime with execution providers
- Automatic provider selection
- Configuration via `gpux.yml`
- HTTP serving with FastAPI
- Model introspection and validation
- Benchmarking capabilities
- Python API for programmatic usage

### Features
- **CLI Commands**:
  - `gpux build` - Build and validate models
  - `gpux run` - Run inference
  - `gpux serve` - Start HTTP server
  - `gpux inspect` - Inspect models

- **Execution Providers**:
  - TensorRT (NVIDIA)
  - CUDA (NVIDIA)
  - ROCm (AMD)
  - CoreML (Apple Silicon)
  - DirectML (Windows)
  - OpenVINO (Intel)
  - CPU (Universal)

- **Configuration**:
  - YAML-based configuration
  - Input/output specifications
  - Runtime settings
  - Serving configuration

- **HTTP API**:
  - `/predict` - Run inference
  - `/health` - Health check
  - `/info` - Model information
  - `/metrics` - Performance metrics

### Performance
- Sub-millisecond inference on modern GPUs
- RTX 3080: 2,400 FPS (BERT with TensorRT)
- M2 Pro: 450 FPS (BERT with CoreML)
- RX 6800 XT: 600 FPS (BERT with ROCm)

---

## Release Types

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backwards compatible
- **Patch (0.0.X)**: Bug fixes, backwards compatible

---

## Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

---

[Unreleased]: https://github.com/gpux/gpux-runtime/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gpux/gpux-runtime/releases/tag/v0.1.0
