# Frequently Asked Questions

Common questions about GPUX.

## General

### What is GPUX?

GPUX is a Docker-like GPU runtime for ML inference that provides universal GPU compatibility.

### Why use GPUX instead of serving frameworks like TorchServe or Triton?

GPUX focuses on simplicity and universal GPU compatibility, while TorchServe and Triton are more complex and NVIDIA-focused.

### Is GPUX production-ready?

Yes! GPUX is built on mature technologies (ONNX Runtime) and is ready for production use.

## Installation

### Which Python versions are supported?

Python 3.11 and higher.

### Do I need a GPU?

No! GPUX works on CPU-only machines with automatic fallback.

## Performance

### How fast is GPUX?

Performance depends on your hardware and model. Use `gpux run --benchmark` to measure.

### Can I use TensorRT for optimization?

Yes! GPUX automatically uses TensorRT when available on NVIDIA GPUs.

!!! info "More Questions?"
    Check our [Help](help.md) page or [open a discussion](https://github.com/gpux/gpux-runtime/discussions).
