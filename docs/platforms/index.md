# Platform Guides

Platform-specific optimization guides for GPUX.

---

## Overview

GPUX provides universal GPU compatibility across all major platforms. Each platform has specific optimizations and execution providers for best performance.

---

## Supported Platforms

### 🟢 NVIDIA GPUs

**Execution Providers**: TensorRT, CUDA

- Best performance with TensorRT optimization
- Wide range of GPU support (GeForce, Quadro, Tesla)
- Excellent for production deployments

[**→ NVIDIA Guide**](nvidia.md)

---

### 🔴 AMD GPUs

**Execution Provider**: ROCm

- Native AMD GPU acceleration
- Support for Radeon and Instinct series
- Linux-focused deployment

[**→ AMD Guide**](amd.md)

---

### 🍎 Apple Silicon

**Execution Provider**: CoreML

- Optimized for M1/M2/M3/M4 chips
- Neural Engine acceleration
- Excellent power efficiency

[**→ Apple Silicon Guide**](apple-silicon.md)

---

### 🔷 Intel GPUs

**Execution Provider**: OpenVINO

- Support for Intel Iris and Arc GPUs
- CPU + iGPU optimization
- Cross-platform support

[**→ Intel Guide**](intel.md)

---

### 🪟 Windows DirectML

**Execution Provider**: DirectML

- Universal Windows GPU support
- Works with any DirectX 12 compatible GPU
- Fallback for non-NVIDIA Windows systems

[**→ Windows Guide**](windows.md)

---

### 💻 CPU-Only

**Execution Provider**: CPU

- Universal fallback
- No GPU required
- Multi-threaded optimization

[**→ CPU Guide**](cpu.md)

---

## Platform Comparison

| Platform | Provider | Performance | Power Efficiency | Ease of Setup |
|----------|----------|-------------|------------------|---------------|
| NVIDIA (TensorRT) | TensorrtExecutionProvider | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| NVIDIA (CUDA) | CUDAExecutionProvider | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AMD (ROCm) | ROCmExecutionProvider | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Apple Silicon | CoreMLExecutionProvider | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Intel (OpenVINO) | OpenVINOExecutionProvider | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Windows (DirectML) | DirectMLExecutionProvider | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CPU | CPUExecutionProvider | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Quick Start by Platform

### NVIDIA GPU

```bash
# Install CUDA dependencies
uv add onnxruntime-gpu

# Configure for CUDA
runtime:
  gpu:
    backend: cuda
    memory: 4GB
```

### Apple Silicon

```bash
# CoreML is built-in with onnxruntime
uv add onnxruntime

# Configure for CoreML
runtime:
  gpu:
    backend: coreml
    memory: 2GB
```

### AMD GPU

```bash
# Install ROCm dependencies
uv add onnxruntime-rocm

# Configure for ROCm
runtime:
  gpu:
    backend: rocm
    memory: 4GB
```

### Windows

```bash
# Install DirectML
uv add onnxruntime-directml

# Configure for DirectML
runtime:
  gpu:
    backend: directml
    memory: 2GB
```

---

## Platform Detection

GPUX automatically detects your platform and selects the best provider:

```python
from gpux import GPUXRuntime

# Auto-detect best provider
runtime = GPUXRuntime(model_path="model.onnx")

# Check selected provider
info = runtime.get_provider_info()
print(f"Using: {info['name']} on {info['platform']}")
```

---

## Performance Tips by Platform

### NVIDIA

- Use **TensorRT** for best performance (up to 4x faster than CUDA)
- Enable FP16 precision for RTX GPUs
- Optimize batch size based on GPU memory

### Apple Silicon

- Use **CoreML** for Neural Engine acceleration
- Models <1GB work best on M-series chips
- Enable ANE (Apple Neural Engine) optimizations

### AMD

- Use **ROCm** with latest drivers
- Monitor GPU utilization with `rocm-smi`
- Linux recommended over Windows

### Intel

- Use **OpenVINO** for iGPU + CPU optimization
- Works well with INT8 quantized models
- Good for edge deployments

### Windows

- **DirectML** works with any DX12 GPU
- Good fallback for non-NVIDIA systems
- Supports NVIDIA, AMD, and Intel GPUs

---

## Environment Setup

### Linux (NVIDIA)

```bash
# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-0

# Install GPUX with GPU support
uv add onnxruntime-gpu
```

### macOS (Apple Silicon)

```bash
# No additional setup needed
uv add onnxruntime
```

### Linux (AMD)

```bash
# Install ROCm
sudo apt-get install rocm-dkms

# Install GPUX with ROCm
uv add onnxruntime-rocm
```

### Windows

```bash
# Install DirectML runtime
# (Usually included with Windows 10/11)

# Install GPUX with DirectML
uv add onnxruntime-directml
```

---

## Troubleshooting

### Provider Not Available

Check available providers:

```bash
gpux inspect
```

### GPU Not Detected

Verify GPU and drivers:

```bash
# NVIDIA
nvidia-smi

# AMD
rocm-smi

# Intel
clinfo
```

### Performance Issues

1. Check GPU utilization
2. Verify correct provider is selected
3. Optimize batch size
4. Consider model quantization

---

## See Also

- [Execution Providers Guide](../guide/providers.md)
- [Performance Optimization](../advanced/optimization.md)
- [Runtime Configuration](../reference/configuration/runtime.md)
