# NVIDIA GPUs

Optimize GPUX for NVIDIA GPUs with CUDA and TensorRT.

---

## Overview

NVIDIA GPUs provide excellent performance for ML inference with two execution providers:

- **TensorRT**: Best performance (4-10x faster than CPU)
- **CUDA**: Good performance, easier setup

---

## Supported GPUs

### GeForce Series
- RTX 40 Series (Ada Lovelace)
- RTX 30 Series (Ampere)
- RTX 20 Series (Turing)
- GTX 16 Series (Turing)
- GTX 10 Series (Pascal)

### Professional
- Quadro RTX Series
- Tesla/A-Series (A100, A10, etc.)
- H-Series (H100)

### Minimum Requirements
- CUDA Compute Capability 6.0+
- 4GB VRAM (8GB+ recommended)

---

## Installation

### 1. Install CUDA Toolkit

**Ubuntu/Debian**:
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.0
sudo apt-get install cuda-toolkit-12-0
```

**Windows**:
1. Download [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Run installer
3. Add to PATH

### 2. Install GPUX with GPU Support

```bash
# Install ONNX Runtime with GPU support
uv add onnxruntime-gpu

# Install GPUX
uv add gpux
```

### 3. Verify Installation

```bash
# Check CUDA
nvidia-smi

# Check GPUX providers
gpux inspect
```

---

## Configuration

### CUDA Provider

```yaml
name: my-model
model:
  source: ./model.onnx

runtime:
  gpu:
    backend: cuda
    memory: 4GB
  batch_size: 8

inputs:
  - name: input
    type: float32
    shape: [1, 10]

outputs:
  - name: output
    type: float32
    shape: [1, 2]
```

### TensorRT Provider

```yaml
runtime:
  gpu:
    backend: tensorrt  # Use TensorRT for best performance
    memory: 8GB
  batch_size: 16
```

---

## Performance Optimization

### 1. Use TensorRT

TensorRT provides 2-10x speedup over CUDA:

```python
from gpux import GPUXRuntime

runtime = GPUXRuntime(
    model_path="model.onnx",
    provider="tensorrt",  # Use TensorRT
    memory_limit="8GB"
)
```

### 2. Enable FP16 Precision

For RTX GPUs with Tensor Cores:

```python
# Convert model to FP16 for 2x speedup
runtime = GPUXRuntime(
    model_path="model_fp16.onnx",
    provider="tensorrt"
)
```

### 3. Optimize Batch Size

```python
# Find optimal batch size
for batch_size in [1, 4, 8, 16, 32]:
    runtime = GPUXRuntime(
        model_path="model.onnx",
        batch_size=batch_size
    )
    metrics = runtime.benchmark(input_data, num_runs=100)
    print(f"Batch {batch_size}: {metrics['throughput_fps']:.1f} FPS")
```

### 4. GPU Memory Management

```yaml
runtime:
  gpu:
    memory: 4GB  # Limit GPU memory usage
```

---

## Performance Benchmarks

### Typical Performance (RTX 3080)

| Model | Provider | Batch Size | Throughput |
|-------|----------|------------|------------|
| BERT-base | TensorRT | 32 | 2,400 FPS |
| BERT-base | CUDA | 32 | 800 FPS |
| ResNet-50 | TensorRT | 16 | 1,800 FPS |
| ResNet-50 | CUDA | 16 | 600 FPS |

### Provider Comparison

- **TensorRT**: 2-10x faster, requires optimization
- **CUDA**: Easier setup, good performance
- **CPU**: 10-50x slower baseline

---

## Troubleshooting

### CUDA Not Available

**Error**:
```
CUDAExecutionProvider not available
```

**Solutions**:
1. Install CUDA Toolkit:
   ```bash
   sudo apt-get install cuda-toolkit-12-0
   ```

2. Install onnxruntime-gpu:
   ```bash
   uv add onnxruntime-gpu
   ```

3. Verify GPU:
   ```bash
   nvidia-smi
   ```

### Out of Memory

**Error**:
```
CUDA out of memory
```

**Solutions**:
1. Reduce batch size:
   ```yaml
   runtime:
     batch_size: 4  # Reduce from 16
   ```

2. Limit GPU memory:
   ```yaml
   runtime:
     gpu:
       memory: 2GB  # Reduce limit
   ```

3. Use model quantization:
   ```python
   # Use INT8 quantized model
   runtime = GPUXRuntime("model_int8.onnx")
   ```

### TensorRT Build Errors

**Error**:
```
TensorRT engine build failed
```

**Solutions**:
1. Use CUDA instead:
   ```yaml
   runtime:
     gpu:
       backend: cuda
   ```

2. Check model compatibility
3. Update TensorRT version

---

## Best Practices

!!! tip "Use TensorRT in Production"
    TensorRT provides best performance for production:
    ```yaml
    runtime:
      gpu:
        backend: tensorrt
        memory: 8GB
    ```

!!! tip "Enable Tensor Cores"
    Use FP16 precision on RTX GPUs:
    - 2x speedup
    - Minimal accuracy loss
    - Requires model conversion

!!! tip "Monitor GPU Utilization"
    ```bash
    # Monitor in real-time
    watch -n 1 nvidia-smi

    # Check utilization
    nvidia-smi --query-gpu=utilization.gpu --format=csv
    ```

!!! warning "Driver Compatibility"
    Ensure CUDA driver matches toolkit version:
    - CUDA 12.0 requires driver 525+
    - CUDA 11.8 requires driver 520+

---

## Advanced Configuration

### Multi-GPU Setup

```python
import os

# Select specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

runtime = GPUXRuntime(
    model_path="model.onnx",
    provider="cuda"
)
```

### Profiling

```yaml
runtime:
  enable_profiling: true  # Enable CUDA profiling
```

```bash
# View profile
nsys profile gpux run model --input @data.json
```

---

## See Also

- [Execution Providers](../guide/providers.md)
- [Performance Optimization](../advanced/optimization.md)
- [Platform Comparison](index.md)
