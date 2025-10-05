# AMD GPUs

Optimize GPUX for AMD GPUs with ROCm.

---

## Overview

AMD GPUs provide native GPU acceleration through the ROCm execution provider.

**Execution Provider**: ROCmExecutionProvider

---

## Supported GPUs

### Radeon Series
- RX 7000 Series (RDNA 3)
- RX 6000 Series (RDNA 2)
- RX 5000 Series (RDNA)

### Instinct Series
- MI300 Series
- MI200 Series
- MI100 Series

### Minimum Requirements
- ROCm 5.0+
- 4GB VRAM

---

## Installation

### 1. Install ROCm (Linux Only)

**Ubuntu 22.04**:
```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key
sudo apt-key add rocm.gpg.key

echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 ubuntu main' | \
  sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt-get update

# Install ROCm
sudo apt-get install rocm-dkms
```

### 2. Install GPUX with ROCm

```bash
# Install ONNX Runtime with ROCm
uv add onnxruntime-rocm

# Install GPUX
uv add gpux
```

### 3. Verify Installation

```bash
# Check ROCm
rocm-smi

# Check GPUX providers
gpux inspect
```

---

## Configuration

```yaml
name: my-model
model:
  source: ./model.onnx

runtime:
  gpu:
    backend: rocm
    memory: 8GB
  batch_size: 16

inputs:
  - name: input
    type: float32

outputs:
  - name: output
    type: float32
```

---

## Performance

### Benchmarks (RX 6800 XT)

| Model | Throughput | vs CPU |
|-------|------------|--------|
| BERT-base | 600 FPS | 15x |
| ResNet-50 | 800 FPS | 20x |
| GPT-2 | 300 FPS | 12x |

---

## Troubleshooting

### ROCm Not Detected

```bash
# Verify ROCm installation
rocm-smi

# Check kernel module
lsmod | grep amdgpu

# Reinstall if needed
sudo apt-get install --reinstall rocm-dkms
```

### Performance Issues

1. Update ROCm drivers
2. Check GPU utilization: `rocm-smi`
3. Optimize batch size

---

## Best Practices

!!! tip "Use Latest ROCm"
    Update to latest ROCm for best performance:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade rocm-dkms
    ```

!!! tip "Monitor GPU"
    ```bash
    watch -n 1 rocm-smi
    ```

!!! warning "Linux Only"
    ROCm is primarily supported on Linux. Windows support is experimental.

---

## See Also

- [Platform Comparison](index.md)
- [Performance Optimization](../advanced/optimization.md)
