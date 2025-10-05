# Windows DirectML

Universal GPU support for Windows with DirectML.

---

## Overview

DirectML provides universal GPU support on Windows for any DirectX 12 compatible GPU.

**Execution Provider**: DirectMLExecutionProvider

---

## Supported GPUs

### All DirectX 12 Compatible GPUs
- NVIDIA GeForce/Quadro
- AMD Radeon
- Intel Iris/Arc
- Qualcomm Adreno (ARM)

### Requirements
- Windows 10 (1903+) or Windows 11
- DirectX 12 compatible GPU

---

## Installation

### 1. Verify DirectX 12

```powershell
# Check DirectX version
dxdiag
```

### 2. Install GPUX

```bash
# DirectML runtime included in Windows 10/11
uv add onnxruntime-directml
uv add gpux
```

---

## Configuration

```yaml
runtime:
  gpu:
    backend: directml
    memory: 4GB
  batch_size: 8
```

---

## Performance

### Typical Performance

| GPU | BERT FPS | ResNet-50 FPS |
|-----|----------|---------------|
| RTX 3060 | 500 | 700 |
| RX 6700 XT | 450 | 650 |
| Arc A750 | 400 | 550 |

DirectML is typically 60-80% of native performance (CUDA/ROCm).

---

## Best Practices

!!! tip "Universal Compatibility"
    DirectML works with any DX12 GPU - great fallback option

!!! tip "When to Use DirectML"
    - Non-NVIDIA Windows systems
    - Mixed GPU environments
    - When driver setup is difficult

!!! warning "Performance"
    For best performance:
    - NVIDIA: Use CUDA instead
    - AMD: Use ROCm on Linux

---

## Troubleshooting

### DirectML Not Available

1. Update Windows to latest version
2. Update GPU drivers
3. Verify DX12 support: `dxdiag`

---

## See Also

- [Platform Comparison](index.md)
- [NVIDIA Guide](nvidia.md) (better Windows performance)
