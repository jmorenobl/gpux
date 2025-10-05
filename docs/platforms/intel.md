# Intel GPUs

Optimize GPUX for Intel GPUs with OpenVINO.

---

## Overview

Intel GPUs provide acceleration through OpenVINO, supporting both integrated and discrete GPUs.

**Execution Provider**: OpenVINOExecutionProvider

---

## Supported GPUs

### Integrated Graphics
- Iris Xe (11th-12th Gen Intel)
- UHD Graphics (10th Gen+)

### Discrete Graphics
- Arc A-Series (A770, A750, A380)
- Data Center GPU Flex/Max

---

## Installation

### 1. Install OpenVINO

**Linux**:
```bash
# Add Intel repository
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

# Install OpenVINO
sudo apt-get install openvino
```

**Windows**:
Download [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

### 2. Install GPUX

```bash
uv add onnxruntime-openvino
uv add gpux
```

---

## Configuration

```yaml
runtime:
  gpu:
    backend: openvino
    memory: 4GB
  batch_size: 8
```

---

## Performance

### Benchmarks (Arc A770)

| Model | Throughput |
|-------|------------|
| BERT | 400 FPS |
| ResNet-50 | 500 FPS |

---

## Best Practices

!!! tip "INT8 Quantization"
    OpenVINO excels with INT8 models:
    - 2-4x faster than FP32
    - Minimal accuracy loss

!!! tip "CPU + GPU"
    OpenVINO can use CPU + iGPU together

---

## See Also

- [Platform Comparison](index.md)
- [Optimization Guide](../advanced/optimization.md)
