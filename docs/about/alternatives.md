# GPUX vs Alternatives

How GPUX compares to other ML serving frameworks.

---

## Comparison Table

| Feature | GPUX | TorchServe | Triton | TFServing | ONNX Runtime |
|---------|------|------------|--------|-----------|--------------|
| **GPU Support** | Universal | NVIDIA-focused | NVIDIA-focused | Limited | Backend-dependent |
| **Setup Complexity** | Simple | Medium | Complex | Medium | Simple |
| **Docker-like UX** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **ONNX Support** | Native | Via conversion | Yes | No | Native |
| **Multi-framework** | ✅ (via ONNX) | PyTorch only | Multi | TensorFlow only | ✅ |
| **Apple Silicon** | ✅ CoreML | ❌ | ❌ | ❌ | Limited |
| **AMD GPU** | ✅ ROCm | Limited | Limited | Limited | Limited |
| **Config File** | YAML | Complex | Complex | Protobuf | Code |

---

## vs TorchServe

### TorchServe
- **Pros**: Deep PyTorch integration, good for PyTorch-only shops
- **Cons**: NVIDIA-focused, complex setup, no universal GPU support

### GPUX
- **Pros**: Universal GPU support, simple config, Docker-like UX
- **Cons**: Requires ONNX conversion for PyTorch models

**Use GPUX if**: You need universal GPU support or simpler deployment
**Use TorchServe if**: You're PyTorch-only and NVIDIA-only

---

## vs NVIDIA Triton

### Triton
- **Pros**: High performance, enterprise features, multi-framework
- **Cons**: Complex setup, NVIDIA-focused, steep learning curve

### GPUX
- **Pros**: Simple setup, universal GPU support, easier to get started
- **Cons**: Fewer enterprise features (no ensemble models, pipelines)

**Use GPUX if**: You want simplicity and universal GPU support
**Use Triton if**: You need advanced enterprise features and NVIDIA-only

---

## vs TensorFlow Serving

### TensorFlow Serving
- **Pros**: Optimized for TensorFlow, mature, production-ready
- **Cons**: TensorFlow-only, limited GPU support, complex config

### GPUX
- **Pros**: Multi-framework (via ONNX), universal GPU, simple config
- **Cons**: Requires model conversion from TensorFlow

**Use GPUX if**: You want universal GPU support or use multiple frameworks
**Use TF Serving if**: You're TensorFlow-only

---

## vs Raw ONNX Runtime

### ONNX Runtime
- **Pros**: Fast, flexible, library-level control
- **Cons**: Requires code for serving, no built-in CLI/HTTP

### GPUX
- **Pros**: Built-in CLI and HTTP serving, Docker-like UX, config-based
- **Cons**: Less flexibility than raw library usage

**Use GPUX if**: You want a complete runtime with serving
**Use ONNX Runtime if**: You need library-level control in your application

---

## Key Differentiators

### 1. Universal GPU Support

**GPUX**: Works on any GPU (NVIDIA, AMD, Apple, Intel)
**Others**: Mostly NVIDIA-focused

### 2. Docker-like UX

**GPUX**:
```bash
gpux build .
gpux run model-name
gpux serve model-name
```

**Others**: Complex configuration files or Python code

### 3. Single Config File

**GPUX**: One `gpux.yml` file
**Others**: Multiple config files, protobuf definitions, or code

### 4. Zero Vendor Lock-in

**GPUX**: ONNX standard, any framework → ONNX → GPUX
**Others**: Framework-specific (PyTorch, TensorFlow)

---

## Use Case Recommendations

### Choose GPUX for:
- ✅ Universal GPU compatibility
- ✅ Simple deployment
- ✅ Multi-framework projects
- ✅ Apple Silicon, AMD, or Intel GPUs
- ✅ Quick prototyping

### Choose Triton for:
- ✅ Advanced enterprise features
- ✅ Model ensembles and pipelines
- ✅ NVIDIA-only deployment
- ✅ Complex serving scenarios

### Choose TorchServe for:
- ✅ PyTorch-only projects
- ✅ Deep PyTorch integration
- ✅ NVIDIA GPU deployment

### Choose TF Serving for:
- ✅ TensorFlow-only projects
- ✅ Google Cloud deployment
- ✅ Mature TensorFlow ecosystem

---

## Migration Guides

### From TorchServe

1. Export PyTorch model to ONNX
2. Create `gpux.yml` configuration
3. Run with GPUX

### From Triton

1. Convert models to ONNX (if needed)
2. Simplify configuration to `gpux.yml`
3. Use GPUX CLI commands

### From Raw ONNX Runtime

1. Add `gpux.yml` configuration
2. Remove custom serving code
3. Use `gpux serve`

---

## Performance Comparison

| Framework | BERT (NVIDIA RTX 3080) | Overhead |
|-----------|------------------------|----------|
| **GPUX (TensorRT)** | 2,400 FPS | Minimal |
| **Triton (TensorRT)** | 2,500 FPS | Minimal |
| **TorchServe** | 800 FPS | Medium |
| **Raw ONNX RT** | 2,400 FPS | None |

GPUX provides near-optimal performance with minimal overhead.

---

## Conclusion

**GPUX** focuses on:
- Simplicity over complexity
- Universal compatibility over vendor lock-in
- Developer experience over enterprise features

For most ML deployment needs, GPUX provides the best balance of simplicity, performance, and compatibility.

---

## See Also

- [About GPUX](index.md)
- [Benchmarks](benchmarks.md)
- [Architecture](architecture.md)
