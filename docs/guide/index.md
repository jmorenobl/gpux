# User Guide

In-depth documentation of GPUX core concepts and features.

---

## 📚 What's in This Guide?

This guide covers core GPUX concepts in depth. Each section builds on previous knowledge from the [Tutorial](../tutorial/index.md).

---

## 📖 Guide Sections

### [Working with Models](models.md)
Learn about ONNX models, conversion from popular frameworks, optimization techniques, and model management.

**Topics:**
- Understanding ONNX format
- Converting from PyTorch, TensorFlow, scikit-learn, Hugging Face
- Model inspection and visualization
- Optimization (quantization, compression)
- Versioning and debugging

---

### [GPU Providers](providers.md)
Deep dive into execution providers and hardware-specific optimization.

**Topics:**
- What execution providers are
- Available providers (TensorRT, CUDA, CoreML, ROCm, etc.)
- Provider selection logic
- Platform-specific setup and configuration
- Performance comparison and troubleshooting

---

### [Inputs & Outputs](inputs-outputs.md)
Master input/output handling, data types, and validation.

**Topics:**
- Input data types and shapes
- JSON format requirements
- Output handling and labels
- Automatic validation
- Type conversion

---

### [Data Preprocessing](preprocessing.md)
Preprocessing pipelines for text, images, and audio data.

**Topics:**
- Text tokenization
- Image preprocessing (resize, normalize)
- Audio resampling
- Configuration options
- Custom preprocessing

---

### [Batch Inference](batch-inference.md)
Process multiple inputs efficiently for higher throughput.

**Topics:**
- Batch input format
- Batch size optimization
- Performance gains
- Memory considerations
- Python API for batching

---

### [Python API](python-api.md)
Complete reference for using GPUX programmatically.

**Topics:**
- GPUXRuntime class
- Inference methods
- Batch processing
- Benchmarking
- Context managers
- Testing

---

### [Error Handling](error-handling.md)
Common errors, solutions, and debugging techniques.

**Topics:**
- Common error messages
- Debugging methods
- Exception handling
- Validation techniques
- Troubleshooting guide

---

## 🎯 How to Use This Guide

### For Beginners

Start with the [Tutorial](../tutorial/index.md) first, then return here for deeper understanding.

### For Experienced Users

Jump directly to relevant sections based on your needs.

### For Reference

Use this guide as a reference when working on specific features.

---

## 🔗 Related Resources

- **[Tutorial](../tutorial/index.md)** - Step-by-step getting started
- **[Examples](../examples/index.md)** - Real-world use cases
- **[API Reference](../reference/index.md)** - Complete API docs
- **[Advanced Topics](../advanced/index.md)** - Performance optimization

---

## 💬 Need Help?

- 📖 [FAQ](../faq.md)
- 🆘 [Help](../help.md)
- 💬 [Discord Community](https://discord.gg/gpux)
- 🐛 [Report Issues](https://github.com/gpux/gpux-runtime/issues)
