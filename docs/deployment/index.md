# Deployment Guide

Deploy GPUX to production environments with universal GPU compatibility.

---

## 🎯 Overview

Complete guides for deploying GPUX in various environments. GPUX provides universal GPU compatibility across NVIDIA, AMD, Apple Silicon, and Intel GPUs through optimized ONNX Runtime execution providers.

---

## 📖 Deployment Options

### [Docker](docker.md) ✅ **Ready**
Containerized deployment with Docker.

**Best for:** Consistent environments, cloud deployment, development

### [Kubernetes](kubernetes.md) 🔄 **In Development**
Orchestrated deployment at scale.

**Best for:** High availability, auto-scaling, enterprise

### [AWS](aws.md) 🔄 **In Development**
Deploy on Amazon Web Services.

**Best for:** AWS-native applications, EC2 GPU instances

### [Google Cloud](gcp.md) 🔄 **In Development**
Deploy on Google Cloud Platform.

**Best for:** GCP-native applications, Cloud GPU

### [Azure](azure.md) 🔄 **In Development**
Deploy on Microsoft Azure.

**Best for:** Azure-native applications, Azure GPU VMs

### [Edge Devices](edge.md) 🔄 **In Development**
Deploy on edge devices (Jetson, Raspberry Pi).

**Best for:** Edge inference, IoT, embedded systems

### [Serverless](serverless.md) 🔄 **In Development**
Serverless deployment patterns.

**Best for:** Event-driven, pay-per-use, auto-scaling

---

## 🚀 Quick Start

### Docker (Recommended) ✅ **Available**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install GPUX
RUN pip install gpux

# Copy model and config
COPY model.onnx .
COPY gpux.yml .

# Expose port
EXPOSE 8080

# Start server
CMD ["gpux", "serve", "model-name", "--port", "8080"]
```

```bash
# Build image
docker build -t my-gpux-model .

# Run container
docker run -p 8080:8080 my-gpux-model

# Test
curl http://localhost:8080/health
```

### Model Registry Integration ✅ **Available**

```bash
# Pull model from Hugging Face Hub
gpux pull microsoft/DialoGPT-medium

# Run inference
gpux run microsoft/DialoGPT-medium --input '{"text": "Hello world"}'

# Start HTTP server
gpux serve microsoft/DialoGPT-medium --port 8080
```

---

## 💡 Choosing a Deployment Method

| Method | Status | Complexity | Scalability | Cost | Best For |
|--------|--------|------------|-------------|------|----------|
| Docker | ✅ Ready | Low | Medium | Low | Getting started, development |
| Kubernetes | 🔄 Dev | High | High | Medium | Enterprise, production |
| AWS/GCP/Azure | 🔄 Dev | Medium | High | Variable | Cloud-native, GPU instances |
| Edge | 🔄 Dev | Medium | Low | Low | IoT/Edge, embedded |
| Serverless | 🔄 Dev | Low | High | Pay-per-use | Event-driven, auto-scaling |

---

## 🎯 Current Capabilities

### ✅ **Available Now**
- **Universal GPU Support**: NVIDIA CUDA, AMD ROCm, Apple CoreML, Intel OpenVINO
- **Model Registry Integration**: Hugging Face Hub with `gpux pull` command
- **ONNX Conversion**: PyTorch to ONNX with automatic optimization
- **Docker Deployment**: Complete containerized deployment
- **CLI Interface**: Docker-like commands (`build`, `run`, `serve`, `pull`, `inspect`)
- **HTTP API**: RESTful API for model serving
- **Configuration**: YAML-based model configuration (`gpux.yml`)

### 🔄 **In Development**
- **Additional Registries**: ONNX Model Zoo, TensorFlow Hub, PyTorch Hub
- **TensorFlow Conversion**: TensorFlow to ONNX conversion pipeline
- **Cloud Deployment**: AWS, GCP, Azure specific guides
- **Kubernetes**: Orchestration and scaling
- **Edge Deployment**: ARM-based devices and embedded systems
- **Serverless**: Function-as-a-Service deployment

---

## 🏗️ Architecture

GPUX uses a strategy pattern architecture with execution providers:

1. **ModelManager Interface**: Abstract base for registry integrations
2. **Execution Providers**: ONNX Runtime providers for different GPUs
3. **Conversion Pipeline**: Automatic model format conversion
4. **Configuration System**: YAML-based model configuration
5. **CLI Interface**: Docker-like user experience

---

**Prerequisites:** Complete [Tutorial](../tutorial/index.md) and [Production Best Practices](../advanced/production.md).
