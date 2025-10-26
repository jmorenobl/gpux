# Docker Deployment

Deploy GPUX models using Docker containers with universal GPU compatibility.

---

## 🎯 Quick Start

### Basic Dockerfile

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

### Build and Run

```bash
# Build image
docker build -t my-gpux-model .

# Run container
docker run -p 8080:8080 my-gpux-model

# Test
curl http://localhost:8080/health
```

### Model Registry Integration ✅ **Available**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install GPUX
RUN pip install gpux

# Pull model from Hugging Face Hub
RUN gpux pull microsoft/DialoGPT-medium

# Expose port
EXPOSE 8080

# Start server
CMD ["gpux", "serve", "microsoft/DialoGPT-medium", "--port", "8080"]
```

---

## 🔧 Advanced Configuration

### Multi-stage Build

```dockerfile
# Stage 1: Build
FROM python:3.11 AS builder
WORKDIR /app
RUN pip install gpux --target /app/deps

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/deps /usr/local/lib/python3.11/site-packages/
COPY model.onnx gpux.yml ./

EXPOSE 8080
CMD ["gpux", "serve", "model-name", "--port", "8080"]
```

### GPU Support (NVIDIA)

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install gpux onnxruntime-gpu

COPY model.onnx gpux.yml ./

CMD ["gpux", "serve", "model-name", "--port", "8080"]
```

Run with GPU:
```bash
docker run --gpus all -p 8080:8080 my-model
```

### Universal GPU Support ✅ **Available**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install GPUX with all execution providers
RUN pip install gpux onnxruntime-gpu

# Copy model and config
COPY model.onnx gpux.yml ./

# Expose port
EXPOSE 8080

# Start server (auto-detects best GPU)
CMD ["gpux", "serve", "model-name", "--port", "8080"]
```

GPUX automatically detects and uses the best available GPU:
- **NVIDIA**: CUDA or TensorRT execution providers
- **AMD**: ROCm execution provider
- **Apple Silicon**: CoreML execution provider
- **Intel**: OpenVINO execution provider
- **CPU**: Fallback to CPU execution provider

---

## 📦 Docker Compose

### Basic Setup

```yaml
version: '3.8'

services:
  gpux-model:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GPUX_LOG_LEVEL=info
    volumes:
      - ./models:/app/models
```

### GPU Support

```yaml
version: '3.8'

services:
  gpux-model:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GPUX_LOG_LEVEL=info
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Multi-Model Setup

```yaml
version: '3.8'

services:
  sentiment-model:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_NAME=sentiment-analysis
    command: ["gpux", "serve", "sentiment-analysis", "--port", "8080"]

  image-model:
    build: .
    ports:
      - "8081:8080"
    environment:
      - MODEL_NAME=image-classifier
    command: ["gpux", "serve", "image-classifier", "--port", "8080"]
```

---

## 🚀 Production Deployment

### Health Checks

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install gpux
COPY model.onnx gpux.yml ./

EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["gpux", "serve", "model-name", "--port", "8080"]
```

### Resource Limits

```yaml
version: '3.8'

services:
  gpux-model:
    build: .
    ports:
      - "8080:8080"
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 💡 Key Takeaways

!!! success
    ✅ **Universal GPU Support**: Works on NVIDIA, AMD, Apple Silicon, Intel GPUs
    ✅ **Model Registry Integration**: Pull models directly from Hugging Face Hub
    ✅ **Automatic Optimization**: ONNX Runtime optimization for best performance
    ✅ **Docker Compose**: Easy multi-service deployment
    ✅ **Health Checks**: Built-in health monitoring
    ✅ **Resource Management**: Proper resource limits and reservations

---

## 🔧 Troubleshooting

### Common Issues

**GPU Not Detected:**
```bash
# Check GPU availability
docker run --gpus all --rm nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Model Loading Issues:**
```bash
# Check model format
gpux inspect model-name --verbose
```

**Memory Issues:**
```bash
# Monitor memory usage
docker stats
```

---

**Previous:** [Deployment Index](index.md) | **Next:** [Kubernetes →](kubernetes.md)
