# Docker Deployment

Deploy GPUX models using Docker containers.

---

## 🎯 Quick Start

### Dockerfile

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

---

## 📦 Docker Compose

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

---

## 💡 Key Takeaways

!!! success
    ✅ Docker provides consistent deployment
    ✅ Multi-stage builds reduce image size
    ✅ GPU support with NVIDIA runtime
    ✅ Docker Compose for multi-service

---

**Previous:** [Deployment Index](index.md) | **Next:** [Kubernetes →](kubernetes.md)
