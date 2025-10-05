# Deployment Guide

Deploy GPUX to production environments.

---

## 🎯 Overview

Complete guides for deploying GPUX in various environments.

---

## 📖 Deployment Options

### [Docker](docker.md)
Containerized deployment with Docker.

**Best for:** Consistent environments, cloud deployment

### [Kubernetes](kubernetes.md)
Orchestrated deployment at scale.

**Best for:** High availability, auto-scaling

### [AWS](aws.md)
Deploy on Amazon Web Services.

**Best for:** AWS-native applications

### [Google Cloud](gcp.md)
Deploy on Google Cloud Platform.

**Best for:** GCP-native applications

### [Azure](azure.md)
Deploy on Microsoft Azure.

**Best for:** Azure-native applications

### [Edge Devices](edge.md)
Deploy on edge devices (Jetson, Raspberry Pi).

**Best for:** Edge inference, IoT

### [Serverless](serverless.md)
Serverless deployment patterns.

**Best for:** Event-driven, pay-per-use

---

## 🚀 Quick Start

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install gpux
EXPOSE 8080

CMD ["gpux", "serve", "model-name", "--port", "8080"]
```

```bash
docker build -t my-model .
docker run -p 8080:8080 my-model
```

---

## 💡 Choosing a Deployment Method

| Method | Complexity | Scalability | Cost | Best For |
|--------|------------|-------------|------|----------|
| Docker | Low | Medium | Low | Getting started |
| Kubernetes | High | High | Medium | Enterprise |
| AWS/GCP/Azure | Medium | High | Variable | Cloud-native |
| Edge | Medium | Low | Low | IoT/Edge |
| Serverless | Low | High | Pay-per-use | Event-driven |

---

**Prerequisites:** Complete [Tutorial](../tutorial/index.md) and [Production Best Practices](../advanced/production.md).
