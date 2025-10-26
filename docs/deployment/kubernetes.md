# Kubernetes Deployment

Deploy GPUX models on Kubernetes with universal GPU compatibility.

---

## 🎯 Overview

Complete guide for deploying GPUX on Kubernetes with automatic scaling and GPU resource management.

!!! info "In Development"
    Detailed deployment guide for Kubernetes is being developed. Basic functionality is available.

---

## 📚 What Will Be Covered

- ✅ **Basic Deployment**: Simple Kubernetes deployment
- 🔄 **GPU Resource Management**: NVIDIA, AMD, Apple Silicon GPU support
- 🔄 **Auto-scaling**: Horizontal Pod Autoscaler (HPA) configuration
- 🔄 **Service Mesh**: Istio integration for traffic management
- 🔄 **Monitoring**: Prometheus and Grafana integration
- 🔄 **Cost Optimization**: Resource limits and node affinity
- 🔄 **Best Practices**: Production-ready configurations

---

## 🚀 Quick Start

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpux-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpux-model
  template:
    metadata:
      labels:
        app: gpux-model
    spec:
      containers:
      - name: gpux
        image: my-gpux-model:latest
        ports:
        - containerPort: 8080
        env:
        - name: GPUX_LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: gpux-model-service
spec:
  selector:
    app: gpux-model
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### GPU Support (NVIDIA)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpux-model-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpux-model-gpu
  template:
    metadata:
      labels:
        app: gpux-model-gpu
    spec:
      containers:
      - name: gpux
        image: my-gpux-model:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
```

---

## 🔧 Advanced Configuration

### Model Registry Integration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpux-huggingface-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpux-huggingface-model
  template:
    metadata:
      labels:
        app: gpux-huggingface-model
    spec:
      initContainers:
      - name: pull-model
        image: my-gpux-model:latest
        command: ["gpux", "pull", "microsoft/DialoGPT-medium"]
        volumeMounts:
        - name: model-cache
          mountPath: /root/.gpux
      containers:
      - name: gpux
        image: my-gpux-model:latest
        command: ["gpux", "serve", "microsoft/DialoGPT-medium", "--port", "8080"]
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: model-cache
          mountPath: /root/.gpux
      volumes:
      - name: model-cache
        emptyDir: {}
```

### Auto-scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpux-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpux-model
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 💡 Key Takeaways

!!! success
    ✅ **Universal GPU Support**: Works with NVIDIA, AMD, Apple Silicon GPUs
    ✅ **Model Registry Integration**: Pull models from Hugging Face Hub
    ✅ **Auto-scaling**: Horizontal Pod Autoscaler support
    ✅ **Resource Management**: Proper GPU resource allocation
    ✅ **Service Discovery**: Kubernetes service integration

---

**Previous:** [Docker →](docker.md) | **Next:** [AWS →](aws.md)
