# Tutorial - Introduction

Welcome to the GPUX tutorial! This guide will take you from zero to deploying production-ready ML inference workloads.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:

- ✅ Install and configure GPUX
- ✅ Convert models to ONNX format
- ✅ Create and understand `gpux.yml` configuration files
- ✅ Run inference on your models
- ✅ Benchmark model performance
- ✅ Deploy models with HTTP servers
- ✅ Optimize for your specific GPU platform

## 📋 Prerequisites

Before starting, you should have:

- **Python 3.11+** installed
- **Basic command-line knowledge**
- **A machine learning model** (or use our examples)
- *(Optional)* **A GPU** (NVIDIA, AMD, Apple Silicon, Intel, or Windows)

!!! tip "Don't have a GPU?"
    GPUX works great on CPU too! It will automatically detect available hardware and select the best provider.

## 🗺️ Tutorial Structure

This tutorial is organized into progressive steps:

### [1. Installation](installation.md)
Install GPUX and verify your setup.

**Time: 5 minutes**

### [2. First Steps](first-steps.md)
Create your first GPUX project and run inference.

**Time: 10 minutes**

### [3. Configuration](configuration.md)
Learn about `gpux.yml` and customize your models.

**Time: 15 minutes**

### [4. Running Inference](running-inference.md)
Master the `gpux run` command with different input formats.

**Time: 10 minutes**

### [5. Benchmarking](benchmarking.md)
Measure and optimize model performance.

**Time: 10 minutes**

### [6. Serving Models](serving.md)
Deploy models with HTTP APIs for production use.

**Time: 15 minutes**

---

## 💡 Learning Path

```mermaid
graph LR
    A[Installation] --> B[First Steps]
    B --> C[Configuration]
    C --> D[Running Inference]
    D --> E[Benchmarking]
    E --> F[Serving]
    F --> G[Production Ready!]

    style A fill:#6366f1,stroke:#4f46e5,color:#fff
    style B fill:#6366f1,stroke:#4f46e5,color:#fff
    style C fill:#6366f1,stroke:#4f46e5,color:#fff
    style D fill:#6366f1,stroke:#4f46e5,color:#fff
    style E fill:#6366f1,stroke:#4f46e5,color:#fff
    style F fill:#6366f1,stroke:#4f46e5,color:#fff
    style G fill:#10b981,stroke:#059669,color:#fff
```

---

## 🚀 Quick Start

If you're already familiar with Docker and ML inference, here's a quick overview:

```bash
# Install GPUX
uv add gpux

# Create a project
mkdir my-model && cd my-model

# Create gpux.yml
cat > gpux.yml << EOF
name: my-model
version: 1.0.0

model:
  source: ./model.onnx
  format: onnx

inputs:
  input:
    type: float32
    shape: [1, 10]

outputs:
  output:
    type: float32
    shape: [1, 2]

runtime:
  gpu:
    backend: auto
EOF

# Build and run
gpux build .
gpux run my-model --input '{"input": [[1,2,3,4,5,6,7,8,9,10]]}'
```

---

## 📖 Alternative Paths

Depending on your experience level, you can choose different paths:

=== "Beginner"
    **New to ML inference?**

    Follow the tutorial in order, starting with [Installation](installation.md).

    We'll explain every concept and provide detailed examples.

=== "Intermediate"
    **Familiar with ML deployment?**

    Skim through [Installation](installation.md) and [First Steps](first-steps.md), then focus on:

    - [Configuration](configuration.md)
    - [Benchmarking](benchmarking.md)
    - [Serving](serving.md)

=== "Advanced"
    **ML ops expert?**

    Jump directly to:

    - [Advanced Topics](../advanced/index.md)
    - [Deployment Guide](../deployment/index.md)
    - [Performance Optimization](../advanced/optimization.md)

---

## 🎓 After the Tutorial

Once you complete this tutorial, explore:

- **[User Guide](../guide/index.md)** - Deep dive into GPUX concepts
- **[Examples](../examples/index.md)** - Real-world use cases
- **[Deployment](../deployment/index.md)** - Production deployment strategies
- **[API Reference](../reference/index.md)** - Complete API documentation

---

## 💬 Get Help

Stuck? We're here to help!

- 📖 Check the [FAQ](../faq.md)
- 🐛 [Open an issue](https://github.com/gpux/gpux-runtime/issues)
- 💬 [Join our Discord](https://discord.gg/gpux)
- 📧 [Email support](mailto:support@gpux.io)

---

**Ready to begin?** Let's start with [Installation →](installation.md)
