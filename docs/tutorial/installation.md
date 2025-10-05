# Installation

This guide will help you install GPUX and verify your setup.

---

## 📋 Requirements

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 500MB for GPUX + space for your models

### Optional Requirements

- **GPU**: NVIDIA, AMD, Apple Silicon, Intel, or Windows GPU (for accelerated inference)
- **Docker**: For containerized deployments (optional)

!!! info "CPU-Only Support"
    GPUX works perfectly on CPU-only machines. GPU acceleration is optional but recommended for better performance.

---

## 🚀 Installation Methods

Choose your preferred installation method:

=== "uv (Recommended)"

    [uv](https://github.com/astral-sh/uv) is a fast, reliable Python package manager.

    ### Install uv

    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows (PowerShell)
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    ### Install GPUX

    ```bash
    # Add GPUX to your project
    uv add gpux

    # Or install globally
    uv pip install gpux
    ```

    ### Why uv?

    - ⚡ **10-100x faster** than pip
    - 🔒 **Deterministic** dependency resolution
    - 🎯 **Modern** Python package management
    - 🚀 **Used by GPUX** internally

=== "pip"

    Standard Python package manager.

    ### Install GPUX

    ```bash
    # Install with pip
    pip install gpux

    # Or with specific version
    pip install gpux==0.1.0

    # Upgrade to latest
    pip install --upgrade gpux
    ```

    ### Create Virtual Environment (Recommended)

    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate (macOS/Linux)
    source venv/bin/activate

    # Activate (Windows)
    venv\Scripts\activate

    # Install GPUX
    pip install gpux
    ```

=== "From Source"

    For development or latest features.

    ### Clone Repository

    ```bash
    git clone https://github.com/gpux/gpux-runtime.git
    cd gpux-runtime
    ```

    ### Install with uv

    ```bash
    # Install dependencies
    uv sync

    # Install in development mode
    uv pip install -e .
    ```

    ### Install with pip

    ```bash
    # Install dependencies
    pip install -e .

    # Or with dev dependencies
    pip install -e ".[dev]"
    ```

---

## ✅ Verify Installation

After installation, verify that GPUX is working correctly:

### Check Version

```bash
gpux --version
```

Expected output:
```
GPUX version 0.1.0
```

### Check Available Commands

```bash
gpux --help
```

Expected output:
```
╭─ Commands ───────────────────────────────────────────────╮
│ build     Build and optimize models for GPU inference.   │
│ run       Run inference on a model.                      │
│ serve     Start HTTP server for model serving.          │
│ inspect   Inspect models and runtime information.       │
╰──────────────────────────────────────────────────────────╯
```

### Verify GPU Providers

Check which GPU providers are available on your system:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Example outputs:

=== "NVIDIA GPU"

    ```
    ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    ```

=== "Apple Silicon"

    ```
    ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    ```

=== "AMD GPU"

    ```
    ['ROCmExecutionProvider', 'CPUExecutionProvider']
    ```

=== "CPU Only"

    ```
    ['CPUExecutionProvider']
    ```

!!! tip "GPU Not Detected?"
    If your GPU isn't listed, you may need to install GPU-specific drivers or ONNX Runtime packages. See [GPU Setup](#gpu-setup) below.

---

## 🖥️ GPU Setup

### NVIDIA GPUs (CUDA)

For NVIDIA GPU acceleration:

```bash
# Install CUDA-enabled ONNX Runtime
pip install onnxruntime-gpu

# Verify CUDA is available
nvidia-smi
```

**Requirements:**
- CUDA 11.8 or 12.x
- cuDNN 8.x
- NVIDIA drivers 520+

!!! info "TensorRT Support"
    For best performance, install TensorRT:
    ```bash
    pip install onnxruntime-gpu tensorrt
    ```

### AMD GPUs (ROCm)

For AMD GPU acceleration:

```bash
# Install ROCm-enabled ONNX Runtime
pip install onnxruntime-rocm

# Verify ROCm
rocm-smi
```

**Requirements:**
- ROCm 5.4+
- AMD drivers

### Apple Silicon (M1/M2/M3)

Apple Silicon support is built-in:

```bash
# Standard ONNX Runtime includes CoreML
pip install onnxruntime
```

**Requirements:**
- macOS 12.0+
- Apple Silicon Mac (M1, M2, M3, etc.)

### Intel GPUs (OpenVINO)

For Intel GPU acceleration:

```bash
# Install OpenVINO-enabled ONNX Runtime
pip install onnxruntime-openvino
```

**Requirements:**
- Intel GPU drivers
- OpenVINO toolkit

### Windows GPUs (DirectML)

DirectML support is built-in on Windows:

```bash
# Standard ONNX Runtime includes DirectML
pip install onnxruntime-directml
```

**Requirements:**
- Windows 10/11
- DirectX 12 compatible GPU

---

## 📦 Optional Dependencies

Install optional features based on your needs:

### ML Framework Support

For model conversion from PyTorch, TensorFlow, etc.:

```bash
# PyTorch support
uv add --group ml torch torchvision

# TensorFlow support
uv add --group ml tensorflow

# Transformers support (BERT, GPT, etc.)
uv add --group ml transformers
```

### HTTP Server

For serving models via HTTP:

```bash
# FastAPI + Uvicorn
uv add --group serve fastapi uvicorn
```

### Development Tools

For contributing or development:

```bash
# Install dev dependencies
uv sync --group dev

# Includes: pytest, ruff, mypy, pre-commit
```

---

## 🧪 Test Your Installation

Let's run a quick test to ensure everything works:

### Create Test Script

Create a file named `test_gpux.py`:

```python
"""Test GPUX installation."""
from gpux.utils.helpers import check_dependencies, get_gpu_info

# Check dependencies
print("Checking dependencies...")
deps = check_dependencies()
for name, available in deps.items():
    status = "✅" if available else "❌"
    print(f"{status} {name}")

# Check GPU info
print("\nChecking GPU...")
gpu_info = get_gpu_info()
if gpu_info["available"]:
    print(f"✅ GPU Available: {gpu_info.get('provider', 'Unknown')}")
else:
    print("⚠️  No GPU detected (CPU only)")

print("\n✅ GPUX is ready to use!")
```

### Run Test

```bash
python test_gpux.py
```

Expected output:
```
Checking dependencies...
✅ onnxruntime
✅ onnx
✅ numpy
✅ yaml
✅ click
✅ typer
✅ rich
✅ pydantic

Checking GPU...
✅ GPU Available: CoreMLExecutionProvider

✅ GPUX is ready to use!
```

---

## 🐛 Troubleshooting

### Command Not Found

If `gpux` command is not found:

```bash
# Check if GPUX is installed
pip list | grep gpux

# Reinstall
pip install --force-reinstall gpux
```

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Verify Python version
python --version  # Should be 3.11+

# Reinstall dependencies
pip install --upgrade gpux
```

### GPU Not Detected

If your GPU isn't detected:

1. **Verify drivers are installed**
   ```bash
   # NVIDIA
   nvidia-smi

   # AMD
   rocm-smi
   ```

2. **Install GPU-specific ONNX Runtime**
   ```bash
   # NVIDIA
   pip install onnxruntime-gpu

   # AMD
   pip install onnxruntime-rocm
   ```

3. **Check provider availability**
   ```bash
   python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
   ```

### Permission Errors

If you encounter permission errors:

```bash
# Use user install (no sudo required)
pip install --user gpux

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install gpux
```

---

## 📚 Next Steps

Now that GPUX is installed, let's create your first model!

**Continue to:** [First Steps →](first-steps.md)

---

## 🆘 Still Having Issues?

- 📖 Check the [FAQ](../faq.md)
- 🐛 [Report installation issues](https://github.com/gpux/gpux-runtime/issues/new?template=installation.md)
- 💬 [Ask on Discord](https://discord.gg/gpux)
- 📧 [Email support](mailto:support@gpux.io)
