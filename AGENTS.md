# AGENTS.md

## Project Overview
GPUX is a Docker-like runtime for ML inference that provides universal GPU compatibility across all platforms (NVIDIA, AMD, Apple Silicon, Intel). It solves the "works on my GPU" problem by using optimized ONNX Runtime backends with execution providers, delivering excellent performance while providing a familiar Docker-like UX for ML practitioners.

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (not pip)
- **ML Runtime**: ONNX Runtime with execution providers (CoreML, CUDA, ROCm, DirectML)
- **CLI Framework**: Click or Typer
- **Configuration**: YAML (gpux.yaml)
- **Testing**: pytest
- **Code Quality**: ruff, black, mypy

## Project Structure
```
gpux-runtime/
├── src/
│   └── gpux/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── build.py
│       │   ├── run.py
│       │   ├── serve.py
│       │   └── inspect.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── runtime.py
│       │   ├── providers.py
│       │   └── models.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── parser.py
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── tests/
├── examples/
├── docs/
├── pyproject.toml
├── README.md
└── AGENTS.md
```

## Development Rules

### Code Style
- Use **ruff** for code formatting and linting with strict rules
- Use **mypy** for type checking with strict mode
- Follow **PEP 8** and **PEP 484** for type hints
- Use **docstrings** for all public functions and classes (Google style)
- Configure ruff with `line-length = 88` and `target-version = "py311"`

### Project Management
- **Always use `uv`** for dependency management, never pip
- Use `uv sync` to install dependencies
- Use `uv add <package>` to add new dependencies
- Use `uv run <command>` to run commands in the virtual environment
- Keep `pyproject.toml` as the single source of truth for dependencies

### Architecture Principles
- **Use ONNX Runtime** with execution providers, never raw WebGPU
- **Focus on platform layer** - your value-add is UX/tooling, not ML kernels
- **Intelligent backend selection** - automatically choose best GPU provider
- **Zero configuration** - works out of the box
- **Docker-like UX** - familiar commands and patterns

### File Organization
- Use **src layout** with `src/gpux/` as the main package
- Separate CLI commands into individual modules in `cli/`
- Core functionality goes in `core/` modules
- Configuration parsing in `config/` modules
- Utilities in `utils/` modules
- All tests in `tests/` directory

### Dependencies
- **Core**: onnxruntime, click, pyyaml, numpy
- **Dev**: pytest, ruff, mypy, pytest-cov
- **Optional**: torch (for model creation examples)

### Testing
- Write **unit tests** for all core functionality
- Use **pytest** as the testing framework
- Aim for **90%+ code coverage**
- Test on **multiple platforms** when possible
- Use **fixtures** for common test data

### Error Handling
- Use **custom exceptions** for GPUX-specific errors
- Provide **clear error messages** with actionable advice
- Handle **graceful degradation** (fallback to CPU if GPU fails)
- Log **warnings** for non-critical issues

### Performance
- **Benchmark** all inference operations
- Use **warmup** before performance measurements
- Support **batch processing** for efficiency
- **Profile** memory usage and optimize

### Documentation
- Write **comprehensive docstrings** for all public APIs
- Include **usage examples** in docstrings
- Create **README.md** with quick start guide
- Document **gpux.yaml format** with examples
- Add **architecture diagrams** where helpful

## Key Implementation Notes

### Provider Selection Priority
1. TensorrtExecutionProvider (NVIDIA TensorRT)
2. CUDAExecutionProvider (NVIDIA CUDA)
3. ROCmExecutionProvider (AMD ROCm)
4. CoreMLExecutionProvider (Apple Silicon)
5. DirectMLExecutionProvider (Windows DirectML)
6. OpenVINOExecutionProvider (Intel OpenVINO)
7. CPUExecutionProvider (Universal fallback)

### gpux.yaml Format
```yaml
name: model-name
version: 1.0.0

model:
  source: ./model.onnx
  format: onnx

inputs:
  input_name:
    type: float32
    shape: [1, 10]

outputs:
  output_name:
    type: float32
    shape: [1, 2]

runtime:
  gpu:
    memory: 1GB
    backend: auto
```

### CLI Commands
- `gpux init <name>` - Initialize new project (creates gpux.yaml)
- `gpux build [path]` - Build/validate project (auto-detects gpux.yaml)
- `gpux run <name> --input <data>` - Run inference
- `gpux serve <name> --port <port>` - Start HTTP server
- `gpux inspect <name>` - Show model information
- `gpux list` - List available projects

## Validation Results
- **Performance**: 0.04ms inference on Apple Silicon (CoreML)
- **Compatibility**: Works on any GPU via execution providers
- **Architecture**: Sound and scalable platform layer approach
- **UX**: Docker-like interface validated with users

## Success Criteria
- Universal GPU compatibility (NVIDIA, AMD, Apple, Intel)
- Sub-1ms inference time on modern GPUs
- Zero-configuration deployment
- Familiar Docker-like UX
- Production-ready reliability

## Anti-Patterns to Avoid
- ❌ Don't use raw WebGPU operations (too slow, too complex)
- ❌ Don't build ML kernels from scratch (reinventing the wheel)
- ❌ Don't use pip for dependency management (use uv)
- ❌ Don't put all code in single files (use modular structure)
- ❌ Don't ignore type hints (use mypy strictly)
- ❌ Don't skip tests (aim for 90%+ coverage)

## References
- [Cursor Documentation - AGENTS.md](https://cursor.com/docs/context/rules#agentsmd)
- ONNX Runtime Documentation
- Click/Typer CLI Framework Documentation
- Python Packaging Best Practices
