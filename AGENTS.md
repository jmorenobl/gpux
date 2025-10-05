# AGENTS.md

This file provides guidance to AI coding assistants (Claude Code, Cursor, GitHub Copilot) when working with code in this repository.

## Project Overview

GPUX is a Docker-like runtime for ML inference that provides universal GPU compatibility across all platforms (NVIDIA, AMD, Apple Silicon, Intel). It solves the "works on my GPU" problem by using optimized ONNX Runtime backends with execution providers, delivering excellent performance while providing a familiar Docker-like UX for ML practitioners.

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (not pip)
- **ML Runtime**: ONNX Runtime with execution providers (CoreML, CUDA, ROCm, DirectML)
- **CLI Framework**: Typer
- **Configuration**: YAML (gpux.yml)
- **Testing**: pytest
- **Code Quality**: ruff, mypy

## Project Structure

```
gpux-runtime/
├── src/
│   └── gpux/
│       ├── __init__.py
│       ├── cli/              # Command-line interface (Typer-based)
│       │   ├── __init__.py
│       │   ├── main.py       # CLI entry point
│       │   ├── build.py      # gpux build command
│       │   ├── run.py        # gpux run command
│       │   ├── serve.py      # gpux serve command (FastAPI server)
│       │   └── inspect.py    # gpux inspect command
│       ├── core/             # Core runtime components
│       │   ├── __init__.py
│       │   ├── runtime.py    # GPUXRuntime - main inference engine
│       │   ├── providers.py  # ExecutionProvider management and selection
│       │   └── models.py     # ModelInspector for ONNX model introspection
│       ├── config/           # Configuration handling
│       │   ├── __init__.py
│       │   └── parser.py     # gpux.yml parser and validator
│       └── utils/            # Utility functions
│           ├── __init__.py
│           └── helpers.py    # System info, file ops, dependency checking
├── tests/
├── examples/
├── docs/
├── pyproject.toml
├── README.md
└── AGENTS.md
```

## Development Commands

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run tests without coverage (faster)
uv run pytest --no-cov

# Run specific test file
uv run pytest tests/test_runtime.py

# Run specific test function
uv run pytest tests/test_runtime.py::test_function_name

# Run with specific markers
uv run pytest -m integration
uv run pytest -m "not slow"
```

### Code Quality
```bash
# Lint code
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Run all checks (lint + type-check + test)
make check
```

### Package Management
```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update dependencies
uv lock --upgrade
uv sync --dev
```

### Pre-commit Hooks
The project uses pre-commit hooks that run on every commit:
- Ruff linting and formatting
- mypy type checking
- pytest test suite (minimum 30% coverage required)

Install hooks with: `uv run pre-commit install`

## Architecture

### Key Components

**GPUXRuntime** (`core/runtime.py`): The main runtime class that:
- Loads ONNX models and creates ONNX Runtime inference sessions
- Auto-selects the best execution provider (GPU/CPU)
- Executes inference with input/output handling
- Provides benchmarking and performance metrics

**ProviderManager** (`core/providers.py`): Manages execution provider selection:
- Priority order: TensorRT > CUDA > ROCm > CoreML > DirectML > OpenVINO > CPU
- Automatic fallback to CPU if GPU providers unavailable
- Platform-specific provider availability checking

**ModelInspector** (`core/models.py`): Inspects ONNX models to extract:
- Input/output names, shapes, and types
- Model size and metadata
- ONNX opset version

**Configuration** (`config/parser.py`): Parses and validates `gpux.yml` files defining:
- Model source and format
- Input/output specifications
- Runtime settings (GPU memory, backend selection)
- Serving configuration (port, host, batch size)

### Data Flow

1. User runs CLI command (e.g., `gpux run model-name`)
2. Configuration loaded from `gpux.yml` via `ConfigParser`
3. `GPUXRuntime` initialized with model path and config
4. `ProviderManager` selects best available execution provider
5. ONNX Runtime session created with selected provider
6. Inference executed, results formatted and returned

### Provider Selection Priority

1. **TensorrtExecutionProvider** (NVIDIA TensorRT - best performance)
2. **CUDAExecutionProvider** (NVIDIA CUDA)
3. **ROCmExecutionProvider** (AMD ROCm)
4. **CoreMLExecutionProvider** (Apple Silicon - M1/M2/M3 optimized)
5. **DirectMLExecutionProvider** (Windows DirectML)
6. **OpenVINOExecutionProvider** (Intel OpenVINO)
7. **CPUExecutionProvider** (Universal fallback)

Platform-specific auto-selection:
- **NVIDIA GPUs**: TensorRT (best) → CUDA
- **AMD GPUs**: ROCm
- **Apple Silicon**: CoreML
- **Intel GPUs**: OpenVINO
- **Windows GPUs**: DirectML
- **Universal fallback**: CPU

### Configuration File Format

The `gpux.yml` file is the single source of truth for model configuration:

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
    required: true

outputs:
  output_name:
    type: float32
    shape: [1, 2]
    labels: [class1, class2]

runtime:
  gpu:
    memory: 2GB
    backend: auto  # auto, cuda, coreml, rocm, vulkan, metal, dx12
  batch_size: 1
  timeout: 30

serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5
```

### CLI Commands

- `gpux build [path]` - Build/validate project (auto-detects gpux.yml)
- `gpux run <name> --input <data>` - Run inference
- `gpux serve <name> --port <port>` - Start HTTP server
- `gpux inspect <name>` - Show model information

## Development Rules

### Code Style

- **Line length**: 88 characters (Black/Ruff standard)
- **Python version**: 3.11+ (uses modern type hints)
- Use **ruff** for code formatting and linting with strict rules
- Use **mypy** for type checking
- Follow **PEP 8** and **PEP 484** for type hints
- Use **docstrings** for all public functions and classes (Google style)
- **Type hints**: Required in function signatures (mypy enforced)
- **Imports**: Auto-sorted by Ruff

### Important Conventions

1. **Provider naming**: Use `ExecutionProvider` enum values (e.g., `ExecutionProvider.CUDA`)
2. **Error handling**: Raise specific exceptions (`ValueError`, `FileNotFoundError`, `RuntimeError`)
3. **Logging**: Use Python's `logging` module with Rich handler for CLI output
4. **CLI output**: Use Typer's `typer.echo()` for user-facing messages
5. **Configuration**: All model config goes in `gpux.yml`, not command-line args

### Project Management

- **Always use `uv`** for dependency management, never pip
- Use `uv sync` to install dependencies
- Use `uv add <package>` to add new dependencies
- Use `uv run <command>` to run commands in the virtual environment
- Keep `pyproject.toml` as the single source of truth for dependencies

### Dependencies

- **Core**: onnxruntime, typer, pyyaml, numpy, rich, pydantic
- **Dev**: pytest, ruff, mypy, pytest-cov, pre-commit
- **Optional**: torch (for model creation examples), fastapi/uvicorn (for serving)

### Git & Version Control

**Use Conventional Commits** for all commit messages following the format:
- `type(scope): description` (e.g., `feat(cli): add new run command`)
- **Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`, `build`
- **Scope**: component being changed (e.g., `cli`, `core`, `config`, `tests`)
- **Description**: clear, concise description of the change

**Examples**:
- `feat(runtime): add GPU memory management`
- `fix(config): resolve YAML parsing edge case`
- `docs(readme): update installation instructions`
- `refactor(providers): simplify provider selection logic`
- `test(runtime): add integration tests for inference`
- `chore(deps): update ruff to latest version`

### File Organization

- Use **src layout** with `src/gpux/` as the main package
- Separate CLI commands into individual modules in `cli/`
- Core functionality goes in `core/` modules
- Configuration parsing in `config/` modules
- Utilities in `utils/` modules
- All tests in `tests/` directory

### Testing Strategy

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test CLI commands end-to-end
- **Coverage target**: Maintain >30% coverage (enforced by pre-commit), aim for 90%+
- **Test fixtures**: Shared in `tests/conftest.py` (temp directories, sample models)
- **Mocking**: Use pytest mocking for ONNX Runtime and file I/O
- Use **pytest** as the testing framework
- Test on **multiple platforms** when possible

When writing tests:
- Mock ONNX Runtime (`onnxruntime.InferenceSession`) to avoid requiring actual models
- Use `tmp_path` fixture for file operations
- Test both success and error paths
- Verify CLI output and exit codes for CLI tests

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
- Document **gpux.yml format** with examples
- Add **architecture diagrams** where helpful

## Architecture Principles

- **Use ONNX Runtime** with execution providers, never raw WebGPU
- **Focus on platform layer** - your value-add is UX/tooling, not ML kernels
- **Intelligent backend selection** - automatically choose best GPU provider
- **Zero configuration** - works out of the box
- **Docker-like UX** - familiar commands and patterns

## Common Development Tasks

### Adding a new CLI command

1. Create command function in `src/gpux/cli/your_command.py`
2. Add command to app in `src/gpux/cli/main.py`
3. Add tests in `tests/test_cli_your_command.py`

### Adding a new execution provider

1. Add enum value to `ExecutionProvider` in `core/providers.py`
2. Update `ProviderManager.get_available_providers()`
3. Update provider priority in `ProviderManager.get_best_provider()`
4. Add tests for the new provider

### Modifying configuration schema

1. Update `gpux.yml` example in `README.md`
2. Update parser in `config/parser.py`
3. Add validation tests in `tests/test_config.py`

## Development Workflow

### Phase-Based Development

- **Plan First**: Always create a detailed plan for complex tasks before starting implementation
- **Phase Confirmation**: After completing each phase, wait for explicit user confirmation before proceeding
- **Commit After Phase**: Commit code changes after each completed phase with conventional commit messages
- **One Phase at a Time**: Focus on completing one phase fully before starting the next

### Quality Gates

- **Test Before Commit**: Ensure all tests pass before committing any phase
- **Lint Before Commit**: Run ruff and mypy checks before committing
- **Documentation Updates**: Update relevant documentation as part of each phase
- **Incremental Progress**: Each phase should deliver working, testable functionality

### Emergency Fixes

- **Critical Issues**: Address failing tests and broken functionality immediately
- **Dependency Issues**: Resolve import errors and dependency conflicts as priority
- **Type Safety**: Fix mypy errors before proceeding with new features

## Anti-Patterns to Avoid

- ❌ Don't use raw WebGPU operations (too slow, too complex)
- ❌ Don't build ML kernels from scratch (reinventing the wheel)
- ❌ Don't use pip for dependency management (use uv)
- ❌ Don't put all code in single files (use modular structure)
- ❌ Don't ignore type hints (use mypy strictly)
- ❌ Don't skip tests (aim for 90%+ coverage)

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

## References

- [Cursor Documentation - AGENTS.md](https://cursor.com/docs/context/rules#agentsmd)
- [Claude Code Documentation](https://docs.claude.com/en/docs/claude-code)
- ONNX Runtime Documentation
- Typer CLI Framework Documentation
- Python Packaging Best Practices
