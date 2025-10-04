# Contributing to GPUX

Thank you for your interest in contributing to GPUX! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/gpux-runtime.git
   cd gpux-runtime
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest
   ```

## 🛠️ Development Workflow

### Code Style

GPUX follows strict code quality standards:

- **Formatting**: [Black](https://black.readthedocs.io/) with line length 88
- **Linting**: [Ruff](https://docs.astral.sh/ruff/) for fast Python linting
- **Type Checking**: [mypy](https://mypy.readthedocs.io/) for static type checking
- **Import Sorting**: [isort](https://pycqa.github.io/isort/) for import organization

### Pre-commit Hooks

Pre-commit hooks automatically run:
- `ruff` for linting and formatting
- `mypy` for type checking
- `pytest` for running tests

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/gpux --cov-report=term-missing

# Run specific test file
pytest tests/test_runtime.py

# Run specific test
pytest tests/test_runtime.py::TestGPUXRuntime::test_infer
```

### Code Quality Checks

```bash
# Run linting
ruff check src/ tests/

# Run type checking
mypy src/

# Run formatting check
ruff format --check src/ tests/

# Fix formatting issues
ruff format src/ tests/
```

## 📝 Making Changes

### Branch Naming

Use descriptive branch names:
- `feat/add-new-provider` - New features
- `fix/memory-leak-issue` - Bug fixes
- `docs/update-api-reference` - Documentation updates
- `refactor/simplify-runtime` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat(runtime): add GPU memory management
fix(config): resolve YAML parsing edge case
docs(readme): update installation instructions
refactor(providers): simplify provider selection logic
test(runtime): add integration tests for inference
chore(deps): update ruff to latest version
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

### Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run all checks** locally
6. **Submit a pull request** with a clear description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All existing tests still pass

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_config.py          # Configuration tests
├── test_providers.py       # Provider management tests
├── test_runtime.py         # Runtime engine tests
├── test_cli_*.py          # CLI command tests
├── test_utils_*.py        # Utility function tests
└── conftest.py            # Test fixtures and configuration
```

### Writing Tests

1. **Test Coverage**: Aim for 90%+ coverage on new code
2. **Test Naming**: Use descriptive test names
3. **Test Organization**: Group related tests in classes
4. **Fixtures**: Use pytest fixtures for common setup
5. **Mocking**: Mock external dependencies appropriately

### Example Test

```python
import pytest
from unittest.mock import MagicMock, patch
from gpux.core.runtime import GPUXRuntime

class TestGPUXRuntime:
    def test_infer_success(self, sample_model_path):
        """Test successful inference."""
        runtime = GPUXRuntime(sample_model_path)
        
        input_data = {"input": [1, 2, 3, 4]}
        results = runtime.infer(input_data)
        
        assert isinstance(results, dict)
        assert "output" in results
        
        runtime.cleanup()
    
    def test_infer_invalid_input(self, sample_model_path):
        """Test inference with invalid input."""
        runtime = GPUXRuntime(sample_model_path)
        
        with pytest.raises(ValueError, match="Invalid input"):
            runtime.infer({"invalid": "data"})
        
        runtime.cleanup()
```

## 🏗️ Architecture Guidelines

### Adding New Features

1. **Core Functionality**: Add to `src/gpux/core/`
2. **CLI Commands**: Add to `src/gpux/cli/`
3. **Configuration**: Update `src/gpux/config/`
4. **Utilities**: Add to `src/gpux/utils/`

### Adding New Execution Providers

1. **Update Provider Enum**: Add to `ExecutionProvider` in `providers.py`
2. **Add Provider Logic**: Implement provider-specific code
3. **Update Priority**: Add to provider priority list
4. **Add Tests**: Test provider detection and usage
5. **Update Documentation**: Document new provider

### Configuration Changes

1. **Update Schema**: Modify Pydantic models in `parser.py`
2. **Add Validation**: Add field validators if needed
3. **Update Tests**: Test new configuration options
4. **Update Documentation**: Document new options

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use Google-style docstrings for all public functions
- **Type Hints**: Add type hints for all function parameters and returns
- **Comments**: Add comments for complex logic

### Example Docstring

```python
def infer(self, input_data: dict[str, Any]) -> dict[str, Any]:
    """Run inference on input data.
    
    Args:
        input_data: Dictionary mapping input names to data arrays
        
    Returns:
        Dictionary mapping output names to result arrays
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If inference fails
        
    Example:
        >>> runtime = GPUXRuntime("model.onnx")
        >>> results = runtime.infer({"input": [1, 2, 3]})
        >>> print(results["output"])
    """
```

### Documentation Updates

When adding new features:
1. Update API documentation in `docs/API.md`
2. Update architecture docs in `docs/ARCHITECTURE.md`
3. Update README if needed
4. Add usage examples

## 🐛 Bug Reports

### Before Submitting

1. **Check existing issues** to avoid duplicates
2. **Test with latest version** to ensure bug still exists
3. **Gather information** about your environment

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 13.0]
- Python: [e.g., 3.11.0]
- GPUX: [e.g., 0.1.0]
- GPU: [e.g., NVIDIA RTX 4090]

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

### Before Submitting

1. **Check existing issues** for similar requests
2. **Consider the project scope** - does it fit GPUX's goals?
3. **Think about implementation** - is it feasible?

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other approaches were considered?

## Additional Context
Any other relevant information
```

## 🏷️ Release Process

### Version Numbering

GPUX follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update changelog** with new features/fixes
3. **Run full test suite** to ensure everything works
4. **Update documentation** if needed
5. **Create release** on GitHub
6. **Publish to PyPI** (maintainers only)

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** in discussions

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: support@gpux.io for direct contact

## 📄 License

By contributing to GPUX, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub** contributor statistics

Thank you for contributing to GPUX! 🚀
