# Contributing to GPUX

Thank you for your interest in contributing to GPUX!

---

## Ways to Contribute

### 🐛 Report Bugs

Found a bug? Please report it!

1. Check [existing issues](https://github.com/gpux/gpux-runtime/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, GPU, Python version)

### 💡 Suggest Features

Have an idea? We'd love to hear it!

1. Check [discussions](https://github.com/gpux/gpux-runtime/discussions)
2. Open a feature request issue
3. Describe the use case and proposed solution

### 📝 Improve Documentation

Documentation improvements are always welcome:

- Fix typos or clarify explanations
- Add examples or tutorials
- Improve API documentation
- Translate to other languages

### 💻 Submit Code

Ready to contribute code?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## Development Setup

### Prerequisites

- Python 3.11+
- uv package manager
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/gpux/gpux-runtime.git
cd gpux-runtime

# Install dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test
uv run pytest tests/test_runtime.py
```

### Code Quality

```bash
# Lint code
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

---

## Contribution Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small
- Line length: 88 characters

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat(cli): add new inspect command
fix(runtime): resolve GPU memory leak
docs(tutorial): add batch inference example
test(providers): add CUDA provider tests
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. **Create a branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow code style guidelines
3. **Add tests**: Maintain >90% coverage
4. **Update docs**: Document new features
5. **Run checks**: `make check` (or run tests + linters)
6. **Commit**: Use conventional commit messages
7. **Push**: `git push origin feature/your-feature`
8. **Create PR**: Describe changes and link issues

### PR Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check`)
- [ ] Type checking passes (`uv run mypy src/`)
- [ ] Documentation updated
- [ ] Changelog updated (for user-facing changes)
- [ ] Conventional commit messages

---

## Community Guidelines

### Be Respectful

- Be welcoming to newcomers
- Respect different perspectives
- Provide constructive feedback
- Focus on the issue, not the person

### Be Collaborative

- Ask for help when needed
- Share knowledge and learnings
- Review others' contributions
- Celebrate successes together

---

## Getting Help

### Documentation

- [API Reference](../reference/index.md)
- [Development Guide](../guide/index.md)
- [Architecture Overview](architecture.md)

### Communication

- **Discussions**: [GitHub Discussions](https://github.com/gpux/gpux-runtime/discussions)
- **Issues**: [GitHub Issues](https://github.com/gpux/gpux-runtime/issues)
- **Discord**: [GPUX Community](https://discord.gg/gpux)

---

## Recognition

Contributors are recognized in:

- README.md contributors section
- Release notes
- Documentation credits

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making GPUX better! 🚀
