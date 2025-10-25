.PHONY: help install install-dev test test-cov lint format type-check clean build publish

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	uv sync

install-dev: ## Install development dependencies
	uv sync --dev

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src/gpux --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	uv run pytest --no-cov

lint: ## Run linting
	uv run ruff check src/ tests/

format: ## Format code
	uv run ruff format src/ tests/

type-check: ## Run type checking
	uv run mypy src/

check: lint type-check test ## Run all checks

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

build: ## Build the package
	uv build

publish: ## Publish to PyPI (requires PYPI_API_TOKEN)
	uv publish

install-pre-commit: ## Install pre-commit hooks
	uv run pre-commit install

update-deps: ## Update dependencies
	uv lock --upgrade
	uv sync --dev

security: ## Run security checks
	uv run ruff check --select S src/ tests/
	uv run bandit -r src/

docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

benchmark: ## Run performance benchmarks
	uv run pytest tests/ -m benchmark

integration: ## Run integration tests
	uv run pytest tests/ -m integration

all: clean install-dev check build ## Run everything (clean, install, check, build)
