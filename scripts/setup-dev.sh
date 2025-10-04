#!/bin/bash
set -e

echo "🚀 Setting up GPUX development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv is installed"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --dev

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
uv run pre-commit install

# Run initial checks
echo "🧪 Running initial checks..."
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest

echo "✅ Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  make help          - Show all available commands"
echo "  make test          - Run tests"
echo "  make lint          - Run linting"
echo "  make format        - Format code"
echo "  make type-check    - Run type checking"
echo "  make check         - Run all checks"
echo "  make build         - Build package"
echo ""
echo "Happy coding! 🎉"
