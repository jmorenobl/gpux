# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY docs/ ./docs/
COPY README.md ./
COPY Makefile ./

# Install dependencies
RUN uv sync --dev

# Create non-root user
RUN useradd --create-home --shell /bin/bash gpux
USER gpux

# Expose port for serving
EXPOSE 8080

# Default command
CMD ["uv", "run", "python", "-m", "gpux.cli.main", "--help"]
