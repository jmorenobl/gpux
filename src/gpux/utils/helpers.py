"""Helper utilities for GPUX."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_system_info() -> dict[str, Any]:
    """Get system information.

    Returns:
        Dictionary containing system information
    """
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture(),
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    value = float(bytes_value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value = value / 1024.0
    return f"{value:.1f} PB"


def format_time(seconds: float) -> str:
    """Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1.5 ms", "2.3 s")
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} μs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files(
    directory: str | Path,
    pattern: str,
    *,
    recursive: bool = True,
) -> list[Path]:
    """Find files matching a pattern.

    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.onnx", "GPUXfile")
        recursive: Whether to search recursively

    Returns:
        List of matching file paths
    """
    directory = Path(directory)

    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))


def validate_file_extension(file_path: str | Path, valid_extensions: list[str]) -> bool:
    """Validate file extension.

    Args:
        file_path: Path to file
        valid_extensions: List of valid extensions (e.g., [".onnx", ".onnx.gz"])

    Returns:
        True if file has valid extension
    """
    file_path = Path(file_path)
    return any(str(file_path).endswith(ext) for ext in valid_extensions)


def run_command(
    command: list[str],
    cwd: str | Path | None = None,
    *,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command and return the result.

    Args:
        command: Command to run as list of strings
        cwd: Working directory
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit code

    Returns:
        CompletedProcess object

    Raises:
        subprocess.CalledProcessError: If command fails and check=True
    """
    try:
        return subprocess.run(  # noqa: S603
            command,
            cwd=cwd,
            capture_output=capture_output,
            check=check,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.exception("Command failed: %s", " ".join(command))
        logger.exception("Exit code: %s", e.returncode)
        if e.stdout:
            logger.exception("Stdout: %s", e.stdout)
        if e.stderr:
            logger.exception("Stderr: %s", e.stderr)
        raise


def check_dependencies() -> dict[str, bool]:
    """Check if required dependencies are available.

    Returns:
        Dictionary mapping dependency names to availability
    """
    dependencies = {}

    # Check ONNX Runtime
    try:
        import onnxruntime  # noqa: F401, PLC0415

        dependencies["onnxruntime"] = True
    except ImportError:
        dependencies["onnxruntime"] = False

    # Check ONNX
    try:
        import onnx  # noqa: F401, PLC0415

        dependencies["onnx"] = True
    except ImportError:
        dependencies["onnx"] = False

    # Check NumPy
    try:
        import numpy as np  # noqa: F401, PLC0415

        dependencies["numpy"] = True
    except ImportError:
        dependencies["numpy"] = False

    # Check PyYAML
    try:
        import yaml  # noqa: F401, PLC0415

        dependencies["yaml"] = True
    except ImportError:
        dependencies["yaml"] = False

    # Check Click
    try:
        import click  # noqa: F401, PLC0415

        dependencies["click"] = True
    except ImportError:
        dependencies["click"] = False

    # Check Typer
    try:
        import typer  # noqa: F401, PLC0415

        dependencies["typer"] = True
    except ImportError:
        dependencies["typer"] = False

    # Check Rich
    try:
        import rich  # noqa: F401, PLC0415

        dependencies["rich"] = True
    except ImportError:
        dependencies["rich"] = False

    # Check Pydantic
    try:
        import pydantic  # noqa: F401, PLC0415

        dependencies["pydantic"] = True
    except ImportError:
        dependencies["pydantic"] = False

    return dependencies


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information if available.

    Returns:
        Dictionary containing GPU information
    """
    gpu_info = {
        "available": False,
        "devices": [],
        "provider": None,
    }

    try:
        import onnxruntime as ort  # noqa: PLC0415

        # Get available providers
        providers = ort.get_available_providers()
        gpu_providers = [p for p in providers if p != "CPUExecutionProvider"]

        if gpu_providers:
            gpu_info["available"] = True
            gpu_info["providers"] = gpu_providers
            gpu_info["provider"] = gpu_providers[0]  # First available GPU provider

        # Try to get device information
        try:
            # This might not work on all systems
            session_options = ort.SessionOptions()
            ort.InferenceSession(
                "dummy.onnx",  # This will fail, but we just want device info
                sess_options=session_options,
                providers=gpu_providers,
            )
        except (RuntimeError, AttributeError, OSError):
            # Expected to fail, but we might get device info
            pass

    except ImportError:
        logger.warning("ONNX Runtime not available for GPU detection")
    except (RuntimeError, AttributeError, OSError) as e:
        logger.warning("Failed to detect GPU: %s", e)

    return gpu_info


def create_gpuxfile_template(
    name: str,
    model_path: str | Path,
    output_path: str | Path = "gpux.yml",
) -> None:
    """Create a gpux.yml template.

    Args:
        name: Model name
        model_path: Path to the model file
        output_path: Output path for gpux.yml
    """
    model_path = Path(model_path)

    template = f"""# gpux.yml - Docker-like configuration for ML inference
name: {name}
version: 1.0.0
description: "ML inference model for {name}"

model:
  source: {model_path.name}
  format: onnx

inputs:
  input_name:
    type: float32
    shape: [1, 10]  # Adjust based on your model
    required: true
    description: "Input description"

outputs:
  output_name:
    type: float32
    shape: [1, 2]  # Adjust based on your model
    description: "Output description"

runtime:
  gpu:
    memory: 2GB
    backend: auto  # vulkan | metal | dx12 | cuda | rocm | coreml

serving:
  port: 8080
  host: 0.0.0.0
  batch_size: 1
  timeout: 5
"""

    with Path(output_path).open("w", encoding="utf-8") as f:
        f.write(template)

    logger.info("Created gpux.yml template at %s", output_path)


def validate_gpuxfile(file_path: str | Path) -> bool:
    """Validate a gpux.yml file.

    Args:
        file_path: Path to gpux.yml file

    Returns:
        True if valid, False otherwise
    """
    try:
        from gpux.config.parser import GPUXConfigParser  # noqa: PLC0415

        parser = GPUXConfigParser()
        parser.parse_file(file_path)
    except Exception:
        logger.exception("gpux.yml validation failed")
        return False
    else:
        return True


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root
    """
    # Look for pyproject.toml or setup.py
    current = Path.cwd()

    for parent in [current, *list(current.parents)]:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent

    # Fallback to current directory
    return current
