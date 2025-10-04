"""GPUX - Docker-like GPU runtime for ML inference.

GPUX provides universal GPU compatibility for ML inference workloads,
allowing you to run the same model on any GPU without compatibility issues.
"""

__version__ = "0.1.0"
__author__ = "GPUX Team"
__email__ = "team@gpux.io"

from gpux.core import ExecutionProvider, GPUXRuntime, ModelInfo

__all__ = [
    "ExecutionProvider",
    "GPUXRuntime",
    "ModelInfo",
]
