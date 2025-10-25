"""Base ONNX converter interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpux.core.managers.base import ModelMetadata

logger = logging.getLogger(__name__)


class ONNXConverter(ABC):
    """Abstract base class for ONNX model converters.

    This class defines the interface that all model format converters must implement.
    It follows the strategy pattern to allow different frameworks to be supported
    with a consistent API.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the converter.

        Args:
            cache_dir: Directory to cache converted models
        """
        self.cache_dir = cache_dir or Path.home() / ".gpux" / "models" / "converted"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def can_convert(self, metadata: ModelMetadata) -> bool:
        """Check if this converter can handle the given model.

        Args:
            metadata: Model metadata from registry

        Returns:
            True if this converter can handle the model
        """
        ...

    @abstractmethod
    def convert(
        self,
        metadata: ModelMetadata,
        output_path: Path | None = None,
        **kwargs: Any,
    ) -> Path:
        """Convert a model to ONNX format.

        Args:
            metadata: Model metadata from registry
            output_path: Output path for converted model (optional)
            **kwargs: Additional conversion parameters

        Returns:
            Path to the converted ONNX model

        Raises:
            ConversionError: If conversion fails
        """
        ...

    @abstractmethod
    def get_input_shapes(self, metadata: ModelMetadata) -> dict[str, list[int]]:
        """Get input shapes for the model.

        Args:
            metadata: Model metadata from registry

        Returns:
            Dictionary mapping input names to shapes
        """
        ...

    @abstractmethod
    def get_output_shapes(self, metadata: ModelMetadata) -> dict[str, list[int]]:
        """Get output shapes for the model.

        Args:
            metadata: Model metadata from registry

        Returns:
            Dictionary mapping output names to shapes
        """
        ...

    def get_conversion_info(self, metadata: ModelMetadata) -> dict[str, Any]:
        """Get information about the conversion process.

        Args:
            metadata: Model metadata from registry

        Returns:
            Dictionary with conversion information
        """
        return {
            "converter": self.__class__.__name__,
            "model_id": metadata.model_id,
            "registry": metadata.registry,
            "original_format": metadata.format,
            "target_format": "onnx",
        }
