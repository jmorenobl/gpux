"""TensorFlow to ONNX conversion implementation."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore[assignment,unused-ignore]

try:
    import tf2onnx
except ImportError:
    tf2onnx = None

from gpux.core.conversion.base import ONNXConverter
from gpux.core.conversion.optimizer import ConversionError, ModelOptimizer

if TYPE_CHECKING:
    from gpux.core.managers.base import ModelMetadata

logger = logging.getLogger(__name__)


class TensorFlowConverter(ONNXConverter):
    """TensorFlow to ONNX converter using tf2onnx."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the TensorFlow converter.

        Args:
            cache_dir: Directory to cache converted models
        """
        if tf is None:
            msg = (
                "TensorFlow is not installed. "
                "Install with: pip install gpux[tensorflow]"
            )
            raise ImportError(msg)
        if tf2onnx is None:
            msg = (
                "tf2onnx is not installed. "
                "Install with: pip install gpux[tensorflow]"
            )
            raise ImportError(msg)
        super().__init__(cache_dir)
        self.optimizer = ModelOptimizer()

    def can_convert(self, metadata: ModelMetadata) -> bool:
        """Check if this converter can handle the given model.

        Args:
            metadata: Model metadata from registry

        Returns:
            True if this converter can handle TensorFlow models
        """
        return metadata.format.lower() in ("tensorflow", "tf", "keras", "h5")

    def convert(
        self,
        metadata: ModelMetadata,
        output_path: Path | None = None,
        **kwargs: Any,
    ) -> Path:
        """Convert a TensorFlow model to ONNX format.

        Args:
            metadata: Model metadata from registry
            output_path: Output path for converted model (optional)
            **kwargs: Additional conversion parameters

        Returns:
            Path to the converted ONNX model

        Raises:
            ConversionError: If conversion fails
        """
        if not self.can_convert(metadata):
            msg = f"Cannot convert model format: {metadata.format}"
            raise ConversionError(msg)

        if output_path is None:
            output_path = (
                self.cache_dir / f"{metadata.model_id.replace('/', '--')}.onnx"
            )

        self.logger.info(
            "Converting TensorFlow model %s to ONNX: %s",
            metadata.model_id,
            output_path,
        )

        try:
            # Try tf2onnx first (preferred method)
            onnx_path = self._convert_with_tf2onnx(metadata, output_path, **kwargs)
        except Exception as e:
            self.logger.warning(
                "tf2onnx conversion failed: %s",
                e,
            )
            # Manual conversion is not implemented
            msg = (
                "Manual TensorFlow to ONNX conversion not implemented. "
                "Please use tf2onnx."
            )
            raise ConversionError(msg) from e

        # Optimize the converted model
        try:
            optimized_path = self.optimizer.optimize_model(onnx_path)
            # Replace original with optimized version
            optimized_path.replace(onnx_path)
        except Exception as e:
            self.logger.warning(
                "Model optimization failed, using unoptimized model: %s", e
            )

        # Validate the final model
        self.optimizer.validate_model(onnx_path)

        self.logger.info("Successfully converted model to ONNX: %s", onnx_path)
        return onnx_path

    def _convert_with_tf2onnx(
        self,
        metadata: ModelMetadata,
        output_path: Path,
        **kwargs: Any,  # noqa: ARG002
    ) -> Path:
        """Convert using tf2onnx library.

        Args:
            metadata: Model metadata from registry
            output_path: Output path for converted model
            **kwargs: Additional conversion parameters

        Returns:
            Path to the converted ONNX model

        Raises:
            ConversionError: If conversion fails
        """
        if tf2onnx is None:
            msg = (
                "tf2onnx is required for TensorFlow to ONNX conversion. "
                "Install with: pip install gpux[tensorflow]"
            )
            raise ConversionError(msg)

        # Find the main model file
        model_file = self._find_model_file(metadata)
        if not model_file:
            msg = f"No TensorFlow model file found for {metadata.model_id}"
            raise ConversionError(msg)

        self.logger.info("Loading TensorFlow model from: %s", model_file)

        try:
            # Load TensorFlow model
            if model_file.suffix == ".h5":
                model = tf.keras.models.load_model(model_file)
            elif model_file.suffix == ".pb":
                # Load SavedModel
                model = tf.saved_model.load(str(model_file))
            else:
                # Try loading as SavedModel directory
                model = tf.saved_model.load(str(model_file))

            # Convert to ONNX using tf2onnx
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "model.onnx"

                # Get input shapes from model
                input_shapes = self._get_tf_input_shapes(model)

                # Convert to ONNX
                tf2onnx.convert.from_keras(
                    model,
                    output_path=str(temp_path),
                    opset=11,  # Use ONNX opset 11 for compatibility
                    input_signature=input_shapes,
                )

                # Move to final location
                temp_path.rename(output_path)

        except Exception as e:
            msg = f"tf2onnx conversion failed: {e}"
            raise ConversionError(msg) from e

        return output_path

    def _find_model_file(self, metadata: ModelMetadata) -> Path | None:
        """Find the main TensorFlow model file.

        Args:
            metadata: Model metadata from registry

        Returns:
            Path to the model file or None if not found
        """
        if not metadata.files:
            return None

        # Look for common TensorFlow model files
        model_file = self._find_tf_model_files(metadata)
        if model_file:
            return model_file

        # If no specific TensorFlow file found, look for directories
        return self._find_savedmodel_directories(metadata)

    def _find_tf_model_files(self, metadata: ModelMetadata) -> Path | None:
        """Find TensorFlow model files with common extensions."""
        tf_extensions = [".h5", ".pb", ".savedmodel"]

        for filepath in metadata.files.values():
            if isinstance(filepath, Path) and filepath.suffix in tf_extensions:
                return filepath
            if isinstance(filepath, str):
                path = Path(filepath)
                if path.suffix in tf_extensions:
                    return path
        return None

    def _find_savedmodel_directories(self, metadata: ModelMetadata) -> Path | None:
        """Find SavedModel directories."""
        for filepath in metadata.files.values():
            if isinstance(filepath, Path) and filepath.is_dir():
                # Check if it's a SavedModel directory
                if (filepath / "saved_model.pb").exists():
                    return filepath
            elif isinstance(filepath, str):
                path = Path(filepath)
                if path.is_dir() and (path / "saved_model.pb").exists():
                    return path
        return None

    def _get_tf_input_shapes(self, model: Any) -> list[tuple[Any, ...]] | None:
        """Get input shapes from TensorFlow model.

        Args:
            model: TensorFlow model

        Returns:
            List of input shapes
        """
        try:
            # Try Keras model first
            keras_shapes = self._extract_keras_shapes(model)
            if keras_shapes is not None:
                return keras_shapes

            # Try SavedModel
            savedmodel_shapes = self._extract_savedmodel_shapes(model)
            if savedmodel_shapes is not None:
                return savedmodel_shapes

        except Exception as e:
            self.logger.warning("Could not determine input shapes: %s", e)

        # No input shapes could be determined
        return None

    def _extract_keras_shapes(self, model: Any) -> list[tuple[Any, ...]] | None:
        """Extract input shapes from Keras model object."""
        if not hasattr(model, "input_shape"):
            return None

        if isinstance(model.input_shape, list):
            return [shape for shape in model.input_shape if shape is not None]
        return [model.input_shape] if model.input_shape else []

    def _extract_savedmodel_shapes(self, model: Any) -> list[tuple[Any, ...]] | None:
        """Extract input shapes from SavedModel object."""
        if not hasattr(model, "signatures") or not model.signatures:
            return None

        signatures = model.signatures
        default_sig = signatures.get("serving_default")
        if not default_sig:
            return None

        input_shapes = []
        for input_tensor in default_sig.inputs:
            shape = input_tensor.shape
            if shape.rank is not None:
                input_shapes.append(tuple(shape.as_list()))
        return input_shapes

    def get_input_shapes(self, metadata: ModelMetadata) -> dict[str, list[int]]:
        """Get input shapes for the model.

        Args:
            metadata: Model metadata from registry

        Returns:
            Dictionary mapping input names to shapes
        """
        model_file = self._find_model_file(metadata)
        if not model_file:
            return {}

        try:
            if model_file.suffix == ".h5":
                return self._get_keras_input_shapes(model_file)
            if model_file.is_dir() and (model_file / "saved_model.pb").exists():
                return self._get_savedmodel_input_shapes(model_file)
        except Exception as e:
            self.logger.warning("Could not determine input shapes: %s", e)

        return {}

    def _get_keras_input_shapes(self, model_file: Path) -> dict[str, list[int]]:
        """Get input shapes from Keras model."""
        model = tf.keras.models.load_model(model_file)
        if hasattr(model, "input_shape"):
            if isinstance(model.input_shape, list):
                return {
                    f"input_{i}": list(shape) if shape else [1]
                    for i, shape in enumerate(model.input_shape)
                }
            return {"input_0": list(model.input_shape) if model.input_shape else [1]}
        return {}

    def _get_savedmodel_input_shapes(self, model_file: Path) -> dict[str, list[int]]:
        """Get input shapes from SavedModel."""
        model = tf.saved_model.load(str(model_file))
        if hasattr(model, "signatures"):
            signatures = model.signatures
            if signatures:
                default_sig = signatures.get("serving_default")
                if default_sig:
                    input_shapes = {}
                    for i, input_tensor in enumerate(default_sig.inputs):
                        shape = input_tensor.shape
                        if shape.rank is not None:
                            input_shapes[f"input_{i}"] = shape.as_list()
                    return input_shapes
        return {}

    def get_output_shapes(self, metadata: ModelMetadata) -> dict[str, list[int]]:
        """Get output shapes for the model.

        Args:
            metadata: Model metadata from registry

        Returns:
            Dictionary mapping output names to shapes
        """
        model_file = self._find_model_file(metadata)
        if not model_file:
            return {}

        try:
            if model_file.suffix == ".h5":
                return self._get_keras_output_shapes(model_file)
            if model_file.is_dir() and (model_file / "saved_model.pb").exists():
                return self._get_savedmodel_output_shapes(model_file)
        except Exception as e:
            self.logger.warning("Could not determine output shapes: %s", e)

        return {}

    def _get_keras_output_shapes(self, model_file: Path) -> dict[str, list[int]]:
        """Get output shapes from Keras model."""
        model = tf.keras.models.load_model(model_file)
        if hasattr(model, "output_shape"):
            if isinstance(model.output_shape, list):
                return {
                    f"output_{i}": list(shape) if shape else [1]
                    for i, shape in enumerate(model.output_shape)
                }
            return {"output_0": list(model.output_shape) if model.output_shape else [1]}
        return {}

    def _get_savedmodel_output_shapes(self, model_file: Path) -> dict[str, list[int]]:
        """Get output shapes from SavedModel."""
        model = tf.saved_model.load(str(model_file))
        if hasattr(model, "signatures"):
            signatures = model.signatures
            if signatures:
                default_sig = signatures.get("serving_default")
                if default_sig:
                    output_shapes = {}
                    for i, output_tensor in enumerate(default_sig.outputs):
                        shape = output_tensor.shape
                        if shape.rank is not None:
                            output_shapes[f"output_{i}"] = shape.as_list()
                    return output_shapes
        return {}
