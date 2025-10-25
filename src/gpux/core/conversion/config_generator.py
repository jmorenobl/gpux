"""Configuration generator for converted models."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml

from gpux.core.conversion.optimizer import ConversionError
from gpux.core.models import ModelInspector

if TYPE_CHECKING:
    from gpux.core.managers.base import ModelMetadata

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Generates gpux.yml configuration files for converted models."""

    def __init__(self) -> None:
        """Initialize the config generator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_config(
        self,
        metadata: ModelMetadata,
        onnx_path: Path,
        output_path: Path | None = None,
        **kwargs: Any,
    ) -> Path:
        """Generate a gpux.yml configuration file for a converted model.

        Args:
            metadata: Model metadata from registry
            onnx_path: Path to the converted ONNX model
            output_path: Output path for the config file (optional)
            **kwargs: Additional configuration parameters

        Returns:
            Path to the generated configuration file

        Raises:
            ConversionError: If config generation fails
        """
        if not onnx_path.exists():
            msg = f"ONNX model not found: {onnx_path}"
            raise ConversionError(msg)

        if output_path is None:
            output_path = onnx_path.parent / "gpux.yml"

        self.logger.info(
            "Generating configuration for model %s: %s",
            metadata.model_id,
            output_path,
        )

        try:
            # Inspect the ONNX model to get input/output information
            inspector = ModelInspector()
            model_info = inspector.inspect(str(onnx_path))

            # Generate configuration
            config = self._create_config_dict(metadata, model_info, onnx_path, **kwargs)

            # Write configuration file
            output_path.write_text(
                yaml.dump(config, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )

            self.logger.info("Configuration generated successfully: %s", output_path)
        except Exception as e:
            self.logger.exception("Failed to generate configuration")
            msg = f"Config generation failed: {e}"
            raise ConversionError(msg) from e
        else:
            return output_path

    def _create_config_dict(
        self,
        metadata: ModelMetadata,
        model_info: Any,
        onnx_path: Path,
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Create the configuration dictionary.

        Args:
            metadata: Model metadata from registry
            model_info: Model information from ModelInspector
            onnx_path: Path to the ONNX model
            **kwargs: Additional configuration parameters

        Returns:
            Configuration dictionary
        """
        # Extract input/output information
        inputs = self._extract_inputs(model_info)
        outputs = self._extract_outputs(model_info)

        # Generate model name from metadata
        model_name = metadata.model_id.replace("/", "-").replace("_", "-").lower()

        config = {
            "name": model_name,
            "version": "1.0.0",
            "description": metadata.description
            or f"Converted model from {metadata.registry}",
            "model": {
                "source": str(onnx_path.relative_to(onnx_path.parent)),
                "format": "onnx",
            },
            "inputs": inputs,
            "outputs": outputs,
            "runtime": {
                "gpu": {
                    "memory": "2GB",
                    "backend": "auto",
                },
                "batch_size": 1,
                "timeout": 30,
            },
            "serving": {
                "port": 8080,
                "host": "localhost",  # Use localhost instead of 0.0.0.0 for security
                "batch_size": 1,
                "timeout": 5,
            },
            "metadata": {
                "original_model_id": metadata.model_id,
                "original_registry": metadata.registry,
                "original_format": metadata.format,
                "conversion_date": datetime.now(UTC).isoformat(),
                "converter": "PyTorchConverter",
                "onnx_opset_version": getattr(model_info, "opset_version", 11),
            },
        }

        # Add tags if available
        if metadata.tags:
            config["tags"] = metadata.tags

        # Add custom metadata
        if (
            metadata.metadata
            and isinstance(metadata.metadata, dict)
            and isinstance(config["metadata"], dict)
        ):
            config["metadata"].update(metadata.metadata)

        return config

    def _extract_inputs(self, model_info: Any) -> dict[str, dict[str, Any]]:
        """Extract input information from model.

        Args:
            model_info: Model information from ModelInspector

        Returns:
            Dictionary mapping input names to their specifications
        """
        inputs = {}

        if hasattr(model_info, "inputs"):
            for input_info in model_info.inputs:
                input_name = input_info.name
                input_type = self._convert_onnx_type_to_numpy(input_info.type)
                input_shape = list(input_info.shape) if input_info.shape else [1]

                # Handle dynamic dimensions
                input_shape = [dim if dim > 0 else 1 for dim in input_shape]

                inputs[input_name] = {
                    "type": input_type,
                    "shape": input_shape,
                    "required": True,
                }

        # Fallback for common transformer inputs
        if not inputs:
            inputs = {
                "input_ids": {
                    "type": "int64",
                    "shape": [1, 128],
                    "required": True,
                },
                "attention_mask": {
                    "type": "int64",
                    "shape": [1, 128],
                    "required": True,
                },
            }

        return inputs

    def _extract_outputs(self, model_info: Any) -> dict[str, dict[str, Any]]:
        """Extract output information from model.

        Args:
            model_info: Model information from ModelInspector

        Returns:
            Dictionary mapping output names to their specifications
        """
        outputs = {}

        if hasattr(model_info, "outputs"):
            for output_info in model_info.outputs:
                output_name = output_info.name
                output_type = self._convert_onnx_type_to_numpy(output_info.type)
                output_shape = list(output_info.shape) if output_info.shape else [1]

                # Handle dynamic dimensions
                output_shape = [dim if dim > 0 else 1 for dim in output_shape]

                outputs[output_name] = {
                    "type": output_type,
                    "shape": output_shape,
                }

        # Fallback for common transformer outputs
        if not outputs:
            outputs = {
                "last_hidden_state": {
                    "type": "float32",
                    "shape": [1, 128, 768],
                },
                "pooler_output": {
                    "type": "float32",
                    "shape": [1, 768],
                },
            }

        return outputs

    def _convert_onnx_type_to_numpy(self, onnx_type: str) -> str:
        """Convert ONNX type to NumPy type string.

        Args:
            onnx_type: ONNX type string

        Returns:
            NumPy type string
        """
        type_mapping = {
            "tensor(float)": "float32",
            "tensor(double)": "float64",
            "tensor(int32)": "int32",
            "tensor(int64)": "int64",
            "tensor(bool)": "bool",
            "tensor(string)": "str",
        }

        return type_mapping.get(onnx_type, "float32")

    def update_existing_config(
        self,
        config_path: Path,
        metadata: ModelMetadata,
        onnx_path: Path,
        **kwargs: Any,
    ) -> Path:
        """Update an existing configuration file.

        Args:
            config_path: Path to existing configuration file
            metadata: Model metadata from registry
            onnx_path: Path to the ONNX model
            **kwargs: Additional configuration parameters

        Returns:
            Path to the updated configuration file
        """
        if not config_path.exists():
            return self.generate_config(metadata, onnx_path, config_path, **kwargs)

        try:
            # Load existing configuration
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            # Update model source
            config["model"]["source"] = str(onnx_path.relative_to(onnx_path.parent))
            config["model"]["format"] = "onnx"

            # Update metadata
            if "metadata" not in config:
                config["metadata"] = {}

            config["metadata"].update(
                {
                    "original_model_id": metadata.model_id,
                    "original_registry": metadata.registry,
                    "original_format": metadata.format,
                    "conversion_date": datetime.now(UTC).isoformat(),
                    "converter": "PyTorchConverter",
                }
            )

            # Write updated configuration
            config_path.write_text(
                yaml.dump(config, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )

            self.logger.info("Configuration updated successfully: %s", config_path)
        except Exception as e:
            self.logger.exception("Failed to update configuration")
            msg = f"Config update failed: {e}"
            raise ConversionError(msg) from e
        else:
            return config_path
