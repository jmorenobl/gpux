"""Configuration parser for GPUX files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class InputConfig(BaseModel):
    """Input configuration for GPUX."""

    name: str
    type: str
    shape: list[int] | None = None
    required: bool = True
    max_length: int | None = None
    description: str | None = None


class OutputConfig(BaseModel):
    """Output configuration for GPUX."""

    name: str
    type: str
    shape: list[int] | None = None
    labels: list[str] | None = None
    description: str | None = None


class ModelConfig(BaseModel):
    """Model configuration for GPUX."""

    source: str | Path
    format: str = "onnx"
    version: str | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str | Path) -> Path:
        """Validate model source path."""
        if isinstance(v, str):
            return Path(v)
        return v


class GPUConfig(BaseModel):
    """GPU configuration for GPUX."""

    memory: str = "2GB"
    backend: str = "auto"  # auto, vulkan, metal, dx12, cuda, rocm, coreml

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, v: str) -> str:
        """Validate memory specification."""
        v = v.upper().strip()
        if not (v.endswith(("GB", "MB", "KB"))):
            msg = "Memory must be specified as GB, MB, or KB"
            raise ValueError(msg)
        return v


class RuntimeConfig(BaseModel):
    """Runtime configuration for GPUX."""

    gpu: GPUConfig = Field(default_factory=GPUConfig)
    timeout: int = 30
    batch_size: int = 1
    enable_profiling: bool = False


class ServingConfig(BaseModel):
    """Serving configuration for GPUX."""

    port: int = 8080
    host: str = "0.0.0.0"  # noqa: S104
    batch_size: int = 1
    timeout: int = 5
    max_workers: int = 4


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration for GPUX."""

    tokenizer: str | None = None
    max_length: int | None = None
    resize: list[int] | None = None
    normalize: str | None = None
    custom: dict[str, Any] | None = None


class GPUXConfig(BaseModel):
    """Main GPUX configuration model."""

    name: str
    version: str = "1.0.0"
    description: str | None = None

    model: ModelConfig
    inputs: list[InputConfig] = Field(default_factory=list)
    outputs: list[OutputConfig] = Field(default_factory=list)

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    serving: ServingConfig | None = None
    preprocessing: PreprocessingConfig | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v: list[InputConfig]) -> list[InputConfig]:
        """Validate inputs configuration."""
        if not v:
            msg = "At least one input must be specified"
            raise ValueError(msg)
        return v

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, v: list[OutputConfig]) -> list[OutputConfig]:
        """Validate outputs configuration."""
        if not v:
            msg = "At least one output must be specified"
            raise ValueError(msg)
        return v


class GPUXConfigParser:
    """Parser for GPUX configuration files."""

    def __init__(self) -> None:
        """Initialize the configuration parser."""
        self._config: GPUXConfig | None = None

    def parse_file(self, config_path: str | Path) -> GPUXConfig:
        """Parse a GPUX configuration file.

        Args:
            config_path: Path to the GPUX configuration file

        Returns:
            Parsed configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        try:
            with config_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                msg = "Configuration file is empty"
                raise ValueError(msg)  # noqa: TRY301

            # Convert inputs and outputs from dict to list if needed
            data = self._normalize_config_data(data)

            self._config = GPUXConfig(**data)
            logger.info("Configuration loaded from %s", config_path)
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in configuration file: {e}"
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Failed to parse configuration: {e}"
            raise ValueError(msg) from e
        else:
            return self._config

    def parse_string(self, config_str: str) -> GPUXConfig:
        """Parse a GPUX configuration string.

        Args:
            config_str: Configuration as YAML string

        Returns:
            Parsed configuration

        Raises:
            ValueError: If config string is invalid
        """
        try:
            data = yaml.safe_load(config_str)

            if not data:
                msg = "Configuration string is empty"
                raise ValueError(msg)  # noqa: TRY301

            # Convert inputs and outputs from dict to list if needed
            data = self._normalize_config_data(data)

            self._config = GPUXConfig(**data)
            logger.info("Configuration loaded from string")
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in configuration string: {e}"
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Failed to parse configuration: {e}"
            raise ValueError(msg) from e
        else:
            return self._config

    def _normalize_config_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize configuration data for Pydantic model.

        Args:
            data: Raw configuration data

        Returns:
            Normalized configuration data
        """
        # Convert inputs from dict to list if needed
        if "inputs" in data and isinstance(data["inputs"], dict):
            inputs = []
            for name, config in data["inputs"].items():
                if isinstance(config, dict):
                    config["name"] = name
                    inputs.append(config)
                else:
                    inputs.append({"name": name, "type": str(config)})
            data["inputs"] = inputs

        # Convert outputs from dict to list if needed
        if "outputs" in data and isinstance(data["outputs"], dict):
            outputs = []
            for name, config in data["outputs"].items():
                if isinstance(config, dict):
                    config["name"] = name
                    outputs.append(config)
                else:
                    outputs.append({"name": name, "type": str(config)})
            data["outputs"] = outputs

        return data

    def get_config(self) -> GPUXConfig | None:
        """Get the current configuration.

        Returns:
            Current configuration or None
        """
        return self._config

    def validate_model_path(self, base_path: str | Path | None = None) -> bool:
        """Validate that the model file exists.

        Args:
            base_path: Base path for relative model paths

        Returns:
            True if model file exists
        """
        if self._config is None:
            return False

        model_path = self._config.model.source
        if isinstance(model_path, str):
            model_path = Path(model_path)

        if not model_path.is_absolute() and base_path:
            model_path = Path(base_path) / model_path

        return model_path.exists()

    def get_model_path(self, base_path: str | Path | None = None) -> Path | None:
        """Get the absolute model path.

        Args:
            base_path: Base path for relative model paths

        Returns:
            Absolute model path or None
        """
        if self._config is None:
            return None

        model_path = self._config.model.source
        if isinstance(model_path, str):
            model_path = Path(model_path)

        if not model_path.is_absolute() and base_path:
            model_path = Path(base_path) / model_path

        return model_path.resolve()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        if self._config is None:
            return {}

        return self._config.model_dump()

    def save(self, path: str | Path) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration
        """
        if self._config is None:
            msg = "No configuration to save"
            raise ValueError(msg)

        config_dict = self.to_dict()

        # Convert Path objects to strings for YAML serialization
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        config_dict = convert_paths(config_dict)

        with Path(path).open("w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info("Configuration saved to %s", path)
