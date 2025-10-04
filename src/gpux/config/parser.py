"""Configuration parser for GPUX files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class InputConfig(BaseModel):
    """Input configuration for GPUX."""
    
    name: str
    type: str
    shape: Optional[List[int]] = None
    required: bool = True
    max_length: Optional[int] = None
    description: Optional[str] = None


class OutputConfig(BaseModel):
    """Output configuration for GPUX."""
    
    name: str
    type: str
    shape: Optional[List[int]] = None
    labels: Optional[List[str]] = None
    description: Optional[str] = None


class ModelConfig(BaseModel):
    """Model configuration for GPUX."""
    
    source: Union[str, Path]
    format: str = "onnx"
    version: Optional[str] = None
    
    @validator("source")
    @classmethod
    def validate_source(cls, v):
        """Validate model source path."""
        if isinstance(v, str):
            return Path(v)
        return v


class GPUConfig(BaseModel):
    """GPU configuration for GPUX."""
    
    memory: str = "2GB"
    backend: str = "auto"  # auto, vulkan, metal, dx12, cuda, rocm, coreml
    
    @validator("memory")
    @classmethod
    def validate_memory(cls, v):
        """Validate memory specification."""
        v = v.upper().strip()
        if not (v.endswith("GB") or v.endswith("MB") or v.endswith("KB")):
            raise ValueError("Memory must be specified as GB, MB, or KB")
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
    host: str = "0.0.0.0"
    batch_size: int = 1
    timeout: int = 5
    max_workers: int = 4


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration for GPUX."""
    
    tokenizer: Optional[str] = None
    max_length: Optional[int] = None
    resize: Optional[List[int]] = None
    normalize: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None


class GPUXConfig(BaseModel):
    """Main GPUX configuration model."""
    
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    
    model: ModelConfig
    inputs: List[InputConfig] = Field(default_factory=list)
    outputs: List[OutputConfig] = Field(default_factory=list)
    
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    serving: Optional[ServingConfig] = None
    preprocessing: Optional[PreprocessingConfig] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("inputs")
    @classmethod
    def validate_inputs(cls, v):
        """Validate inputs configuration."""
        if not v:
            raise ValueError("At least one input must be specified")
        return v
    
    @validator("outputs")
    @classmethod
    def validate_outputs(cls, v):
        """Validate outputs configuration."""
        if not v:
            raise ValueError("At least one output must be specified")
        return v


class GPUXConfigParser:
    """Parser for GPUX configuration files."""
    
    def __init__(self) -> None:
        """Initialize the configuration parser."""
        self._config: Optional[GPUXConfig] = None
    
    def parse_file(self, config_path: Union[str, Path]) -> GPUXConfig:
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
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if not data:
                raise ValueError("Configuration file is empty")
            
            # Convert inputs and outputs from dict to list if needed
            data = self._normalize_config_data(data)
            
            self._config = GPUXConfig(**data)
            logger.info(f"Configuration loaded from {config_path}")
            
            return self._config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse configuration: {e}") from e
    
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
                raise ValueError("Configuration string is empty")
            
            # Convert inputs and outputs from dict to list if needed
            data = self._normalize_config_data(data)
            
            self._config = GPUXConfig(**data)
            logger.info("Configuration loaded from string")
            
            return self._config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration string: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse configuration: {e}") from e
    
    def _normalize_config_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def get_config(self) -> Optional[GPUXConfig]:
        """Get the current configuration.
        
        Returns:
            Current configuration or None
        """
        return self._config
    
    def validate_model_path(self, base_path: Optional[Union[str, Path]] = None) -> bool:
        """Validate that the model file exists.
        
        Args:
            base_path: Base path for relative model paths
            
        Returns:
            True if model file exists
        """
        if self._config is None:
            return False
        
        model_path = self._config.model.source
        
        if not model_path.is_absolute() and base_path:
            model_path = Path(base_path) / model_path
        
        return model_path.exists()
    
    def get_model_path(self, base_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Get the absolute model path.
        
        Args:
            base_path: Base path for relative model paths
            
        Returns:
            Absolute model path or None
        """
        if self._config is None:
            return None
        
        model_path = self._config.model.source
        
        if not model_path.is_absolute() and base_path:
            model_path = Path(base_path) / model_path
        
        return model_path.resolve()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        if self._config is None:
            return {}
        
        return self._config.dict()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        if self._config is None:
            raise ValueError("No configuration to save")
        
        config_dict = self.to_dict()
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {path}")
