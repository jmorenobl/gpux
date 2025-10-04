"""Model information and metadata for GPUX."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class InputSpec:
    """Specification for model input."""
    
    name: str
    type: str
    shape: List[int]
    required: bool = True
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InputSpec:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OutputSpec:
    """Specification for model output."""
    
    name: str
    type: str
    shape: List[int]
    labels: Optional[List[str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OutputSpec:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelInfo:
    """Model information and metadata."""
    
    name: str
    version: str
    format: str
    path: Path
    size_bytes: int
    inputs: List[InputSpec]
    outputs: List[OutputSpec]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "inputs": [inp.to_dict() for inp in self.inputs],
            "outputs": [out.to_dict() for out in self.outputs],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelInfo:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            format=data["format"],
            path=Path(data["path"]),
            size_bytes=data["size_bytes"],
            inputs=[InputSpec.from_dict(inp) for inp in data["inputs"]],
            outputs=[OutputSpec.from_dict(out) for out in data["outputs"]],
            metadata=data["metadata"],
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model info to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> ModelInfo:
        """Load model info from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ModelInspector:
    """Inspect ONNX models and extract metadata."""
    
    def __init__(self) -> None:
        """Initialize the model inspector."""
        self._session: Optional[ort.InferenceSession] = None
    
    def inspect(self, model_path: Union[str, Path]) -> ModelInfo:
        """Inspect an ONNX model and extract information.
        
        Args:
            model_path: Path to the ONNX model file
            
        Returns:
            Model information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model cannot be loaded
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load the model
            self._session = ort.InferenceSession(str(model_path))
            
            # Extract basic information
            name = model_path.stem
            version = "1.0.0"  # Default version
            format_type = "onnx"
            size_bytes = model_path.stat().st_size
            
            # Extract input specifications
            inputs = []
            for input_meta in self._session.get_inputs():
                input_spec = InputSpec(
                    name=input_meta.name,
                    type=self._get_numpy_type_name(input_meta.type),
                    shape=list(input_meta.shape) if input_meta.shape else [],
                    required=True,
                )
                inputs.append(input_spec)
            
            # Extract output specifications
            outputs = []
            for output_meta in self._session.get_outputs():
                output_spec = OutputSpec(
                    name=output_meta.name,
                    type=self._get_numpy_type_name(output_meta.type),
                    shape=list(output_meta.shape) if output_meta.shape else [],
                )
                outputs.append(output_spec)
            
            # Extract metadata
            metadata = self._extract_metadata()
            
            return ModelInfo(
                name=name,
                version=version,
                format=format_type,
                path=model_path,
                size_bytes=size_bytes,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to inspect model: {e}") from e
        finally:
            self._session = None
    
    def _get_numpy_type_name(self, onnx_type: str) -> str:
        """Convert ONNX type to numpy type name."""
        type_mapping = {
            "tensor(float)": "float32",
            "tensor(double)": "float64",
            "tensor(int32)": "int32",
            "tensor(int64)": "int64",
            "tensor(bool)": "bool",
            "tensor(string)": "string",
        }
        return type_mapping.get(onnx_type, "unknown")
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the ONNX model."""
        metadata = {}
        
        if self._session is None:
            return metadata
        
        try:
            # Get model metadata
            model_metadata = self._session.get_modelmeta()
            if model_metadata:
                metadata.update({
                    "producer_name": model_metadata.producer_name,
                    "producer_version": model_metadata.producer_version,
                    "domain": model_metadata.domain,
                    "model_version": model_metadata.model_version,
                    "doc_string": model_metadata.doc_string,
                })
            
            # Get execution providers
            providers = self._session.get_providers()
            metadata["execution_providers"] = providers
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def validate_input(self, input_data: Dict[str, np.ndarray]) -> bool:
        """Validate input data against model specifications.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            True if input is valid
        """
        if self._session is None:
            return False
        
        try:
            # Check if all required inputs are provided
            required_inputs = {inp.name for inp in self._session.get_inputs()}
            provided_inputs = set(input_data.keys())
            
            if required_inputs != provided_inputs:
                missing = required_inputs - provided_inputs
                extra = provided_inputs - required_inputs
                logger.error(f"Input mismatch. Missing: {missing}, Extra: {extra}")
                return False
            
            # Check input shapes and types
            for input_meta in self._session.get_inputs():
                input_name = input_meta.name
                if input_name not in input_data:
                    continue
                
                data = input_data[input_name]
                
                # Check type
                expected_type = self._get_numpy_type_name(input_meta.type)
                if expected_type != "unknown" and data.dtype.name != expected_type:
                    logger.warning(f"Type mismatch for {input_name}: expected {expected_type}, got {data.dtype.name}")
                
                # Check shape (if specified)
                if input_meta.shape:
                    expected_shape = list(input_meta.shape)
                    actual_shape = list(data.shape)
                    
                    # Handle dynamic dimensions (represented as -1 or 0)
                    for i, (exp, act) in enumerate(zip(expected_shape, actual_shape)):
                        if exp > 0 and exp != act:
                            logger.error(f"Shape mismatch for {input_name} at dim {i}: expected {exp}, got {act}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
