"""Core GPUX runtime for ML inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort

from gpux.core.models import ModelInfo, ModelInspector
from gpux.core.providers import ExecutionProvider, ProviderManager

logger = logging.getLogger(__name__)


class GPUXRuntime:
    """Main GPUX runtime for ML inference with universal GPU compatibility."""
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GPUX runtime.
        
        Args:
            model_path: Path to the ONNX model file
            provider: Preferred execution provider (e.g., "cuda", "coreml")
            **kwargs: Additional runtime configuration
        """
        self._model_path: Optional[Path] = None
        self._model_info: Optional[ModelInfo] = None
        self._session: Optional[ort.InferenceSession] = None
        self._provider_manager = ProviderManager()
        self._selected_provider: Optional[ExecutionProvider] = None
        
        # Runtime configuration
        self._config = {
            "memory_limit": kwargs.get("memory_limit", "2GB"),
            "batch_size": kwargs.get("batch_size", 1),
            "timeout": kwargs.get("timeout", 30),
            "enable_profiling": kwargs.get("enable_profiling", False),
            **kwargs,
        }
        
        if model_path:
            self.load_model(model_path, provider)
    
    def load_model(
        self,
        model_path: Union[str, Path],
        provider: Optional[str] = None,
    ) -> None:
        """Load an ONNX model for inference.
        
        Args:
            model_path: Path to the ONNX model file
            provider: Preferred execution provider
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model cannot be loaded
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Inspect the model
            inspector = ModelInspector()
            self._model_info = inspector.inspect(model_path)
            self._model_path = model_path
            
            # Select execution provider
            self._selected_provider = self._provider_manager.get_best_provider(provider)
            provider_config = self._provider_manager.get_provider_config(self._selected_provider)
            
            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.enable_profiling = self._config["enable_profiling"]
            
            # Set memory limit if specified
            if "memory_limit" in self._config:
                memory_limit = self._parse_memory_limit(self._config["memory_limit"])
                session_options.add_session_config_entry("session.memory_limit", str(memory_limit))
            
            # Create session with selected provider
            providers = [self._selected_provider.value]
            if provider_config:
                providers = [(self._selected_provider.value, provider_config)]
            
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
            
            logger.info(f"Model loaded successfully: {model_path}")
            logger.info(f"Using provider: {self._selected_provider.value}")
            logger.info(f"Model info: {self._model_info.name} v{self._model_info.version}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def infer(
        self,
        input_data: Union[Dict[str, np.ndarray], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Run inference on input data.
        
        Args:
            input_data: Input data as dictionary or single array
            
        Returns:
            Dictionary of output arrays
            
        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if self._session is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            # Prepare input data
            if isinstance(input_data, np.ndarray):
                # Single input case
                if len(self._model_info.inputs) != 1:
                    raise ValueError(f"Expected {len(self._model_info.inputs)} inputs, got 1")
                input_dict = {self._model_info.inputs[0].name: input_data}
            else:
                input_dict = input_data
            
            # Validate input
            if not self._validate_input(input_dict):
                raise ValueError("Input validation failed")
            
            # Run inference
            start_time = time.time()
            outputs = self._session.run(None, input_dict)
            inference_time = time.time() - start_time
            
            # Prepare output dictionary
            output_dict = {}
            for i, output_meta in enumerate(self._model_info.outputs):
                output_dict[output_meta.name] = outputs[i]
            
            # Log performance
            logger.debug(f"Inference completed in {inference_time:.4f}s")
            
            return output_dict
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}") from e
    
    def batch_infer(
        self,
        batch_data: List[Union[Dict[str, np.ndarray], np.ndarray]],
        **kwargs: Any,
    ) -> List[Dict[str, np.ndarray]]:
        """Run batch inference on multiple inputs.
        
        Args:
            batch_data: List of input data
            **kwargs: Additional inference parameters
            
        Returns:
            List of output dictionaries
        """
        results = []
        
        for i, input_data in enumerate(batch_data):
            try:
                result = self.infer(input_data, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch inference failed for item {i}: {e}")
                raise
        
        return results
    
    def benchmark(
        self,
        input_data: Union[Dict[str, np.ndarray], np.ndarray],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            input_data: Input data for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Performance metrics
        """
        if self._session is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        logger.info(f"Benchmarking model with {num_runs} runs (warmup: {warmup_runs})")
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                self.infer(input_data)
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
        
        # Benchmark runs
        times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                self.infer(input_data)
                times.append(time.time() - start_time)
            except Exception as e:
                logger.error(f"Benchmark run {i} failed: {e}")
                raise
        
        # Calculate statistics
        times = np.array(times)
        metrics = {
            "mean_time_ms": float(np.mean(times) * 1000),
            "std_time_ms": float(np.std(times) * 1000),
            "min_time_ms": float(np.min(times) * 1000),
            "max_time_ms": float(np.max(times) * 1000),
            "median_time_ms": float(np.median(times) * 1000),
            "p95_time_ms": float(np.percentile(times, 95) * 1000),
            "p99_time_ms": float(np.percentile(times, 99) * 1000),
            "throughput_fps": float(1.0 / np.mean(times)),
        }
        
        logger.info(f"Benchmark results: {metrics['mean_time_ms']:.2f}ms ± {metrics['std_time_ms']:.2f}ms")
        
        return metrics
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get model information.
        
        Returns:
            Model information or None if no model loaded
        """
        return self._model_info
    
    def get_provider_info(self) -> Optional[Dict[str, Any]]:
        """Get current execution provider information.
        
        Returns:
            Provider information or None if no provider selected
        """
        if self._selected_provider is None:
            return None
        
        return self._provider_manager.get_provider_info(self._selected_provider)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available execution providers.
        
        Returns:
            List of available provider names
        """
        return self._provider_manager.get_available_providers()
    
    def _validate_input(self, input_data: Dict[str, np.ndarray]) -> bool:
        """Validate input data against model specifications."""
        if self._model_info is None:
            return False
        
        try:
            # Check if all required inputs are provided
            required_inputs = {inp.name for inp in self._model_info.inputs}
            provided_inputs = set(input_data.keys())
            
            if required_inputs != provided_inputs:
                missing = required_inputs - provided_inputs
                extra = provided_inputs - required_inputs
                logger.error(f"Input mismatch. Missing: {missing}, Extra: {extra}")
                return False
            
            # Check input shapes and types
            for input_spec in self._model_info.inputs:
                if input_spec.name not in input_data:
                    continue
                
                data = input_data[input_spec.name]
                
                # Check type
                if input_spec.type != "unknown" and data.dtype.name != input_spec.type:
                    logger.warning(f"Type mismatch for {input_spec.name}: expected {input_spec.type}, got {data.dtype.name}")
                
                # Check shape (if specified)
                if input_spec.shape:
                    expected_shape = input_spec.shape
                    actual_shape = list(data.shape)
                    
                    # Handle dynamic dimensions (represented as -1 or 0)
                    for i, (exp, act) in enumerate(zip(expected_shape, actual_shape)):
                        if exp > 0 and exp != act:
                            logger.error(f"Shape mismatch for {input_spec.name} at dim {i}: expected {exp}, got {act}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _parse_memory_limit(self, memory_str: str) -> int:
        """Parse memory limit string to bytes.
        
        Args:
            memory_str: Memory string (e.g., "2GB", "512MB")
            
        Returns:
            Memory limit in bytes
        """
        memory_str = memory_str.upper().strip()
        
        if memory_str.endswith("GB"):
            return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
        elif memory_str.endswith("MB"):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        elif memory_str.endswith("KB"):
            return int(float(memory_str[:-2]) * 1024)
        else:
            # Assume bytes
            return int(memory_str)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._session is not None:
            self._session = None
        self._model_info = None
        self._model_path = None
        self._selected_provider = None
        logger.info("Runtime cleaned up")
    
    def __enter__(self) -> GPUXRuntime:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()
