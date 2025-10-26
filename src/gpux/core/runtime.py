"""Core GPUX runtime for ML inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import onnxruntime as ort

from gpux.core.models import ModelInfo, ModelInspector
from gpux.core.providers import ExecutionProvider, ProviderManager

if TYPE_CHECKING:
    import types

logger = logging.getLogger(__name__)


class GPUXRuntime:
    """Main GPUX runtime for ML inference with universal GPU compatibility."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GPUX runtime.

        Args:
            model_path: Path to the ONNX model file
            provider: Preferred execution provider (e.g., "cuda", "coreml")
            **kwargs: Additional runtime configuration
        """
        self._model_path: Path | None = None
        self._model_info: ModelInfo | None = None
        self._session: ort.InferenceSession | None = None
        self._provider_manager = ProviderManager()
        self._selected_provider: ExecutionProvider | None = None

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
        model_path: str | Path,
        provider: str | None = None,
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
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        try:
            # Inspect the model
            inspector = ModelInspector()
            self._model_info = inspector.inspect(model_path)
            self._model_path = model_path

            # Select execution provider
            self._selected_provider = self._provider_manager.get_best_provider(provider)
            provider_config = self._provider_manager.get_provider_config(
                self._selected_provider
            )

            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.enable_profiling = self._config["enable_profiling"]

            # Set memory limit if specified
            if "memory_limit" in self._config:
                memory_limit = self._parse_memory_limit(self._config["memory_limit"])
                session_options.add_session_config_entry(
                    "session.memory_limit", str(memory_limit)
                )

            # Create session with selected provider
            providers: list[str | tuple[str, dict[str, Any]]] = [
                self._selected_provider.value
            ]
            if provider_config:
                providers = [(self._selected_provider.value, provider_config)]

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )

            logger.info("Model loaded successfully: %s", model_path)
            logger.info("Using provider: %s", self._selected_provider.value)
            logger.info(
                "Model info: %s v%s",
                self._model_info.name,
                self._model_info.version,
            )

        except Exception as e:
            msg = f"Failed to load model: {e}"
            raise RuntimeError(msg) from e

    def infer(
        self,
        input_data: dict[str, np.ndarray] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run inference on input data.

        Args:
            input_data: Input data as dictionary or single array

        Returns:
            Dictionary of output arrays

        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if self._session is None:
            msg = "No model loaded. Call load_model() first."
            raise RuntimeError(msg)

        try:
            # Prepare input data
            if isinstance(input_data, np.ndarray):
                # Single input case
                if self._model_info is None:
                    msg = "Model info not available"
                    raise RuntimeError(msg)  # noqa: TRY301
                if len(self._model_info.inputs) != 1:
                    msg = f"Expected {len(self._model_info.inputs)} inputs, got 1"
                    raise ValueError(msg)  # noqa: TRY301
                input_dict = {self._model_info.inputs[0].name: input_data}
            else:
                input_dict = input_data

            # Validate input
            if not self._validate_input(input_dict):
                msg = "Input validation failed"
                raise ValueError(msg)  # noqa: TRY301

            # Run inference
            start_time = time.perf_counter()
            outputs = self._session.run(None, input_dict)
            inference_time = time.perf_counter() - start_time

            # Prepare output dictionary
            output_dict = {}
            if self._model_info is not None:
                for i, output_meta in enumerate(self._model_info.outputs):
                    output_dict[output_meta.name] = outputs[i]

            # Log performance
            logger.debug("Inference completed in %.4fs", inference_time)
        except Exception as e:
            msg = f"Inference failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return output_dict

    def batch_infer(
        self,
        batch_data: list[dict[str, np.ndarray] | np.ndarray],
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray]]:
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
            except Exception:
                logger.exception("Batch inference failed for item %d", i)
                raise

        return results

    def benchmark(
        self,
        input_data: dict[str, np.ndarray] | np.ndarray,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> dict[str, float]:
        """Benchmark model performance.

        Args:
            input_data: Input data for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Performance metrics
        """
        if self._session is None:
            msg = "No model loaded. Call load_model() first."
            raise RuntimeError(msg)

        logger.info(
            "Benchmarking model with %d runs (warmup: %d)", num_runs, warmup_runs
        )

        # Warmup runs
        for _ in range(warmup_runs):
            try:
                self.infer(input_data)
            except (RuntimeError, ValueError) as e:
                logger.warning("Warmup run failed: %s", e)

        # Benchmark runs
        times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            try:
                self.infer(input_data)
                dt = time.perf_counter() - start_time
                # Clamp to timer resolution to avoid zero durations on some platforms
                if dt <= 0.0:
                    dt = 1e-9
                times.append(dt)
            except Exception:
                logger.exception("Benchmark run %d failed", i)
                raise

        # Calculate statistics
        times_array = np.array(times)
        mean_sec = float(np.mean(times_array)) if times_array.size else 0.0
        if mean_sec <= 0.0:
            mean_sec = 1e-9
        metrics = {
            "mean_time_ms": float(mean_sec * 1000),
            "std_time_ms": float(np.std(times_array) * 1000),
            "min_time_ms": float(np.min(times_array) * 1000),
            "max_time_ms": float(np.max(times_array) * 1000),
            "median_time_ms": float(np.median(times_array) * 1000),
            "p95_time_ms": float(np.percentile(times_array, 95) * 1000),
            "p99_time_ms": float(np.percentile(times_array, 99) * 1000),
            "throughput_fps": float(1.0 / mean_sec),
        }

        logger.info(
            "Benchmark results: %.2fms ± %.2fms",
            metrics["mean_time_ms"],
            metrics["std_time_ms"],
        )

        return metrics

    def get_model_info(self) -> ModelInfo | None:
        """Get model information.

        Returns:
            Model information or None if no model loaded
        """
        return self._model_info

    def get_provider_info(self) -> dict[str, Any] | None:
        """Get current execution provider information.

        Returns:
            Provider information or None if no provider selected
        """
        if self._selected_provider is None:
            return None

        return self._provider_manager.get_provider_info(self._selected_provider)

    def get_available_providers(self) -> list[str]:
        """Get list of available execution providers.

        Returns:
            List of available provider names
        """
        return self._provider_manager.get_available_providers()

    def _validate_input(self, input_data: dict[str, np.ndarray]) -> bool:  # noqa: C901
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
                logger.error("Input mismatch. Missing: %s, Extra: %s", missing, extra)
                return False

            # Check input shapes and types
            for input_spec in self._model_info.inputs:
                if input_spec.name not in input_data:
                    continue

                data = input_data[input_spec.name]

                # Check type
                if input_spec.type not in ("unknown", data.dtype.name):
                    logger.warning(
                        "Type mismatch for %s: expected %s, got %s",
                        input_spec.name,
                        input_spec.type,
                        data.dtype.name,
                    )

                # Check shape (if specified)
                if input_spec.shape:
                    expected_shape = input_spec.shape
                    actual_shape = list(data.shape)

                    # Handle dynamic dimensions (represented as -1, 0, or string names)
                    for i, (exp, act) in enumerate(
                        zip(expected_shape, actual_shape, strict=True)
                    ):
                        # Skip validation for dynamic dimensions
                        if isinstance(exp, str) or (isinstance(exp, int) and exp <= 0):
                            continue

                        if exp != act:
                            logger.error(
                                "Shape mismatch for %s at dim %d: expected %d, got %d",
                                input_spec.name,
                                i,
                                exp,
                                act,
                            )
                            return False
        except (ValueError, TypeError, AttributeError):
            logger.exception("Input validation failed")
            return False
        else:
            return True

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
        if memory_str.endswith("MB"):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        if memory_str.endswith("KB"):
            return int(float(memory_str[:-2]) * 1024)
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

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.cleanup()
