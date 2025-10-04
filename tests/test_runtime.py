"""Tests for GPUX runtime functionality."""

from typing import Any

import numpy as np
import pytest
from gpux.core.models import ModelInspector
from gpux.core.runtime import GPUXRuntime


class TestGPUXRuntime:
    """Test cases for GPUXRuntime class."""

    def test_runtime_initialization(self, simple_onnx_model):
        """Test runtime initialization."""
        runtime = GPUXRuntime(model_path=simple_onnx_model)

        assert runtime._model_path == simple_onnx_model
        assert runtime._session is not None
        assert runtime._model_info is not None
        assert runtime._selected_provider is not None

        runtime.cleanup()

    def test_runtime_context_manager(self, simple_onnx_model):
        """Test runtime as context manager."""
        with GPUXRuntime(model_path=simple_onnx_model) as runtime:
            assert runtime._session is not None
            assert runtime._model_info is not None

        # Should be cleaned up after context
        assert runtime._session is None
        assert runtime._model_info is None

    def test_inference(self, runtime, sample_input_data, expected_output_data):
        """Test model inference."""
        results = runtime.infer(sample_input_data)

        assert "output" in results
        np.testing.assert_array_almost_equal(
            results["output"], expected_output_data["output"], decimal=5
        )

    def test_batch_inference(self, runtime, sample_input_data, expected_output_data):
        """Test batch inference."""
        batch_data = [sample_input_data, sample_input_data]
        results = runtime.batch_infer(batch_data)

        assert len(results) == 2
        for result in results:
            assert "output" in result
            np.testing.assert_array_almost_equal(
                result["output"], expected_output_data["output"], decimal=5
            )

    def test_benchmark(self, runtime, sample_input_data):
        """Test model benchmarking."""
        metrics = runtime.benchmark(sample_input_data, num_runs=5, warmup_runs=2)

        assert "mean_time_ms" in metrics
        assert "std_time_ms" in metrics
        assert "throughput_fps" in metrics
        assert metrics["mean_time_ms"] > 0
        assert metrics["throughput_fps"] > 0

    def test_get_model_info(self, runtime):
        """Test getting model information."""
        model_info = runtime.get_model_info()

        assert model_info is not None
        assert model_info.name == "simple_model"
        assert model_info.format == "onnx"
        assert len(model_info.inputs) == 1
        assert len(model_info.outputs) == 1

    def test_get_provider_info(self, runtime):
        """Test getting provider information."""
        provider_info = runtime.get_provider_info()

        assert provider_info is not None
        assert "name" in provider_info
        assert "available" in provider_info
        assert provider_info["available"] is True

    def test_get_available_providers(self, runtime):
        """Test getting available providers."""
        providers = runtime.get_available_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "CPUExecutionProvider" in providers

    def test_invalid_input_validation(self, runtime):
        """Test input validation with invalid data."""
        # Missing required input
        invalid_input: dict[str, Any] = {}

        with pytest.raises(RuntimeError, match="Input validation failed"):
            runtime.infer(invalid_input)

        # Wrong input name
        wrong_input = {"wrong_name": np.array([[1.0, 2.0]], dtype=np.float32)}

        with pytest.raises(RuntimeError, match="Input validation failed"):
            runtime.infer(wrong_input)

    def test_memory_limit_parsing(self, runtime):
        """Test memory limit parsing."""
        # Test GB
        assert runtime._parse_memory_limit("2GB") == 2 * 1024 * 1024 * 1024

        # Test MB
        assert runtime._parse_memory_limit("512MB") == 512 * 1024 * 1024

        # Test KB
        assert runtime._parse_memory_limit("1024KB") == 1024 * 1024

        # Test bytes
        assert runtime._parse_memory_limit("1024") == 1024


class TestModelInspector:
    """Test cases for ModelInspector class."""

    def test_inspect_model(self, simple_onnx_model):
        """Test model inspection."""
        inspector = ModelInspector()
        model_info = inspector.inspect(simple_onnx_model)

        assert model_info.name == "simple_model"
        assert model_info.format == "onnx"
        assert model_info.size_bytes > 0
        assert len(model_info.inputs) == 1
        assert len(model_info.outputs) == 1

        # Check input specification
        input_spec = model_info.inputs[0]
        assert input_spec.name == "input"
        assert input_spec.type == "float32"
        assert input_spec.shape == [1, 2]

        # Check output specification
        output_spec = model_info.outputs[0]
        assert output_spec.name == "output"
        assert output_spec.type == "float32"
        assert output_spec.shape == [1, 2]

    def test_inspect_nonexistent_model(self, temp_dir):
        """Test inspecting non-existent model."""
        inspector = ModelInspector()
        nonexistent_model = temp_dir / "nonexistent.onnx"

        with pytest.raises(FileNotFoundError):
            inspector.inspect(nonexistent_model)

    def test_validate_input(self, simple_onnx_model, sample_input_data):
        """Test input validation."""
        inspector = ModelInspector()
        inspector._session = inspector._create_session(simple_onnx_model)

        # Valid input
        assert inspector.validate_input(sample_input_data) is True

        # Invalid input - missing required input
        invalid_input: dict[str, Any] = {}
        assert inspector.validate_input(invalid_input) is False

        # Invalid input - wrong input name
        wrong_input = {"wrong_name": np.array([[1.0, 2.0]], dtype=np.float32)}
        assert inspector.validate_input(wrong_input) is False
