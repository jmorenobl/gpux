"""Tests for GPUX runtime functionality."""

from pathlib import Path
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


class TestGPUXRuntimeErrorHandling:
    """Test cases for GPUXRuntime error handling and edge cases."""

    def test_load_model_file_not_found(self):
        """Test load_model with non-existent file."""
        runtime = GPUXRuntime()

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            runtime.load_model("nonexistent_model.onnx")

    def test_load_model_exception_handling(self):
        """Test load_model with exception during loading."""
        runtime = GPUXRuntime()

        # Create a file that exists but is invalid
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"invalid onnx content")
            invalid_model_path = f.name

        try:
            with pytest.raises(RuntimeError, match="Failed to load model"):
                runtime.load_model(invalid_model_path)
        finally:
            Path(invalid_model_path).unlink()

    def test_infer_no_model_loaded(self):
        """Test infer without loading a model."""
        runtime = GPUXRuntime()

        with pytest.raises(RuntimeError, match="No model loaded"):
            runtime.infer({"input": np.array([[1.0, 2.0]], dtype=np.float32)})

    def test_infer_model_info_not_available(self, simple_onnx_model):
        """Test infer with model info not available."""
        runtime = GPUXRuntime(model_path=simple_onnx_model)
        # Manually set model_info to None to test the error path
        runtime._model_info = None

        with pytest.raises(RuntimeError, match="Model info not available"):
            runtime.infer(np.array([[1.0, 2.0]], dtype=np.float32))

    def test_infer_single_input_wrong_count(self, simple_onnx_model):
        """Test infer with single input but wrong input count."""
        runtime = GPUXRuntime(model_path=simple_onnx_model)
        # Mock model_info to have multiple inputs
        runtime._model_info.inputs = [
            runtime._model_info.inputs[0],
            runtime._model_info.inputs[0],  # Duplicate to simulate multiple inputs
        ]

        with pytest.raises(RuntimeError, match="Inference failed"):
            runtime.infer(np.array([[1.0, 2.0]], dtype=np.float32))

    def test_batch_infer_exception_handling(self, runtime, sample_input_data):
        """Test batch_infer with exception handling."""
        # Create invalid input data that will cause inference to fail
        invalid_input = {"invalid": np.array([[1.0, 2.0]], dtype=np.float32)}
        batch_data = [sample_input_data, invalid_input]

        with pytest.raises(RuntimeError):
            runtime.batch_infer(batch_data)

    def test_benchmark_no_model_loaded(self):
        """Test benchmark without loading a model."""
        runtime = GPUXRuntime()

        with pytest.raises(RuntimeError, match="No model loaded"):
            runtime.benchmark({"input": np.array([[1.0, 2.0]], dtype=np.float32)})

    def test_benchmark_warmup_failure(self, runtime, sample_input_data):
        """Test benchmark with warmup failure."""
        # Mock infer to fail during warmup
        original_infer = runtime.infer
        call_count = 0

        def failing_infer(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail during warmup
                msg = "Warmup failed"
                raise RuntimeError(msg)
            return original_infer(*args, **kwargs)

        runtime.infer = failing_infer

        # Should not raise exception, just log warning
        result = runtime.benchmark(sample_input_data, warmup_runs=2, num_runs=1)
        assert "mean_time_ms" in result

    def test_benchmark_run_failure(self, runtime, sample_input_data):
        """Test benchmark with run failure."""
        # Mock infer to fail during benchmark runs
        original_infer = runtime.infer
        call_count = 0

        def failing_infer(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:  # Fail during benchmark runs
                msg = "Benchmark run failed"
                raise RuntimeError(msg)
            return original_infer(*args, **kwargs)

        runtime.infer = failing_infer

        with pytest.raises(RuntimeError, match="Benchmark run failed"):
            runtime.benchmark(sample_input_data, warmup_runs=2, num_runs=1)

    def test_get_provider_info_no_provider(self):
        """Test get_provider_info with no provider selected."""
        runtime = GPUXRuntime()
        runtime._selected_provider = None

        result = runtime.get_provider_info()
        assert result is None

    def test_validate_input_no_model_info(self):
        """Test _validate_input with no model info."""
        runtime = GPUXRuntime()
        runtime._model_info = None

        result = runtime._validate_input(
            {"input": np.array([[1.0, 2.0]], dtype=np.float32)}
        )
        assert result is False

    def test_validate_input_missing_inputs(self, runtime):
        """Test _validate_input with missing inputs."""
        # Create input data with missing required input
        invalid_input = {"wrong_input": np.array([[1.0, 2.0]], dtype=np.float32)}

        result = runtime._validate_input(invalid_input)
        assert result is False

    def test_validate_input_extra_inputs(self, runtime):
        """Test _validate_input with extra inputs."""
        # Create input data with extra input
        extra_input = {
            "input": np.array([[1.0, 2.0]], dtype=np.float32),
            "extra_input": np.array([[1.0, 2.0]], dtype=np.float32),
        }

        result = runtime._validate_input(extra_input)
        assert result is False

    def test_validate_input_type_mismatch(self, runtime):
        """Test _validate_input with type mismatch."""
        # Create input data with wrong type
        wrong_type_input = {"input": np.array([[1.0, 2.0]], dtype=np.int32)}

        result = runtime._validate_input(wrong_type_input)
        # Should still return True but log warning
        assert result is True

    def test_validate_input_shape_mismatch(self, runtime):
        """Test _validate_input with shape mismatch."""
        # Create input data with wrong shape
        wrong_shape_input = {"input": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}

        result = runtime._validate_input(wrong_shape_input)
        assert result is False

    def test_validate_input_exception_handling(self, runtime):
        """Test _validate_input with exception during validation."""
        # Create input data that will cause an exception
        runtime._model_info.inputs[0].shape = [1, 2]  # Set specific shape

        # Create input with wrong shape that will cause exception
        invalid_input = {"input": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}

        result = runtime._validate_input(invalid_input)
        assert result is False

    def test_parse_memory_limit(self, runtime):
        """Test _parse_memory_limit method."""
        # Test valid memory limit
        result = runtime._parse_memory_limit("1GB")
        assert result == 1024 * 1024 * 1024

        result = runtime._parse_memory_limit("512MB")
        assert result == 512 * 1024 * 1024

        result = runtime._parse_memory_limit("1024KB")
        assert result == 1024 * 1024

        result = runtime._parse_memory_limit("2048")
        assert result == 2048

    def test_parse_memory_limit_invalid_format(self, runtime):
        """Test _parse_memory_limit with invalid format."""
        with pytest.raises(ValueError, match="invalid literal for int"):
            runtime._parse_memory_limit("invalid")

    def test_parse_memory_limit_invalid_unit(self, runtime):
        """Test _parse_memory_limit with invalid unit."""
        with pytest.raises(ValueError, match="invalid literal for int"):
            runtime._parse_memory_limit("1024TB")

    def test_parse_memory_limit_invalid_number(self, runtime):
        """Test _parse_memory_limit with invalid number."""
        with pytest.raises(ValueError, match="could not convert string to float"):
            runtime._parse_memory_limit("abcGB")
