"""Integration tests for end-to-end GPUX workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gpux.core.conversion.pytorch import PyTorchConverter
from gpux.core.conversion.tensorflow import TensorFlowConverter
from gpux.core.conversion.optimizer import ConversionError
from gpux.core.managers.base import ModelMetadata, RegistryConfig
from gpux.core.managers.huggingface import HuggingFaceManager


class TestEndToEndWorkflow:
    """Test complete GPUX workflow from pull to run."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = RegistryConfig(
            name="huggingface",
            api_url="https://huggingface.co",
            cache_dir=self.temp_dir,
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pytorch_model_workflow(self) -> None:
        """Test complete workflow for PyTorch model."""
        # Mock HuggingFaceManager
        _ = HuggingFaceManager(self.config)

        # Mock model metadata for PyTorch model
        metadata = ModelMetadata(
            registry="huggingface",
            model_id="microsoft/DialoGPT-medium",
            revision="main",
            format="pytorch",
            files={
                "pytorch_model.bin": self.temp_dir / "pytorch_model.bin",
                "config.json": self.temp_dir / "config.json",
                "tokenizer.json": self.temp_dir / "tokenizer.json",
            },
            size_bytes=1024 * 1024 * 500,  # 500MB
            description="DialoGPT medium model",
            tags=["text-generation", "gpt", "conversational"],
        )

        # Test PyTorch converter
        converter = PyTorchConverter(cache_dir=self.temp_dir)

        # Verify converter can handle this model
        assert converter.can_convert(metadata)

        # Test input/output shape detection
        input_shapes = converter.get_input_shapes(metadata)
        output_shapes = converter.get_output_shapes(metadata)

        # Should have some shapes detected
        assert isinstance(input_shapes, dict)
        assert isinstance(output_shapes, dict)

    def test_tensorflow_model_workflow(self) -> None:
        """Test complete workflow for TensorFlow model."""
        # Mock model metadata for TensorFlow model
        metadata = ModelMetadata(
            registry="huggingface",
            model_id="bert-base-uncased",
            revision="main",
            format="tensorflow",
            files={
                "tf_model.h5": self.temp_dir / "tf_model.h5",
                "config.json": self.temp_dir / "config.json",
                "tokenizer.json": self.temp_dir / "tokenizer.json",
            },
            size_bytes=1024 * 1024 * 400,  # 400MB
            description="BERT base uncased model",
            tags=["text-classification", "bert", "nlp"],
        )

        # Test TensorFlow converter
        try:
            converter = TensorFlowConverter(cache_dir=self.temp_dir)
        except ImportError:
            pytest.skip("TensorFlow not available")

        # Verify converter can handle this model
        assert converter.can_convert(metadata)

        # Test input/output shape detection
        input_shapes = converter.get_input_shapes(metadata)
        output_shapes = converter.get_output_shapes(metadata)

        # Should have some shapes detected
        assert isinstance(input_shapes, dict)
        assert isinstance(output_shapes, dict)

    def test_model_format_detection(self) -> None:
        """Test automatic model format detection."""
        pytorch_metadata = ModelMetadata(
            registry="huggingface",
            model_id="test-pytorch",
            revision="main",
            format="pytorch",
            files={"pytorch_model.bin": Path(self.temp_dir / "model.bin")},
            size_bytes=1024,
        )

        tensorflow_metadata = ModelMetadata(
            registry="huggingface",
            model_id="test-tensorflow",
            revision="main",
            format="tensorflow",
            files={"tf_model.h5": Path(self.temp_dir / "model.h5")},
            size_bytes=1024,
        )

        pytorch_converter = PyTorchConverter()
        try:
            tensorflow_converter = TensorFlowConverter()
        except ImportError:
            pytest.skip("TensorFlow not available")

        # Test format detection
        assert pytorch_converter.can_convert(pytorch_metadata)
        assert not pytorch_converter.can_convert(tensorflow_metadata)

        assert tensorflow_converter.can_convert(tensorflow_metadata)
        assert not tensorflow_converter.can_convert(pytorch_metadata)

    def test_conversion_pipeline_integration(self) -> None:
        """Test integration between different conversion components."""
        # Test that converters can be imported and instantiated
        pytorch_converter = PyTorchConverter()
        try:
            tensorflow_converter = TensorFlowConverter()
        except ImportError:
            pytest.skip("TensorFlow not available")

        # Test that they have the required methods
        assert hasattr(pytorch_converter, "can_convert")
        assert hasattr(pytorch_converter, "convert")
        assert hasattr(pytorch_converter, "get_input_shapes")
        assert hasattr(pytorch_converter, "get_output_shapes")

        assert hasattr(tensorflow_converter, "can_convert")
        assert hasattr(tensorflow_converter, "convert")
        assert hasattr(tensorflow_converter, "get_input_shapes")
        assert hasattr(tensorflow_converter, "get_output_shapes")

    def test_model_metadata_validation(self) -> None:
        """Test model metadata validation."""
        # Valid metadata
        valid_metadata = ModelMetadata(
            registry="huggingface",
            model_id="test-model",
            revision="main",
            format="pytorch",
            files={"model.bin": Path(self.temp_dir / "model.bin")},
            size_bytes=1024,
        )

        assert valid_metadata.registry == "huggingface"
        assert valid_metadata.model_id == "test-model"
        assert valid_metadata.format == "pytorch"
        assert len(valid_metadata.files) == 1

    def test_registry_config_validation(self) -> None:
        """Test registry configuration validation."""
        config = RegistryConfig(
            name="test-registry",
            api_url="https://test.com",
            cache_dir=self.temp_dir,
        )

        assert config.name == "test-registry"
        assert config.api_url == "https://test.com"
        assert config.cache_dir == self.temp_dir

    def test_huggingface_manager_integration(self) -> None:
        """Test HuggingFaceManager integration with conversion pipeline."""
        manager = HuggingFaceManager(self.config)

        # Test that manager can be instantiated
        assert manager.config == self.config
        assert manager.logger is not None

    def test_conversion_error_handling(self) -> None:
        """Test error handling in conversion pipeline."""
        pytorch_converter = PyTorchConverter()

        # Test with invalid metadata
        invalid_metadata = ModelMetadata(
            registry="test",
            model_id="invalid",
            revision="main",
            format="unknown",
            files={"file.txt": Path(self.temp_dir / "file.txt")},
            size_bytes=1024,
        )

        # Should not be able to convert unknown format
        assert not pytorch_converter.can_convert(invalid_metadata)

        # Should raise error when trying to convert
        with pytest.raises(ConversionError):
            pytorch_converter.convert(invalid_metadata)
